"""
MSS Topic-Aware Diversity: 结构化相似度替代 Cosine Similarity
============================================================
核心改变:
  Cosine sim 在医学卡片上方差只有 0.019，MMR 退化为纯 base score 排序。
  本脚本用 **专科饱和度 + 聚类饱和度** 替代 cosine sim 作为冗余惩罚：

  Penalty(ci, S) = α · SpecSat(ci, S) + β · ClusterSat(ci, S)

  其中:
    SpecSat = min(1, count(spec_i in S) / target_spec_i)     → 该专科已选了多少/目标
    ClusterSat = min(1, count(cluster_i in S) / target_cluster) → 该聚类已选了多少/目标

  MMR 变成:
    c_next = argmax { Score(ci) - λ · Penalty(ci, S) }

  这给出 0→1 的完整动态范围 (vs cosine 的 0.85→0.91)。

评估指标也换掉:
  - Specialty Entropy (归一化)  → 专科分布均匀度
  - Cluster Coverage Ratio      → 覆盖了多少个聚类
  - Specialty Gini              → 分布的不均匀度 (越低越好)
  - Coverage (保留, embedding 级别覆盖仍有意义)
"""

import argparse
import json
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ── 配置 ────────────────────────────────────────────────────────────
INPUT_FILE = "deepseek_cards.json"
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

TIER_MAPPING = {
    "Emergency Medicine": 1.5,
    "Cardiology": 1.5,
    "Neurology": 1.5,
    "Respiratory": 1.5,
    "Gastroenterology": 1.5,
    "Pediatrics": 1.5,
    "ObGyn": 1.2,
    "Infectious Disease": 1.2,
    "Endocrinology": 1.2,
    "Psychiatry": 1.2,
    "General Surgery": 1.2,
    "Nephrology": 1.0,
    "Hematology": 1.0,
    "Orthopedics": 1.0,
    "Dermatology": 1.0,
    "Internal Medicine": 0.8,
    "Family Medicine": 0.8,
    "Other": 0.5,
}
CANDIDATE_LABELS = list(TIER_MAPPING.keys())


# ══════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════
def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)

def minmax(v: np.ndarray) -> np.ndarray:
    return MinMaxScaler().fit_transform(v.reshape(-1, 1)).ravel()


# ══════════════════════════════════════════════════════════════════════
# Base Score
# ══════════════════════════════════════════════════════════════════════
def build_base_score(
    model: SentenceTransformer, emb: np.ndarray,
    w_tier: float, w_rep: float, w_unc: float,
    rep_k: int = 10, batch_size: int = 1024,
) -> Tuple[np.ndarray, List[str]]:
    n = emb.shape[0]
    label_emb = l2n(np.asarray(model.encode(CANDIDATE_LABELS), dtype=np.float32))
    score_mat = emb @ label_emb.T
    top_idx = np.argmax(score_mat, axis=1)
    pred_labels = [CANDIDATE_LABELS[i] for i in top_idx]
    tier = np.array([TIER_MAPPING.get(CANDIDATE_LABELS[i], 0.5) for i in top_idx], dtype=np.float32)

    k = min(rep_k, n - 1)
    rep = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sim_block = emb[start:end] @ emb.T
        for i in range(end - start):
            sim_block[i, start + i] = -1.0
        rep[start:end] = np.mean(np.sort(sim_block, axis=1)[:, -k:], axis=1)
        if end % 4096 == 0 or end == n:
            print(f"    rep: {end}/{n}")

    sorted_scores = np.sort(score_mat, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    unc = 1.0 - minmax(margin)

    base = w_tier * minmax(tier) + w_rep * minmax(rep) + w_unc * minmax(unc)
    return minmax(base), pred_labels


# ══════════════════════════════════════════════════════════════════════
# 全池聚类
# ══════════════════════════════════════════════════════════════════════
def cluster_pool(emb: np.ndarray, n_clusters: int) -> np.ndarray:
    print(f"  K-Means clustering (n={n_clusters})...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(emb)


# ══════════════════════════════════════════════════════════════════════
# Topic-Aware MMR (核心新算法)
# ══════════════════════════════════════════════════════════════════════
def topic_mmr(
    base: np.ndarray,
    pred_labels: List[str],
    cluster_ids: np.ndarray,
    max_k: int,
    mmr_lambda: float,
    alpha_spec: float = 0.6,
    beta_cluster: float = 0.4,
) -> List[int]:
    """
    Topic-Aware MMR:
      c_next = argmax { Score(ci) - λ · [α·SpecSat(ci,S) + β·ClusterSat(ci,S)] }

    SpecSat = min(1, count_spec_in_S / target_spec)  → 专科饱和度
    ClusterSat = min(1, count_cluster_in_S / target_cluster)  → 聚类饱和度

    全向量化, O(N) per step.
    """
    n = len(pred_labels)

    # ── 整数化标签 ──
    specs = sorted(set(pred_labels))
    spec_to_id = {s: i for i, s in enumerate(specs)}
    spec_ids = np.array([spec_to_id[s] for s in pred_labels], dtype=np.int32)
    n_specs = len(specs)

    cluster_arr = np.array(cluster_ids, dtype=np.int32)
    n_clusters = int(cluster_arr.max()) + 1

    # ── 计算 target (按 tier × sqrt(pool_count) 加权) ──
    pool_counts = Counter(pred_labels)
    weights = {}
    for s in specs:
        tier = TIER_MAPPING.get(s, 0.5)
        weights[s] = tier * np.sqrt(pool_counts[s])
    total_w = sum(weights.values())

    spec_target = np.zeros(n_specs, dtype=np.float32)
    min_per = max(5, max_k // (n_specs * 3))
    for s, sid in spec_to_id.items():
        spec_target[sid] = max(min_per, max_k * weights[s] / total_w)

    cluster_target = max(1.0, max_k / n_clusters)

    # ── 计数器 (向量化) ──
    spec_count = np.zeros(n_specs, dtype=np.float32)
    cluster_count = np.zeros(n_clusters, dtype=np.float32)
    used = np.zeros(n, dtype=bool)

    # ── 初始化: 选 base score 最高的卡 ──
    selected: List[int] = []
    first = int(np.argmax(base))
    selected.append(first)
    used[first] = True
    spec_count[spec_ids[first]] += 1
    cluster_count[cluster_arr[first]] += 1

    log_interval = max(1, max_k // 10)

    while len(selected) < max_k:
        # 向量化 penalty: 每张卡的饱和度惩罚
        spec_sat = np.minimum(1.0, spec_count[spec_ids] / spec_target[spec_ids].clip(min=1))
        cluster_sat = np.minimum(1.0, cluster_count[cluster_arr] / max(1.0, cluster_target))
        penalty = alpha_spec * spec_sat + beta_cluster * cluster_sat

        mmr = base - mmr_lambda * penalty
        mmr[used] = -np.inf

        pick = int(np.argmax(mmr))
        if mmr[pick] == -np.inf:
            break

        selected.append(pick)
        used[pick] = True
        spec_count[spec_ids[pick]] += 1
        cluster_count[cluster_arr[pick]] += 1

        if len(selected) % log_interval == 0:
            n_spec_hit = int(np.sum(spec_count > 0))
            n_clust_hit = int(np.sum(cluster_count > 0))
            print(f"    {len(selected)}/{max_k} | "
                  f"specs={n_spec_hit}/{n_specs} clusters={n_clust_hit}/{n_clusters}")

    return selected


# ══════════════════════════════════════════════════════════════════════
# 新评估指标 (不再用 cosine redundancy)
# ══════════════════════════════════════════════════════════════════════
def gini_coefficient(counts: np.ndarray) -> float:
    """Gini coefficient: 0=完全均匀, 1=完全集中"""
    if len(counts) == 0 or np.sum(counts) == 0:
        return 0.0
    sorted_c = np.sort(counts).astype(float)
    n = len(sorted_c)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_c) / (n * np.sum(sorted_c))) - (n + 1) / n)


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def effective_spec_ratio(probs: np.ndarray) -> float:
    nz = probs[probs > 0]
    if len(nz) == 0:
        return 0.0
    eff = float(np.exp(-np.sum(nz * np.log(nz))))
    return eff / max(len(probs), 1)


def evaluate_topic(
    emb: np.ndarray,
    selected: List[int],
    base: np.ndarray,
    all_labels: List[str],
    all_clusters: np.ndarray,
    total_n_clusters: int,
    pool_spec_dist: Dict[str, int],
    batch_size: int = 2048,
) -> Dict:
    s = np.array(selected, dtype=np.int32)
    k = len(s)
    n = emb.shape[0]

    # 1) Embedding Coverage (保留, 衡量知识覆盖)
    sel_vecs = emb[s]
    max_sims: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = emb[start:end] @ sel_vecs.T
        max_sims.append(np.max(sims, axis=1))
    coverage = float(np.mean(np.concatenate(max_sims)))

    # 2) Specialty Metrics
    mss_labels = [all_labels[i] for i in selected]
    pool_specs = set(pool_spec_dist.keys())
    mss_specs = set(mss_labels)
    spec_coverage = len(mss_specs) / max(len(pool_specs), 1)

    spec_counts = Counter(mss_labels)
    counts_arr = np.array([spec_counts.get(s, 0) for s in sorted(pool_specs)], dtype=float)

    # Normalized entropy
    probs = counts_arr / counts_arr.sum()
    probs = probs[probs > 0]
    max_entropy = np.log(len(pool_specs))
    entropy = -float(np.sum(probs * np.log(probs)))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Gini
    gini = gini_coefficient(counts_arr)
    pool_arr = np.array([pool_spec_dist.get(s, 0) for s in sorted(pool_specs)], dtype=float)
    pool_probs = pool_arr / pool_arr.sum() if pool_arr.sum() > 0 else np.zeros_like(pool_arr)
    sel_probs = counts_arr / counts_arr.sum() if counts_arr.sum() > 0 else np.zeros_like(counts_arr)
    tv_to_pool = tv_distance(sel_probs, pool_probs)
    eff_spec_ratio = effective_spec_ratio(sel_probs)

    # 3) Cluster Metrics
    mss_clusters = set(int(all_clusters[i]) for i in selected)
    cluster_coverage = len(mss_clusters) / max(total_n_clusters, 1)

    cluster_counts = Counter(int(all_clusters[i]) for i in selected)
    cluster_arr = np.array(list(cluster_counts.values()), dtype=float)
    cluster_gini = gini_coefficient(cluster_arr)

    # 4) Base score
    avg_base = float(np.mean(base[s]))

    return {
        "emb_coverage": round(coverage, 5),
        "spec_coverage": round(spec_coverage, 4),
        "spec_count": len(mss_specs),
        "spec_entropy": round(norm_entropy, 4),
        "spec_gini": round(gini, 4),
        "tv_to_pool": round(tv_to_pool, 5),
        "effective_spec_ratio": round(eff_spec_ratio, 5),
        "cluster_coverage": round(cluster_coverage, 4),
        "cluster_count": len(mss_clusters),
        "cluster_gini": round(cluster_gini, 4),
        "avg_base_score": round(avg_base, 5),
        "spec_distribution": dict(spec_counts.most_common()),
    }


# ══════════════════════════════════════════════════════════════════════
# CLI & Main
# ══════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Topic-Aware Diversity MSS: structured similarity replaces cosine"
    )
    p.add_argument("--input_file", default=INPUT_FILE)
    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--k_values", type=int, nargs="+", default=[2000, 3000, 5000, 8000])
    p.add_argument("--export_k", type=int, default=3000)
    p.add_argument("--n_pool_clusters", type=int, default=60,
                    help="全池聚类数 (给MMR提供细粒度topic信号)")
    p.add_argument("--mmr_lambdas", type=float, nargs="+",
                    default=[0.50, 0.60, 0.70, 0.80, 0.90])
    p.add_argument("--alpha_spec", type=float, default=0.6,
                    help="专科饱和度权重 (in penalty)")
    p.add_argument("--beta_cluster", type=float, default=0.4,
                    help="聚类饱和度权重 (in penalty)")
    p.add_argument("--w_tier", type=float, default=1.0)
    p.add_argument("--w_rep", type=float, default=1.0)
    p.add_argument("--w_unc", type=float, default=0.8)
    p.add_argument("--w_tv", type=float, default=0.45,
                    help="weight for (1 - TV_to_pool) in final score")
    p.add_argument("--w_eff", type=float, default=0.35,
                    help="weight for effective specialty ratio in final score")
    p.add_argument("--w_base", type=float, default=0.20,
                    help="weight for avg base score in final score")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_results", default="mss_topic_results.json")
    p.add_argument("--output_cards", default="mss_topic_cards.json")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    t0 = time.time()

    # ── 1. 加载 ──
    print("=" * 70)
    print("STEP 1: Load")
    print("=" * 70)
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    N = len(data)
    key_f = "improved_front" if "improved_front" in data[0] else "front"
    key_b = "improved_back" if "improved_back" in data[0] else "back"
    texts = [f"{x.get(key_f, '')} {x.get(key_b, '')}".strip() for x in data]
    print(f"  {N} cards")

    # ── 2. 编码 ──
    print("\n" + "=" * 70)
    print("STEP 2: Encode")
    print("=" * 70)
    model = SentenceTransformer(args.model_name)
    t_enc = time.time()
    emb = l2n(np.asarray(model.encode(texts, show_progress_bar=True), dtype=np.float32))
    print(f"  {emb.shape} ({time.time()-t_enc:.1f}s)")

    # ── 3. Base Score ──
    print("\n" + "=" * 70)
    print("STEP 3: Base Score")
    print("=" * 70)
    t_bs = time.time()
    base, pred_labels = build_base_score(model, emb, args.w_tier, args.w_rep, args.w_unc)
    print(f"  Done ({time.time()-t_bs:.1f}s)")

    pool_dist = Counter(pred_labels)
    print(f"\n  Pool specialty distribution ({len(pool_dist)} specs):")
    for spec, cnt in pool_dist.most_common():
        print(f"    {spec:25s}  {cnt:5d} ({cnt/N*100:5.1f}%)")

    # ── 4. 全池聚类 ──
    print("\n" + "=" * 70)
    print(f"STEP 4: Pool Clustering ({args.n_pool_clusters} clusters)")
    print("=" * 70)
    t_cl = time.time()
    pool_clusters = cluster_pool(emb, args.n_pool_clusters)
    n_total_clusters = int(pool_clusters.max()) + 1
    print(f"  Done ({time.time()-t_cl:.1f}s), {n_total_clusters} clusters")

    # ── 5. Topic-Aware MMR sweep ──
    lambdas = sorted(args.mmr_lambdas)
    k_values = sorted(args.k_values)
    print("\n" + "=" * 70)
    print(f"STEP 5: Topic-Aware MMR")
    print(f"  K:      {k_values}")
    print(f"  λ:      {lambdas}")
    print(f"  α_spec: {args.alpha_spec}, β_cluster: {args.beta_cluster}")
    print("=" * 70)

    all_eval: List[Dict] = []
    all_selections: Dict[str, List[int]] = {}

    for lam in lambdas:
        # 只需跑一次 max_k, 然后 prefix 截取
        max_k = max(k_values)
        print(f"\n  ── λ = {lam:.2f} (max_k={max_k}) ──")
        t_run = time.time()
        ranking = topic_mmr(
            base, pred_labels, pool_clusters,
            max_k, lam, args.alpha_spec, args.beta_cluster,
        )
        print(f"  Done ({time.time()-t_run:.1f}s)")

        for k in k_values:
            sel = ranking[:k]
            key = f"{lam:.2f}_{k}"
            all_selections[key] = sel

            met = evaluate_topic(emb, sel, base, pred_labels,
                                 pool_clusters, n_total_clusters, dict(pool_dist))
            met["lambda"] = lam
            met["k"] = k
            met["ratio_pct"] = round(k / N * 100, 2)
            all_eval.append(met)

    # ── 6. 结果表 ──
    print("\n" + "=" * 70)
    print("STEP 6: Results")
    print("=" * 70)
    header = (f"  {'λ':>4}  {'K':>5}  {'%':>5}  {'EmbCov':>6}  "
              f"{'SpEnt':>5}  {'SpGini':>6}  {'SpCov':>5}  "
              f"{'ClCov':>5}  {'ClGini':>6}  {'Base':>6}  {'Sp':>3}")
    print(header)
    print("  " + "-" * 72)
    for e in all_eval:
        print(f"  {e['lambda']:>4.2f}  {e['k']:>5d}  {e['ratio_pct']:>4.1f}%  "
              f"{e['emb_coverage']:>6.4f}  "
              f"{e['spec_entropy']:>5.3f}  {e['spec_gini']:>6.3f}  "
              f"{e['spec_coverage']:>5.3f}  "
              f"{e['cluster_coverage']:>5.3f}  {e['cluster_gini']:>6.3f}  "
              f"{e['avg_base_score']:>6.4f}  {e['spec_count']:>3d}")

    # ── 7. 找最佳 ──
    # 综合: 高 (1-TV_to_pool) + 高 effective_spec_ratio + 高 avg_base_score
    export_k = args.export_k
    candidates = [e for e in all_eval if e["k"] == export_k]
    if not candidates:
        closest_k = min(k_values, key=lambda x: abs(x - export_k))
        candidates = [e for e in all_eval if e["k"] == closest_k]
        export_k = closest_k

    for e in candidates:
        e["selection_score"] = (
            args.w_tv * (1.0 - e["tv_to_pool"])
            + args.w_eff * e["effective_spec_ratio"]
            + args.w_base * e["avg_base_score"]
        )
    candidates.sort(key=lambda e: e["selection_score"], reverse=True)
    best = candidates[0]
    best_lam = best["lambda"]
    best_key = f"{best_lam:.2f}_{export_k}"

    print(f"\n  ★ Best: λ={best_lam}, K={export_k}")
    print(f"    Score={best['selection_score']:.4f}  TV={best['tv_to_pool']:.4f}  "
          f"EffSpec={best['effective_spec_ratio']:.4f}  Base={best['avg_base_score']:.4f}  "
          f"Specs={best['spec_count']}")

    print(f"\n  Specialty distribution:")
    for spec, cnt in best["spec_distribution"].items():
        pct = cnt / export_k * 100
        bar = "█" * max(1, int(pct * 0.6))
        print(f"    {spec:25s}  {cnt:5d} ({pct:5.1f}%)  {bar}")

    # ── 8. 导出 ──
    print("\n" + "=" * 70)
    print("STEP 7: Export")
    print("=" * 70)
    export_sel = all_selections[best_key]

    cards_out = []
    for rank, idx in enumerate(export_sel):
        card = dict(data[idx])
        card["_mss"] = {
            "rank": rank,
            "pool_index": idx,
            "base_score": round(float(base[idx]), 5),
            "predicted_specialty": pred_labels[idx],
            "cluster_id": int(pool_clusters[idx]),
        }
        cards_out.append(card)

    with open(args.output_cards, "w", encoding="utf-8") as f:
        json.dump(cards_out, f, indent=2, ensure_ascii=False)
    print(f"  Cards ({export_k}) -> {args.output_cards}")

    results = {
        "method": "topic_aware_mmr",
        "formula": {
            "Score": "λ1·Q(ci) + λ2·P_local(ci) + λ3·U(ci)",
            "Penalty": "α·SpecSaturation + β·ClusterSaturation",
            "MMR": "c_next = argmax { Score(ci) - λ·Penalty(ci,S) }",
            "SelectionScore": "w_tv*(1-TV_to_pool) + w_eff*EffectiveSpecRatio + w_base*AvgBase",
            "alpha_spec": args.alpha_spec,
            "beta_cluster": args.beta_cluster,
            "w_tv": args.w_tv,
            "w_eff": args.w_eff,
            "w_base": args.w_base,
            "note": "Replaces cosine sim (std=0.019) with structured sim (range 0→1)",
        },
        "config": {
            "pool_size": N,
            "n_pool_clusters": n_total_clusters,
            "k_values": k_values,
            "mmr_lambdas": lambdas,
            "best_lambda": best_lam,
            "export_k": export_k,
        },
        "pool_specialty_distribution": dict(pool_dist.most_common()),
        "evaluation_all": all_eval,
        "best_evaluation": best,
        "selected_indices": export_sel,
        "total_time_sec": round(time.time() - t0, 1),
    }
    with open(args.output_results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results -> {args.output_results}")

    elapsed = time.time() - t0
    print(f"\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Pool: {N} | MSS: {export_k} ({export_k/N*100:.1f}%)")
    print(f"  Best λ={best_lam} | SpecEntropy={best['spec_entropy']:.3f} | "
          f"ClusterCov={best['cluster_coverage']:.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
