"""
MSS Full Pool Run v3: Stratified MMR with Specialty Quota Constraint
--------------------------------------------------------------------
全量卡池 16715 张 × PCA 64d 投影 × 两阶段 Stratified MMR × 聚类分析

两阶段选择策略:
  Phase 1 — 按专科配额在各专科内部跑 MMR，保证每个专科最低覆盖
  Phase 2 — 剩余名额跑全局 MMR（以 Phase 1 已选为初始集），质量竞争

支持多 K × 多 λ 扫描（编码/投影只做一次）。
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# ── 默认配置 ──────────────────────────────────────────────────────────
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
# 工具函数
# ══════════════════════════════════════════════════════════════════════
def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)


def minmax(v: np.ndarray) -> np.ndarray:
    return MinMaxScaler().fit_transform(v.reshape(-1, 1)).ravel()


# ══════════════════════════════════════════════════════════════════════
# Base Score (批量版)
# ══════════════════════════════════════════════════════════════════════
def build_base_score(
    model: SentenceTransformer,
    emb: np.ndarray,
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


def project_pca(x: np.ndarray, dim: int, seed: int = 42) -> np.ndarray:
    if dim <= 0 or dim >= x.shape[1]:
        return x
    z = PCA(n_components=dim, random_state=seed).fit_transform(x)
    return l2n(np.asarray(z, dtype=np.float32))


# ══════════════════════════════════════════════════════════════════════
# MMR 选择算法
# ══════════════════════════════════════════════════════════════════════
def _mmr_on_subset(
    vecs: np.ndarray,
    base: np.ndarray,
    indices: List[int],
    n_select: int,
    mmr_lambda: float,
) -> List[int]:
    """在 indices 子集内部跑 stream MMR，返回全局索引列表。"""
    if n_select <= 0:
        return []
    idx = np.array(indices, dtype=np.int64)
    if n_select >= len(idx):
        return idx.tolist()

    sub_v = vecs[idx]
    sub_b = base[idx]
    m = len(idx)

    sel_local: List[int] = []
    used = np.zeros(m, dtype=bool)

    first = int(np.argmax(sub_b))
    sel_local.append(first)
    used[first] = True
    best_sim = sub_v @ sub_v[first]
    best_sim[first] = 1.0

    while len(sel_local) < n_select:
        mmr = sub_b - mmr_lambda * best_sim
        mmr[used] = -np.inf
        pick = int(np.argmax(mmr))
        if mmr[pick] == -np.inf:
            break
        sel_local.append(pick)
        used[pick] = True
        best_sim = np.maximum(best_sim, sub_v @ sub_v[pick])

    return [int(idx[i]) for i in sel_local]


def compute_quotas(
    spec_indices: Dict[str, List[int]],
    total_k: int,
    quota_ratio: float,
) -> Dict[str, int]:
    """
    按 tier_weight × sqrt(pool_count) 分配配额。
    保证每个有卡的专科至少拿到 min_per 张。
    """
    quota_budget = int(total_k * quota_ratio)
    n_specs = len(spec_indices)
    min_per = max(3, quota_budget // (n_specs * 4))

    weights = {}
    for spec, idxs in spec_indices.items():
        tier = TIER_MAPPING.get(spec, 0.5)
        weights[spec] = tier * np.sqrt(len(idxs))
    total_w = sum(weights.values())

    quotas = {}
    for spec, w in weights.items():
        raw = int(quota_budget * w / total_w)
        avail = len(spec_indices[spec])
        quotas[spec] = min(max(raw, min_per), avail)

    # 如果超预算，按比例缩
    total_q = sum(quotas.values())
    if total_q > quota_budget:
        scale = quota_budget / total_q
        quotas = {
            s: max(min_per, min(int(q * scale), len(spec_indices[s])))
            for s, q in quotas.items()
        }
    return quotas


def stratified_mmr(
    base: np.ndarray,
    pred_labels: List[str],
    cluster_ids: np.ndarray,
    max_k: int,
    mmr_lambda: float,
    quota_ratio: float = 0.60,
    alpha_spec: float = 0.6,
    beta_cluster: float = 0.4,
) -> Tuple[List[int], Dict]:
    """
    两阶段 Stratified MMR。
    Phase 1: 按专科配额，在各专科子集内做 MMR，保证覆盖。
    Phase 2: 剩余名额做全局 stream MMR（以 Phase 1 已选为初始集）。
    """
    n = len(pred_labels)
    cluster_arr = np.asarray(cluster_ids, dtype=np.int32)

    # ── 分组 ──
    spec_indices: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(pred_labels):
        spec_indices[label].append(i)
    specs = sorted(spec_indices.keys())
    spec_to_id = {s: i for i, s in enumerate(specs)}
    spec_ids = np.array([spec_to_id[s] for s in pred_labels], dtype=np.int32)
    n_specs = len(specs)
    n_clusters = int(cluster_arr.max()) + 1

    # ── 配额 ──
    quotas = compute_quotas(spec_indices, max_k, quota_ratio)
    spec_target = np.ones(n_specs, dtype=np.float32)
    for s, sid in spec_to_id.items():
        spec_target[sid] = float(max(1, quotas.get(s, 1)))
    cluster_target = max(1.0, max_k / max(n_clusters, 1))

    # ── Phase 1: per-specialty MMR ──
    phase1: List[int] = []
    phase1_by_spec: Dict[str, List[int]] = {}
    used_set: set = set()
    spec_count = np.zeros(n_specs, dtype=np.float32)
    cluster_count = np.zeros(n_clusters, dtype=np.float32)

    for spec in sorted(quotas.keys(), key=lambda s: -quotas[s]):
        quota = quotas[spec]
        candidates = [i for i in spec_indices[spec] if i not in used_set]
        if not candidates or quota <= 0:
            phase1_by_spec[spec] = []
            continue
        sel = []
        spec_id = spec_to_id[spec]
        while len(sel) < quota and candidates:
            cand = np.array(candidates, dtype=np.int32)
            spec_sat = min(1.0, float(spec_count[spec_id] / max(spec_target[spec_id], 1.0)))
            cl_sat = np.minimum(1.0, cluster_count[cluster_arr[cand]] / max(cluster_target, 1.0))
            penalty = alpha_spec * spec_sat + beta_cluster * cl_sat
            score = base[cand] - mmr_lambda * penalty
            pick_local = int(np.argmax(score))
            pick = int(cand[pick_local])
            sel.append(pick)
            used_set.add(pick)
            spec_count[spec_id] += 1
            cluster_count[cluster_arr[pick]] += 1
            candidates.pop(pick_local)
        phase1.extend(sel)
        phase1_by_spec[spec] = sel

    # ── Phase 2: global MMR on remaining ──
    remaining_budget = max_k - len(phase1)
    phase2: List[int] = []

    if remaining_budget > 0:
        used_arr = np.zeros(n, dtype=bool)
        for i in phase1:
            used_arr[i] = True

        log_interval = max(1, remaining_budget // 5)
        while len(phase2) < remaining_budget:
            spec_sat = np.minimum(1.0, spec_count[spec_ids] / spec_target[spec_ids].clip(min=1.0))
            cl_sat = np.minimum(1.0, cluster_count[cluster_arr] / max(cluster_target, 1.0))
            penalty = alpha_spec * spec_sat + beta_cluster * cl_sat
            mmr = base - mmr_lambda * penalty
            mmr[used_arr] = -np.inf
            pick = int(np.argmax(mmr))
            if mmr[pick] == -np.inf:
                break
            phase2.append(pick)
            used_arr[pick] = True
            sid = spec_ids[pick]
            spec_count[sid] += 1
            cluster_count[cluster_arr[pick]] += 1
            if len(phase2) % log_interval == 0:
                print(f"      Phase2 global: {len(phase2)}/{remaining_budget}")

    all_selected = phase1 + phase2

    info = {
        "phase1_count": len(phase1),
        "phase2_count": len(phase2),
        "quota_ratio": quota_ratio,
        "quotas": {
            s: {
                "quota": quotas.get(s, 0),
                "actual": len(phase1_by_spec.get(s, [])),
                "pool_available": len(spec_indices.get(s, [])),
            }
            for s in sorted(set(list(quotas.keys()) + list(spec_indices.keys())))
        },
    }
    return all_selected, info


# ══════════════════════════════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════════════════════════════
def evaluate(
    eval_vecs: np.ndarray,
    selected: List[int],
    base: np.ndarray,
    all_labels: List[str],
    all_clusters: np.ndarray,
    pool_spec_dist: Dict[str, int],
    alpha_spec: float = 0.6,
    beta_cluster: float = 0.4,
    batch_size: int = 2048,
) -> Dict:
    s = np.array(selected, dtype=np.int32)
    sel_vecs = eval_vecs[s]
    n = eval_vecs.shape[0]
    k = len(s)

    # Coverage
    max_sims: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = eval_vecs[start:end] @ sel_vecs.T
        max_sims.append(np.max(sims, axis=1))
    coverage = float(np.mean(np.concatenate(max_sims)))

    # Redundancy (structured topic/cluster similarity)
    if k > 1:
        sel_labels = np.array([all_labels[i] for i in selected], dtype=object)
        sel_clusters = np.array([int(all_clusters[i]) for i in selected], dtype=np.int32)
        same_spec = (sel_labels[:, None] == sel_labels[None, :]).astype(np.float32)
        same_cluster = (sel_clusters[:, None] == sel_clusters[None, :]).astype(np.float32)
        sim_red = np.clip(alpha_spec * same_spec + beta_cluster * same_cluster, 0.0, 1.0)
        raw_red = float((np.sum(sim_red) - k) / (k * (k - 1)))
    else:
        raw_red = 0.0
    redundancy = float(np.clip(raw_red, 0.0, 1.0))
    diversity = 1.0 - redundancy

    # Label coverage
    unique_pool = set(all_labels)
    unique_mss = set(all_labels[i] for i in selected)
    label_cov = len(unique_mss) / max(len(unique_pool), 1)

    # Per-specialty coverage
    spec_counts = Counter(all_labels[i] for i in selected)
    spec_keys = sorted(pool_spec_dist.keys())
    sel_arr = np.array([spec_counts.get(s, 0) for s in spec_keys], dtype=float)
    pool_arr = np.array([pool_spec_dist.get(s, 0) for s in spec_keys], dtype=float)
    sel_probs = sel_arr / sel_arr.sum() if sel_arr.sum() > 0 else np.zeros_like(sel_arr)
    pool_probs = pool_arr / pool_arr.sum() if pool_arr.sum() > 0 else np.zeros_like(pool_arr)
    tv_to_pool = float(0.5 * np.sum(np.abs(sel_probs - pool_probs)))
    nz = sel_probs[sel_probs > 0]
    eff_spec_ratio = float(np.exp(-np.sum(nz * np.log(nz))) / max(len(sel_probs), 1)) if len(nz) > 0 else 0.0

    return {
        "coverage": round(coverage, 5),
        "redundancy": round(redundancy, 5),
        "diversity": round(diversity, 5),
        "avg_base_score": round(float(np.mean(base[s])), 5),
        "tv_to_pool": round(tv_to_pool, 5),
        "effective_spec_ratio": round(eff_spec_ratio, 5),
        "label_coverage": round(label_cov, 5),
        "unique_specialties": len(unique_mss),
        "specialty_counts": dict(spec_counts.most_common()),
    }


# ══════════════════════════════════════════════════════════════════════
# 聚类分析
# ══════════════════════════════════════════════════════════════════════
def cluster_analysis(emb_sel: np.ndarray, labels_sel: List[str], n_clusters: int) -> Dict:
    if n_clusters >= len(labels_sel):
        n_clusters = max(2, len(labels_sel) // 2)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cids = km.fit_predict(emb_sel)
    sil = float(silhouette_score(emb_sel, cids, metric="cosine"))

    clusters = {}
    for c in range(n_clusters):
        mask = cids == c
        labs = [labels_sel[i] for i, m in enumerate(mask) if m]
        clusters[c] = {
            "size": int(mask.sum()),
            "top_specialties": [[s, cnt] for s, cnt in Counter(labs).most_common(3)],
        }
    return {
        "n_clusters": n_clusters,
        "silhouette_cosine": round(sil, 4),
        "specialty_distribution": dict(Counter(labels_sel).most_common()),
        "clusters": clusters,
        "cluster_ids": cids.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════
# CLI & Main
# ══════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified MMR: specialty-constrained MSS selection"
    )
    p.add_argument("--input_file", default=INPUT_FILE)
    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--k_values", type=int, nargs="+", default=[2000, 3000, 5000, 8000])
    p.add_argument("--export_k", type=int, default=3000)
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--mmr_lambdas", type=float, nargs="+",
                    default=[0.70, 0.75, 0.80, 0.85, 0.90])
    p.add_argument("--quota_ratio", type=float, default=0.60,
                    help="Phase1 专科配额占 K 的比例 (0.6 = 60%% 配额 + 40%% 自由竞争)")
    p.add_argument("--n_pool_clusters", type=int, default=60,
                    help="全池 cluster 数，用于结构化冗余惩罚")
    p.add_argument("--alpha_spec", type=float, default=0.6,
                    help="结构化冗余中专科同类惩罚权重")
    p.add_argument("--beta_cluster", type=float, default=0.4,
                    help="结构化冗余中cluster同类惩罚权重")
    p.add_argument("--w_tier", type=float, default=1.0)
    p.add_argument("--w_rep", type=float, default=1.0)
    p.add_argument("--w_unc", type=float, default=0.8)
    p.add_argument("--w_tv", type=float, default=0.45,
                    help="weight for (1 - TV_to_pool) in final score")
    p.add_argument("--w_eff", type=float, default=0.35,
                    help="weight for effective specialty ratio in final score")
    p.add_argument("--w_base", type=float, default=0.20,
                    help="weight for avg base score in final score")
    p.add_argument("--n_clusters", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_results", default="mss_stratified_results.json")
    p.add_argument("--output_cards", default="mss_stratified_cards.json")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    t_start = time.time()

    # ── 1. 加载 ──
    print("=" * 70)
    print("STEP 1: 加载全量卡片")
    print("=" * 70)
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    N = len(data)
    print(f"  卡片总数: {N}")

    key_f = "improved_front" if "improved_front" in data[0] else "front"
    key_b = "improved_back" if "improved_back" in data[0] else "back"
    texts = [f"{x.get(key_f, '')} {x.get(key_b, '')}".strip() for x in data]

    # ── 2. 编码 (一次) ──
    print("\n" + "=" * 70)
    print("STEP 2: PubMedBERT 编码")
    print("=" * 70)
    model = SentenceTransformer(args.model_name)
    t_enc = time.time()
    emb = l2n(np.asarray(
        model.encode(texts, convert_to_tensor=False, show_progress_bar=True),
        dtype=np.float32))
    print(f"  Embedding: {emb.shape}  ({time.time() - t_enc:.1f}s)")

    # ── 3. Base Score (一次) ──
    print("\n" + "=" * 70)
    print("STEP 3: Base Score (batched)")
    print("=" * 70)
    t_bs = time.time()
    base, pred_labels = build_base_score(model, emb, args.w_tier, args.w_rep, args.w_unc)
    print(f"  完成 ({time.time() - t_bs:.1f}s), mean={base.mean():.4f}")

    # 池子专科分布
    pool_dist = Counter(pred_labels)
    print(f"\n  全量池专科分布 ({len(pool_dist)} 个专科):")
    for spec, cnt in pool_dist.most_common():
        print(f"    {spec:25s}  {cnt:5d} ({cnt/N*100:5.1f}%)")

    # ── 4. PCA 投影 (一次) ──
    print("\n" + "=" * 70)
    print(f"STEP 4: PCA → {args.proj_dim}d")
    print("=" * 70)
    z = project_pca(emb, args.proj_dim, args.seed)
    print(f"  {z.shape}")
    print(f"  Pool topic clusters: {args.n_pool_clusters}")
    pool_cluster_ids = KMeans(n_clusters=max(2, min(args.n_pool_clusters, N)), random_state=args.seed, n_init=10).fit_predict(z)

    # ── 5. Stratified MMR: 多 K × 多 λ ──
    lambdas = sorted(args.mmr_lambdas)
    k_values = sorted(args.k_values)
    total_runs = len(lambdas) * len(k_values)
    print("\n" + "=" * 70)
    print(f"STEP 5: Stratified MMR ({total_runs} runs)")
    print(f"  K values:  {k_values}")
    print(f"  λ values:  {lambdas}")
    print(f"  配额比例:  {args.quota_ratio:.0%}")
    print("=" * 70)

    all_eval: List[Dict] = []
    all_selections: Dict[str, List[int]] = {}  # "lam_k" -> indices
    all_strat_info: Dict[str, Dict] = {}

    run_i = 0
    for lam in lambdas:
        for k in k_values:
            run_i += 1
            key = f"{lam:.2f}_{k}"
            t_run = time.time()
            print(f"\n  [{run_i}/{total_runs}] λ={lam:.2f}, K={k}")

            selected, strat_info = stratified_mmr(
                base, pred_labels, pool_cluster_ids, k, lam, args.quota_ratio, args.alpha_spec, args.beta_cluster
            )
            all_selections[key] = selected
            all_strat_info[key] = strat_info

            met = evaluate(
                emb, selected, base, pred_labels, pool_cluster_ids, dict(pool_dist),
                args.alpha_spec, args.beta_cluster
            )
            met["lambda"] = lam
            met["k"] = k
            met["selection_ratio_pct"] = round(k / N * 100, 2)
            met["phase1"] = strat_info["phase1_count"]
            met["phase2"] = strat_info["phase2_count"]
            all_eval.append(met)

            print(f"    Phase1={strat_info['phase1_count']}, Phase2={strat_info['phase2_count']}")
            print(f"    Cov={met['coverage']:.4f}  TV={met['tv_to_pool']:.4f}  "
                  f"EffSpec={met['effective_spec_ratio']:.4f}  Base={met['avg_base_score']:.4f}  "
                  f"LblCov={met['label_coverage']:.3f} ({met['unique_specialties']} specs)  "
                  f"({time.time()-t_run:.1f}s)")

    # ── 6. 结果总表 ──
    print("\n" + "=" * 70)
    print("STEP 6: 结果总表 (原始768维评估)")
    print("=" * 70)
    header = (f"  {'λ':>5s}  {'K':>6s}  {'%':>5s}  {'P1':>5s}  {'P2':>5s}  "
              f"{'Cover':>7s}  {'Redund':>7s}  {'Divers':>7s}  "
              f"{'Base':>7s}  {'LblCov':>6s}  {'Sp':>3s}")
    print(header)
    print("  " + "-" * 76)
    for e in all_eval:
        print(f"  {e['lambda']:>5.2f}  {e['k']:>6d}  {e['selection_ratio_pct']:>4.1f}%  "
              f"{e['phase1']:>5d}  {e['phase2']:>5d}  "
              f"{e['coverage']:>7.4f}  {e['redundancy']:>7.4f}  "
              f"{e['diversity']:>7.4f}  {e['avg_base_score']:>7.4f}  "
              f"{e['label_coverage']:>6.3f}  {e['unique_specialties']:>3d}")

    # ── 7. 选最佳配置导出 ──
    export_k = args.export_k
    # 综合得分: 高 (1-TV_to_pool) + 高 effective_spec_ratio + 高 avg_base_score
    scored = []
    for e in all_eval:
        if e["k"] == export_k:
            score = (
                args.w_tv * (1.0 - e["tv_to_pool"])
                + args.w_eff * e["effective_spec_ratio"]
                + args.w_base * e["avg_base_score"]
            )
            e["selection_score"] = round(float(score), 6)
            scored.append((score, e))
    if not scored:
        # 取最接近 export_k 的
        closest_k = min(k_values, key=lambda x: abs(x - export_k))
        for e in all_eval:
            if e["k"] == closest_k:
                score = (
                    args.w_tv * (1.0 - e["tv_to_pool"])
                    + args.w_eff * e["effective_spec_ratio"]
                    + args.w_base * e["avg_base_score"]
                )
                e["selection_score"] = round(float(score), 6)
                scored.append((score, e))
        export_k = closest_k

    scored.sort(key=lambda x: -x[0])
    best_met = scored[0][1]
    best_lam = best_met["lambda"]
    best_key = f"{best_lam:.2f}_{export_k}"

    print(f"\n  ★ 最佳: λ={best_lam:.2f}, K={export_k}")
    print(f"    Score={best_met.get('selection_score', 0.0):.4f}  TV={best_met['tv_to_pool']:.4f}  "
          f"EffSpec={best_met['effective_spec_ratio']:.4f}  Base={best_met['avg_base_score']:.4f}  "
          f"LblCov={best_met['label_coverage']:.3f} ({best_met['unique_specialties']} specs)")

    export_indices = all_selections[best_key]
    export_strat = all_strat_info[best_key]

    # 打印配额详情
    print(f"\n  专科配额详情 (λ={best_lam}, K={export_k}):")
    print(f"    {'Specialty':25s}  {'Pool':>5s}  {'Quota':>5s}  {'Actual':>6s}")
    print(f"    {'-'*47}")
    for spec in sorted(export_strat["quotas"].keys()):
        q = export_strat["quotas"][spec]
        print(f"    {spec:25s}  {q['pool_available']:>5d}  {q['quota']:>5d}  {q['actual']:>6d}")

    # ── 8. 聚类 ──
    print("\n" + "=" * 70)
    print(f"STEP 7: K-Means 聚类 (K={export_k}, {args.n_clusters} clusters)")
    print("=" * 70)
    emb_sel = emb[export_indices]
    labels_sel = [pred_labels[i] for i in export_indices]
    cluster_result = cluster_analysis(emb_sel, labels_sel, args.n_clusters)
    print(f"  Silhouette (cosine): {cluster_result['silhouette_cosine']:.4f}")

    print(f"\n  专科分布 (MSS K={export_k}):")
    for spec, cnt in cluster_result["specialty_distribution"].items():
        pct = cnt / export_k * 100
        bar = "█" * max(1, int(pct * 0.8))
        print(f"    {spec:25s}  {cnt:5d} ({pct:5.1f}%)  {bar}")

    # ── 9. 保存 ──
    print("\n" + "=" * 70)
    print("STEP 8: 保存")
    print("=" * 70)

    cards_out = []
    cids = cluster_result["cluster_ids"]
    for rank, idx in enumerate(export_indices):
        card = dict(data[idx])
        card["_mss"] = {
            "rank": rank,
            "pool_index": idx,
            "base_score": round(float(base[idx]), 5),
            "predicted_specialty": pred_labels[idx],
            "cluster_id": cids[rank],
            "pool_cluster_id": int(pool_cluster_ids[idx]),
            "phase": "quota" if rank < export_strat["phase1_count"] else "free",
        }
        cards_out.append(card)

    with open(args.output_cards, "w", encoding="utf-8") as f:
        json.dump(cards_out, f, indent=2, ensure_ascii=False)
    print(f"  MSS cards ({export_k}) -> {args.output_cards}")

    results = {
        "config": {
            "input_file": args.input_file,
            "pool_size": N,
            "proj_dim": args.proj_dim,
            "mmr_lambdas": lambdas,
            "k_values": k_values,
            "quota_ratio": args.quota_ratio,
            "selection_weights": {
                "w_tv": args.w_tv,
                "w_eff": args.w_eff,
                "w_base": args.w_base,
            },
            "best_lambda": best_lam,
            "export_k": export_k,
        },
        "pool_specialty_distribution": dict(pool_dist.most_common()),
        "evaluation_all": all_eval,
        "best_evaluation": best_met,
        "best_stratification": export_strat,
        "clustering": {
            k: v for k, v in cluster_result.items() if k != "cluster_ids"
        },
        "selected_indices": export_indices,
        "total_time_sec": round(time.time() - t_start, 1),
    }
    with open(args.output_results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results -> {args.output_results}")

    # ── 总结 ──
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  全量池:     {N} 张 ({len(pool_dist)} 个专科)")
    print(f"  投影:       PCA → {args.proj_dim}d")
    print(f"  策略:       Stratified MMR (配额 {args.quota_ratio:.0%} + 自由 {1-args.quota_ratio:.0%})")
    print(f"  最佳:       λ={best_lam}, K={export_k}")
    print(f"  专科覆盖:   {best_met['unique_specialties']}/{len(pool_dist)}")
    print(f"  耗时:       {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
