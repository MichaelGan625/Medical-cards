from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
INPUT_FILE = "deepseek_cards.json"

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MSS with low-d soft-prompt style projection.")
    p.add_argument("--input_file", default=INPUT_FILE)
    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--pool_limit", type=int, default=3000)
    p.add_argument("--k_values", type=int, nargs="+", default=[100, 200, 400, 800])
    p.add_argument("--low_dim", type=int, default=32)
    p.add_argument("--projection", choices=["random", "pca"], default="random")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="+", default=["mmr", "facility"])
    p.add_argument("--mmr_lambda", type=float, default=0.45)
    p.add_argument("--n_topic_clusters", type=int, default=60)
    p.add_argument("--alpha_spec", type=float, default=0.6)
    p.add_argument("--beta_cluster", type=float, default=0.4)
    p.add_argument("--w_tier", type=float, default=1.0)
    p.add_argument("--w_rep", type=float, default=1.0)
    p.add_argument("--w_unc", type=float, default=0.8)
    p.add_argument("--review_sec_per_card", type=float, default=30.0)
    p.add_argument("--output_json", default="mss_softprompt_projection.json")
    p.add_argument("--output_html", default="mss_softprompt_projection.html")
    p.add_argument("--export_k", type=int, default=800, help="Export final MSS at this K.")
    p.add_argument("--export_method", choices=["mmr", "facility", "topk"], default="mmr")
    p.add_argument(
        "--export_select_space",
        choices=["projected", "original"],
        default="projected",
        help="Export MSS from selection in projected or original space.",
    )
    p.add_argument("--output_mss_ids", default="mss_ids.json")
    p.add_argument("--output_mss_cards", default="mss_cards.json")
    p.add_argument("--export_cards", action="store_true", help="Also export full selected card objects.")
    p.add_argument(
        "--stream_mmr",
        action="store_true",
        help="Use streaming MMR to avoid NxN similarity matrix (recommended for full pool).",
    )
    return p.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(v.reshape(-1, 1)).reshape(-1)


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)


def build_base_score(
    model: SentenceTransformer,
    emb: np.ndarray,
    w_tier: float,
    w_rep: float,
    w_unc: float,
) -> Tuple[np.ndarray, List[str]]:
    label_emb = model.encode(CANDIDATE_LABELS, convert_to_tensor=False)
    label_emb = l2n(np.asarray(label_emb, dtype=np.float32))
    score_mat = emb @ label_emb.T
    top_idx = np.argmax(score_mat, axis=1)
    pred_labels = [CANDIDATE_LABELS[i] for i in top_idx]
    tier = np.array([TIER_MAPPING.get(CANDIDATE_LABELS[i], 0.5) for i in top_idx], dtype=np.float32)

    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    k = min(10, max(1, emb.shape[0] - 1))
    rep = np.mean(np.sort(sim, axis=1)[:, -k:], axis=1)

    sort_scores = np.sort(score_mat, axis=1)
    margin = sort_scores[:, -1] - sort_scores[:, -2]
    unc = 1.0 - normalize(margin)

    base = w_tier * normalize(tier) + w_rep * normalize(rep) + w_unc * normalize(unc)
    return normalize(base), pred_labels


def project_soft_prompt_style(x: np.ndarray, low_dim: int, mode: str, seed: int) -> np.ndarray:
    d = x.shape[1]
    if low_dim <= 0 or low_dim >= d:
        return x

    if mode == "pca":
        z = PCA(n_components=low_dim, random_state=seed).fit_transform(x)
    else:
        # InstructZero-style random projection proxy: z in low-d, with distance preservation.
        rng = np.random.default_rng(seed)
        A = rng.normal(0.0, 1.0 / np.sqrt(low_dim), size=(d, low_dim)).astype(np.float32)
        z = x @ A
    return l2n(np.asarray(z, dtype=np.float32))


def cluster_topics(emb: np.ndarray, n_clusters: int) -> np.ndarray:
    n = emb.shape[0]
    k = max(2, min(n_clusters, n))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(emb)


def structured_similarity(pred_labels: List[str], cluster_ids: np.ndarray, alpha_spec: float, beta_cluster: float) -> np.ndarray:
    labels = np.asarray(pred_labels)
    same_spec = (labels[:, None] == labels[None, :]).astype(np.float32)
    clusters = np.asarray(cluster_ids)
    same_cluster = (clusters[:, None] == clusters[None, :]).astype(np.float32)
    sim_new = alpha_spec * same_spec + beta_cluster * same_cluster
    np.fill_diagonal(sim_new, 1.0)
    return np.clip(sim_new, 0.0, 1.0)


def select_topk(base: np.ndarray, max_k: int) -> List[int]:
    return list(np.argsort(-base)[:max_k])


def select_mmr(sim: np.ndarray, base: np.ndarray, max_k: int, mmr_lambda: float) -> List[int]:
    n = sim.shape[0]
    selected: List[int] = []
    candidates = np.arange(n)
    first = int(np.argmax(base))
    selected.append(first)
    candidates = candidates[candidates != first]
    best_sim = sim[candidates, first].copy()

    while len(selected) < max_k and len(candidates) > 0:
        mmr_score = base[candidates] - mmr_lambda * best_sim
        pos = int(np.argmax(mmr_score))
        pick = int(candidates[pos])
        selected.append(pick)
        candidates = np.delete(candidates, pos)
        if len(candidates) > 0:
            best_sim = np.maximum(best_sim[np.arange(len(best_sim)) != pos], sim[candidates, pick])
    return selected


def select_mmr_stream(sim_red: np.ndarray, base: np.ndarray, max_k: int, mmr_lambda: float) -> List[int]:
    n = sim_red.shape[0]
    selected: List[int] = []
    used = np.zeros(n, dtype=bool)

    first = int(np.argmax(base))
    selected.append(first)
    used[first] = True

    best_sim = sim_red[:, first].copy()
    best_sim[first] = 1.0

    while len(selected) < max_k:
        mmr = base - mmr_lambda * best_sim
        mmr[used] = -np.inf
        pick = int(np.argmax(mmr))
        selected.append(pick)
        used[pick] = True
        best_sim = np.maximum(best_sim, sim_red[:, pick])

    return selected


def select_facility(sim: np.ndarray, base: np.ndarray, max_k: int, alpha: float = 0.25) -> List[int]:
    n = sim.shape[0]
    selected: List[int] = []
    candidates = np.arange(n)
    best_cover = np.zeros(n, dtype=np.float32)
    while len(selected) < max_k and len(candidates) > 0:
        gains = []
        for idx in candidates:
            new_cover = np.maximum(best_cover, sim[:, idx])
            gain = float(np.sum(new_cover - best_cover) + alpha * base[idx])
            gains.append(gain)
        pos = int(np.argmax(gains))
        pick = int(candidates[pos])
        selected.append(pick)
        best_cover = np.maximum(best_cover, sim[:, pick])
        candidates = np.delete(candidates, pos)
    return selected


def evaluate_set(
    sim_cover_eval: np.ndarray,
    sim_red_eval: np.ndarray,
    selected: List[int],
    base: np.ndarray,
    sec_per_card: float,
) -> Dict[str, float]:
    s = np.array(selected, dtype=np.int32)
    cov = float(np.mean(np.max(sim_cover_eval[:, s], axis=1)))
    if len(s) > 1:
        sub = sim_red_eval[np.ix_(s, s)]
        red = float((np.sum(sub) - len(s)) / (len(s) * (len(s) - 1)))
    else:
        red = 0.0
    red = float(np.clip(red, 0.0, 1.0))
    return {
        "coverage": cov,
        "redundancy": red,
        "diversity": 1.0 - red,
        "avg_base_score": float(np.mean(base[s])),
        "time_hours": float(len(s) * sec_per_card / 3600.0),
    }


def evaluate_set_from_vectors(
    eval_vecs: np.ndarray,
    selected: List[int],
    base: np.ndarray,
    sec_per_card: float,
    batch_size: int = 2048,
) -> Dict[str, float]:
    raise RuntimeError("evaluate_set_from_vectors is not used under topic/label structured redundancy.")


def run_selector(method: str, sim: np.ndarray, base: np.ndarray, max_k: int, mmr_lambda: float) -> List[int]:
    if method == "topk":
        return select_topk(base, max_k)
    if method == "mmr":
        return select_mmr(sim, base, max_k, mmr_lambda)
    if method == "facility":
        return select_facility(sim, base, max_k)
    raise ValueError(f"Unknown method: {method}")


def create_plot(df: pd.DataFrame, out_html: str):
    fig = go.Figure()
    colors = {
        "original": "#1f77b4",
        "projected": "#d62728",
    }
    for method in sorted(df["method"].unique()):
        for space in ["original", "projected"]:
            d = df[(df["method"] == method) & (df["select_space"] == space)]
            fig.add_trace(
                go.Scatter(
                    x=d["k"],
                    y=d["coverage_eval_original"],
                    mode="lines+markers",
                    name=f"{method}-{space}",
                    line=dict(color=colors[space], dash="solid" if method == "mmr" else "dash"),
                )
            )
    fig.update_layout(
        title="Coverage in Original Space: Selection in Original vs Low-d Projected",
        xaxis_title="K",
        yaxis_title="Coverage (original space eval)",
        template="plotly_white",
    )
    fig.write_html(out_html)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.pool_limit > 0:
        data = data[: args.pool_limit]
    print(f"📊 pool size: {len(data)}")

    key_front = "improved_front" if "improved_front" in data[0] else "front"
    key_back = "improved_back" if "improved_back" in data[0] else "back"
    texts = [f"{x.get(key_front, '')} {x.get(key_back, '')}".strip() for x in data]

    model = SentenceTransformer(args.model_name)
    print("🧠 encoding BERT embeddings...")
    x = np.asarray(model.encode(texts, convert_to_tensor=False, show_progress_bar=True), dtype=np.float32)
    x = l2n(x)
    z = project_soft_prompt_style(x, args.low_dim, args.projection, args.seed)

    base, pred_labels = build_base_score(model, x, args.w_tier, args.w_rep, args.w_unc)
    use_stream = args.stream_mmr
    if use_stream and any(m != "mmr" for m in args.methods):
        raise ValueError("--stream_mmr currently supports only method=mmr.")

    # coverage keeps cosine view; redundancy/selection switches to structured similarity.
    sim_cov_x = x @ x.T
    sim_cov_z = z @ z.T
    np.fill_diagonal(sim_cov_x, 1.0)
    np.fill_diagonal(sim_cov_z, 1.0)

    cluster_x = cluster_topics(x, args.n_topic_clusters)
    cluster_z = cluster_topics(z, args.n_topic_clusters)
    sim_red_x = structured_similarity(pred_labels, cluster_x, args.alpha_spec, args.beta_cluster)
    sim_red_z = structured_similarity(pred_labels, cluster_z, args.alpha_spec, args.beta_cluster)

    max_k = min(max(args.k_values), len(data))
    rows = []
    selected_store = {}

    for method in args.methods:
        if use_stream and method == "mmr":
            rank_x = select_mmr_stream(sim_red_x, base, max_k=max_k, mmr_lambda=args.mmr_lambda)
            rank_z = select_mmr_stream(sim_red_z, base, max_k=max_k, mmr_lambda=args.mmr_lambda)
        else:
            # Baseline: select in original space.
            rank_x = run_selector(method, sim_red_x, base, max_k=max_k, mmr_lambda=args.mmr_lambda)
            # Proposed: select in projected low-d space.
            rank_z = run_selector(method, sim_red_z, base, max_k=max_k, mmr_lambda=args.mmr_lambda)
        selected_store[f"{method}_original"] = rank_x
        selected_store[f"{method}_projected"] = rank_z

        for k in args.k_values:
            kk = min(k, len(rank_x))
            met_x_in_x = evaluate_set(sim_cov_x, sim_red_x, rank_x[:kk], base, args.review_sec_per_card)
            met_z_in_x = evaluate_set(sim_cov_x, sim_red_x, rank_z[:kk], base, args.review_sec_per_card)
            met_z_in_z = evaluate_set(sim_cov_z, sim_red_z, rank_z[:kk], base, args.review_sec_per_card)
            met_x_in_z = evaluate_set(sim_cov_z, sim_red_z, rank_x[:kk], base, args.review_sec_per_card)

            rows.append(
                {
                    "method": method,
                    "k": kk,
                    "select_space": "original",
                    "coverage_eval_original": met_x_in_x["coverage"],
                    "coverage_eval_projected": met_x_in_z["coverage"],
                    "redundancy_eval_original": met_x_in_x["redundancy"],
                    "avg_base_score": met_x_in_x["avg_base_score"],
                    "time_hours": met_x_in_x["time_hours"],
                }
            )
            rows.append(
                {
                    "method": method,
                    "k": kk,
                    "select_space": "projected",
                    "coverage_eval_original": met_z_in_x["coverage"],
                    "coverage_eval_projected": met_z_in_z["coverage"],
                    "redundancy_eval_original": met_z_in_x["redundancy"],
                    "avg_base_score": met_z_in_x["avg_base_score"],
                    "time_hours": met_z_in_x["time_hours"],
                }
            )
            print(
                f"{method:8s} K={kk:4d} | "
                f"cov_orig(select@orig)={met_x_in_x['coverage']:.4f} "
                f"vs cov_orig(select@proj)={met_z_in_x['coverage']:.4f}"
            )

    df = pd.DataFrame(rows)
    create_plot(df, args.output_html)

    export_key = f"{args.export_method}_{args.export_select_space}"
    if export_key not in selected_store:
        raise ValueError(f"Cannot export MSS: selector key '{export_key}' not found.")
    export_k = min(args.export_k, len(selected_store[export_key]))
    export_indices = selected_store[export_key][:export_k]

    # Build practical MSS outputs (id list + optional cards).
    id_key_exists = all("id" in x for x in data)
    selected_ids = [data[i]["id"] if id_key_exists else i for i in export_indices]
    mss_ids_payload = {
        "config": {
            "method": args.export_method,
            "select_space": args.export_select_space,
            "k": export_k,
            "pool_size": len(data),
            "projection": args.projection,
            "low_dim": args.low_dim,
        },
        "selected_indices": export_indices,
        "selected_ids": selected_ids,
    }
    with open(args.output_mss_ids, "w", encoding="utf-8") as f:
        json.dump(mss_ids_payload, f, indent=2, ensure_ascii=False)

    if args.export_cards:
        # cost proxy: per-card word-based reading time + base score
        cards_out = []
        for i in export_indices:
            card = dict(data[i])
            words = len((f"{card.get(key_front, '')} {card.get(key_back, '')}").split())
            card["_mss"] = {
                "index": i,
                "base_score": float(base[i]),
                "estimated_read_sec": float(words / 220.0 * 60.0),
                "words": words,
            }
            cards_out.append(card)
        with open(args.output_mss_cards, "w", encoding="utf-8") as f:
            json.dump(cards_out, f, indent=2, ensure_ascii=False)

    summary = {
        "config": vars(args),
        "projection_note": "Projected selection is performed in low-d continuous space and mapped back by original card index.",
        "results": df.to_dict(orient="records"),
        "export": {
            "output_mss_ids": args.output_mss_ids,
            "output_mss_cards": args.output_mss_cards if args.export_cards else None,
            "export_method": args.export_method,
            "export_select_space": args.export_select_space,
            "export_k": export_k,
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ saved json -> {args.output_json}")
    print(f"✅ saved html -> {args.output_html}")
    print(f"✅ saved MSS ids -> {args.output_mss_ids}")
    if args.export_cards:
        print(f"✅ saved MSS cards -> {args.output_mss_cards}")


if __name__ == "__main__":
    main()
