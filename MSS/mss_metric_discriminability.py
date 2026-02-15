import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _normalize_counts(counts: Dict[str, int], keys: List[str]) -> np.ndarray:
    arr = np.array([float(counts.get(k, 0)) for k in keys], dtype=np.float64)
    s = float(arr.sum())
    if s <= 0:
        return np.zeros_like(arr)
    return arr / s


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (_safe_log(p) - _safe_log(m))))
    kl_qm = float(np.sum(q * (_safe_log(q) - _safe_log(m))))
    return 0.5 * (kl_pm + kl_qm)


def norm_entropy(p: np.ndarray) -> float:
    n = len(p)
    if n <= 1:
        return 0.0
    nz = p[p > 0]
    h = -float(np.sum(nz * np.log(nz)))
    return h / math.log(n)


def gini_from_prob(p: np.ndarray) -> float:
    # Convert probability to pseudo-count scale for robust Gini.
    counts = p * 100000.0
    if np.sum(counts) <= 0:
        return 0.0
    c = np.sort(counts)
    n = len(c)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * c) / (n * np.sum(c))) - (n + 1) / n)


def effective_num_ratio(p: np.ndarray) -> float:
    nz = p[p > 0]
    if len(nz) == 0:
        return 0.0
    h = -float(np.sum(nz * np.log(nz)))
    eff = math.exp(h)
    return eff / len(p)


def rare_coverage_score(p_sel: np.ndarray, p_pool: np.ndarray) -> float:
    # Weighted recall for low-frequency specialties.
    # Higher when selected set allocates more mass to rare specialties.
    inv = 1.0 / np.clip(p_pool, 1e-12, None)
    inv = inv / np.sum(inv)
    return float(np.sum(inv * p_sel))


def spread(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"range": 0.0, "std": 0.0, "cv": 0.0}
    mn = min(values)
    mx = max(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    mu = float(sum(values) / len(values))
    cv = sd / (abs(mu) + 1e-12)
    return {"range": float(mx - mn), "std": float(sd), "cv": float(cv), "mean": mu, "min": mn, "max": mx}


def load_rows(path: Path) -> Tuple[Dict, List[Dict], Dict[str, int], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("evaluation_all", [])
    pool = data.get("pool_specialty_distribution", {})
    if "method" in data and data["method"] == "topic_aware_mmr":
        mode = "topic"
    else:
        mode = "stratified"
    return data, rows, pool, mode


def get_spec_counts(row: Dict, mode: str) -> Dict[str, int]:
    if mode == "topic":
        return row.get("spec_distribution", {})
    return row.get("specialty_counts", {})


def main() -> None:
    p = argparse.ArgumentParser(description="Measure discriminability of MSS metrics on full-pool runs.")
    p.add_argument("--input", nargs="+", required=True, help="Result JSON files (full pool).")
    p.add_argument("--k_values", type=int, nargs="+", default=[2000, 3000, 5000, 8000])
    p.add_argument("--output_json", default="mss_metric_discriminability_report.json")
    args = p.parse_args()

    report = {"inputs": args.input, "k_values": args.k_values, "files": []}

    for fp in args.input:
        path = Path(fp)
        data, rows, pool_dist, mode = load_rows(path)
        if not rows or not pool_dist:
            continue

        keys = sorted(pool_dist.keys())
        p_pool = _normalize_counts(pool_dist, keys)
        per_row = []

        for r in rows:
            k = int(r.get("k", -1))
            if k not in args.k_values:
                continue
            spec_counts = get_spec_counts(r, mode)
            p_sel = _normalize_counts(spec_counts, keys)

            tv = tv_distance(p_sel, p_pool)
            js = js_divergence(p_sel, p_pool)
            ent = norm_entropy(p_sel)
            gini = gini_from_prob(p_sel)
            eff_ratio = effective_num_ratio(p_sel)
            rare_score = rare_coverage_score(p_sel, p_pool)

            row_out = {
                "k": k,
                "lambda": r.get("lambda"),
                "tv_to_pool": tv,
                "js_to_pool": js,
                "norm_entropy": ent,
                "spec_gini_recomputed": gini,
                "effective_spec_ratio": eff_ratio,
                "rare_coverage_score": rare_score,
                "avg_base_score": r.get("avg_base_score"),
                "emb_coverage": r.get("emb_coverage", r.get("coverage")),
            }
            per_row.append(row_out)

        metrics = [
            "tv_to_pool",
            "js_to_pool",
            "norm_entropy",
            "spec_gini_recomputed",
            "effective_spec_ratio",
            "rare_coverage_score",
            "avg_base_score",
            "emb_coverage",
        ]

        by_k = {}
        for k in args.k_values:
            sub = [x for x in per_row if x["k"] == k]
            if not sub:
                continue
            k_stats = {}
            for m in metrics:
                vals = [float(x[m]) for x in sub if x.get(m) is not None]
                k_stats[m] = spread(vals)
            by_k[str(k)] = k_stats

        overall = {}
        for m in metrics:
            vals = [float(x[m]) for x in per_row if x.get(m) is not None]
            overall[m] = spread(vals)

        # Rank metrics by discriminability (range first, then std).
        ranking = sorted(
            metrics,
            key=lambda m: (overall[m]["range"], overall[m]["std"]),
            reverse=True,
        )

        report["files"].append(
            {
                "file": str(path),
                "mode": mode,
                "n_rows_used": len(per_row),
                "overall_spread": overall,
                "spread_by_k": by_k,
                "discriminability_ranking": ranking,
                "rows_with_new_metrics": per_row,
            }
        )

    Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {args.output_json}")

    # Console quick view
    for f in report["files"]:
        print("\n" + "=" * 70)
        print(f"{f['mode'].upper()} | {f['file']}")
        print("=" * 70)
        print("Top metrics by discriminability:")
        for m in f["discriminability_ranking"][:5]:
            s = f["overall_spread"][m]
            print(f"  {m:24s} range={s['range']:.6f} std={s['std']:.6f} cv={s['cv']:.6f}")
        print("Per-K top metric (by range):")
        for k, stats in f["spread_by_k"].items():
            top = sorted(stats.items(), key=lambda kv: kv[1]["range"], reverse=True)[0]
            print(f"  K={k}: {top[0]} (range={top[1]['range']:.6f})")


if __name__ == "__main__":
    main()
import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _normalize_counts(counts: Dict[str, int], keys: List[str]) -> np.ndarray:
    arr = np.array([float(counts.get(k, 0)) for k in keys], dtype=np.float64)
    s = float(arr.sum())
    if s <= 0:
        return np.zeros_like(arr)
    return arr / s


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (_safe_log(p) - _safe_log(m))))
    kl_qm = float(np.sum(q * (_safe_log(q) - _safe_log(m))))
    return 0.5 * (kl_pm + kl_qm)


def norm_entropy(p: np.ndarray) -> float:
    n = len(p)
    if n <= 1:
        return 0.0
    nz = p[p > 0]
    h = -float(np.sum(nz * np.log(nz)))
    return h / math.log(n)


def gini_from_prob(p: np.ndarray) -> float:
    # Convert probability to pseudo-count scale for robust Gini.
    counts = p * 100000.0
    if np.sum(counts) <= 0:
        return 0.0
    c = np.sort(counts)
    n = len(c)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * c) / (n * np.sum(c))) - (n + 1) / n)


def effective_num_ratio(p: np.ndarray) -> float:
    nz = p[p > 0]
    if len(nz) == 0:
        return 0.0
    h = -float(np.sum(nz * np.log(nz)))
    eff = math.exp(h)
    return eff / len(p)


def rare_coverage_score(p_sel: np.ndarray, p_pool: np.ndarray) -> float:
    # Weighted recall for low-frequency specialties.
    # Higher when selected set allocates more mass to rare specialties.
    inv = 1.0 / np.clip(p_pool, 1e-12, None)
    inv = inv / np.sum(inv)
    return float(np.sum(inv * p_sel))


def spread(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"range": 0.0, "std": 0.0, "cv": 0.0}
    mn = min(values)
    mx = max(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    mu = float(sum(values) / len(values))
    cv = sd / (abs(mu) + 1e-12)
    return {"range": float(mx - mn), "std": float(sd), "cv": float(cv), "mean": mu, "min": mn, "max": mx}


def load_rows(path: Path) -> Tuple[Dict, List[Dict], Dict[str, int], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("evaluation_all", [])
    pool = data.get("pool_specialty_distribution", {})
    if "method" in data and data["method"] == "topic_aware_mmr":
        mode = "topic"
    else:
        mode = "stratified"
    return data, rows, pool, mode


def get_spec_counts(row: Dict, mode: str) -> Dict[str, int]:
    if mode == "topic":
        return row.get("spec_distribution", {})
    return row.get("specialty_counts", {})


def main() -> None:
    p = argparse.ArgumentParser(description="Measure discriminability of MSS metrics on full-pool runs.")
    p.add_argument("--input", nargs="+", required=True, help="Result JSON files (full pool).")
    p.add_argument("--k_values", type=int, nargs="+", default=[2000, 3000, 5000, 8000])
    p.add_argument("--output_json", default="mss_metric_discriminability_report.json")
    args = p.parse_args()

    report = {"inputs": args.input, "k_values": args.k_values, "files": []}

    for fp in args.input:
        path = Path(fp)
        data, rows, pool_dist, mode = load_rows(path)
        if not rows or not pool_dist:
            continue

        keys = sorted(pool_dist.keys())
        p_pool = _normalize_counts(pool_dist, keys)
        per_row = []

        for r in rows:
            k = int(r.get("k", -1))
            if k not in args.k_values:
                continue
            spec_counts = get_spec_counts(r, mode)
            p_sel = _normalize_counts(spec_counts, keys)

            tv = tv_distance(p_sel, p_pool)
            js = js_divergence(p_sel, p_pool)
            ent = norm_entropy(p_sel)
            gini = gini_from_prob(p_sel)
            eff_ratio = effective_num_ratio(p_sel)
            rare_score = rare_coverage_score(p_sel, p_pool)

            row_out = {
                "k": k,
                "lambda": r.get("lambda"),
                "tv_to_pool": tv,
                "js_to_pool": js,
                "norm_entropy": ent,
                "spec_gini_recomputed": gini,
                "effective_spec_ratio": eff_ratio,
                "rare_coverage_score": rare_score,
                "avg_base_score": r.get("avg_base_score"),
                "emb_coverage": r.get("emb_coverage", r.get("coverage")),
            }
            per_row.append(row_out)

        metrics = [
            "tv_to_pool",
            "js_to_pool",
            "norm_entropy",
            "spec_gini_recomputed",
            "effective_spec_ratio",
            "rare_coverage_score",
            "avg_base_score",
            "emb_coverage",
        ]

        by_k = {}
        for k in args.k_values:
            sub = [x for x in per_row if x["k"] == k]
            if not sub:
                continue
            k_stats = {}
            for m in metrics:
                vals = [float(x[m]) for x in sub if x.get(m) is not None]
                k_stats[m] = spread(vals)
            by_k[str(k)] = k_stats

        overall = {}
        for m in metrics:
            vals = [float(x[m]) for x in per_row if x.get(m) is not None]
            overall[m] = spread(vals)

        # Rank metrics by discriminability (range first, then std).
        ranking = sorted(
            metrics,
            key=lambda m: (overall[m]["range"], overall[m]["std"]),
            reverse=True,
        )

        report["files"].append(
            {
                "file": str(path),
                "mode": mode,
                "n_rows_used": len(per_row),
                "overall_spread": overall,
                "spread_by_k": by_k,
                "discriminability_ranking": ranking,
                "rows_with_new_metrics": per_row,
            }
        )

    Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {args.output_json}")

    # Console quick view
    for f in report["files"]:
        print("\n" + "=" * 70)
        print(f"{f['mode'].upper()} | {f['file']}")
        print("=" * 70)
        print("Top metrics by discriminability:")
        for m in f["discriminability_ranking"][:5]:
            s = f["overall_spread"][m]
            print(f"  {m:24s} range={s['range']:.6f} std={s['std']:.6f} cv={s['cv']:.6f}")
        print("Per-K top metric (by range):")
        for k, stats in f["spread_by_k"].items():
            top = sorted(stats.items(), key=lambda kv: kv[1]["range"], reverse=True)[0]
            print(f"  K={k}: {top[0]} (range={top[1]['range']:.6f})")


if __name__ == "__main__":
    main()
