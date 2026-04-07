"""
Step 2: Run NMFk on the concatenated spectrogram matrix to select k and learn W.

Uses nmf-torch with HALS solver for GPU-accelerated NMF. Implements NMFk-style
rank selection via silhouette-based stability + AIC, following the methodology
of Alexandrov & Vesselinov (SmartTensors/NMFk.jl).

For each candidate k:
  1. Run NMF n_runs times with perturbed inputs + varied initialization.
  2. Match components across runs via cosine similarity (Hungarian assignment).
  3. Compute per-component silhouette scores to measure stability.
  4. Compute AIC to penalize model complexity.
  5. Select k that maximizes silhouette while AIC is still decreasing.

Output: W matrix (n_mels x k), selected k, and diagnostics.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from nmf import run_nmf
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_nmf_for_k(V_np: np.ndarray, k: int, n_runs: int = 10,
                   max_iter: int = 500, seed: int = 42,
                   algo: str = "hals", use_gpu: bool = True,
                   perturb_std: float = 0.01) -> dict:
    """Run NMF n_runs times at rank k with bootstrap perturbation.

    V_np is (f, T) — frequency bins × time frames.
    Each run uses a slightly perturbed version of V (NMFk bootstrap).
    """
    f, T = V_np.shape
    V_norm = np.linalg.norm(V_np, "fro")
    best_error = float("inf")
    best_W, best_H = None, None
    errors = []
    W_runs = []

    k_start = time.time()

    for i in range(n_runs):
        run_start = time.time()

        # Bootstrap perturbation: add small noise to input matrix
        rng = np.random.RandomState(seed + i)
        noise = perturb_std * V_np.mean() * rng.randn(f, T).astype(np.float32)
        V_perturbed = np.maximum(V_np + noise, 1e-10)  # keep non-negative

        # nmf-torch expects (samples, features) = (T, f) = V.T
        Vt = V_perturbed.T

        # run_nmf returns H_nmf (T, k), W_nmf (k, f), err
        H_nmf, W_nmf, err = run_nmf(Vt, n_components=k, algo=algo,
                                      batch_max_iter=max_iter,
                                      random_state=seed + i,
                                      use_gpu=use_gpu,
                                      init="nndsvdar",
                                      beta_loss="frobenius")

        # Convert to our convention: W (f, k), H (k, T)
        W = W_nmf.T if not isinstance(W_nmf, np.ndarray) else W_nmf.T
        H = H_nmf.T if not isinstance(H_nmf, np.ndarray) else H_nmf.T

        if isinstance(W, torch.Tensor):
            W = W.cpu().numpy()
            H = H.cpu().numpy()

        # Compute error against ORIGINAL (unperturbed) V
        recon_error = np.linalg.norm(V_np - W @ H, "fro")
        relative_error = recon_error / V_norm
        errors.append(relative_error)
        W_runs.append(W)

        if recon_error < best_error:
            best_error = recon_error
            best_W = W
            best_H = H

        elapsed = time.time() - run_start
        log.info(
            f"  k={k} run {i+1}/{n_runs}: "
            f"rel_error={relative_error:.4f}, "
            f"time={elapsed:.1f}s"
        )

    # Compute silhouette-based stability
    silhouette = compute_silhouette_stability(W_runs)

    # Compute AIC
    aic = compute_aic(V_np, best_W, best_H, k)

    total_time = time.time() - k_start

    log.info(
        f"  k={k} DONE: "
        f"error={np.mean(errors):.4f}±{np.std(errors):.4f}, "
        f"silhouette={silhouette:.4f}, "
        f"AIC={aic:.1f}, "
        f"total={total_time:.1f}s"
    )

    return {
        "k": k,
        "best_W": best_W,
        "best_H": best_H,
        "mean_relative_error": float(np.mean(errors)),
        "std_relative_error": float(np.std(errors)),
        "silhouette": float(silhouette),
        "aic": float(aic),
        "total_time_s": round(total_time, 1),
    }


def compute_silhouette_stability(W_runs: list[np.ndarray]) -> float:
    """NMFk-style silhouette stability for component columns across runs.

    1. Align components across all runs to a reference (run 0) via
       Hungarian matching on cosine similarity.
    2. Stack all aligned component vectors into a single matrix.
    3. Compute silhouette score using the component index as the label.

    High silhouette (near 1.0) = components are tight within each group
    and well-separated from other components = good k.
    Low silhouette = components are bleeding into each other = k too high.
    """
    if len(W_runs) < 2:
        return 1.0

    n_runs = len(W_runs)
    k = W_runs[0].shape[1]

    # Use first run as reference
    W_ref = W_runs[0]
    W_ref_n = W_ref / (np.linalg.norm(W_ref, axis=0, keepdims=True) + 1e-10)

    # Align all runs to reference via Hungarian matching
    aligned_columns = [[] for _ in range(k)]  # k lists of vectors

    for W in W_runs:
        Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)
        # Cosine similarity matrix
        cos_sim = W_ref_n.T @ Wn  # (k, k)
        # Hungarian assignment (maximize similarity = minimize -similarity)
        row_ind, col_ind = linear_sum_assignment(-cos_sim)
        for ref_j, run_j in zip(row_ind, col_ind):
            aligned_columns[ref_j].append(Wn[:, run_j])

    # Stack into (n_runs * k, f) matrix with labels
    vectors = []
    labels = []
    for j in range(k):
        for vec in aligned_columns[j]:
            vectors.append(vec)
            labels.append(j)

    X = np.array(vectors)  # (n_runs * k, f)
    y = np.array(labels)   # (n_runs * k,)

    if len(np.unique(y)) < 2:
        return 1.0

    return float(silhouette_score(X, y, metric="cosine"))


def compute_aic(V: np.ndarray, W: np.ndarray, H: np.ndarray, k: int) -> float:
    """Compute AIC for NMF model selection.

    AIC = n * ln(RSS/n) + 2p
    where n = number of elements in V, p = number of NMF parameters.
    """
    f, T = V.shape
    n = f * T
    rss = np.sum((V - W @ H) ** 2)
    p = k * (f + T)  # parameters in W and H
    aic = n * np.log(rss / n + 1e-10) + 2 * p
    return float(aic)


def select_k(results: list[dict]) -> dict:
    """Select optimal k using NMFk criteria.

    Strategy:
    1. Find k values where silhouette is above threshold (stable solutions).
    2. Among those, find where AIC is minimized (best complexity tradeoff).
    3. If AIC keeps decreasing at the boundary, fall back to highest silhouette.
    4. If AIC minimum is at the first k tested, warn that k_min may be too high.
    """
    # Filter for reasonable silhouette (> 0.5 = "reasonable structure")
    stable = [r for r in results if r["silhouette"] > 0.5]
    if not stable:
        stable = results
        log.warning("No k values with silhouette > 0.5, using all results")

    # Check if AIC has a clear interior minimum
    aic_values = [r["aic"] for r in stable]
    min_aic_idx = np.argmin(aic_values)

    if min_aic_idx == len(stable) - 1:
        log.warning(
            f"AIC still decreasing at max k={stable[-1]['k']} tested "
            f"(AIC={stable[-1]['aic']:.1f}). The k range likely needs to be "
            f"extended upward to find the AIC elbow. "
            f"Falling back to highest silhouette."
        )
        best = max(stable, key=lambda r: r["silhouette"])
        best["selection_method"] = "silhouette_fallback_aic_boundary"
    elif min_aic_idx == 0:
        log.warning(
            f"AIC minimum at first k={stable[0]['k']} tested. "
            f"Consider lowering k_min to confirm this is a true minimum."
        )
        best = stable[min_aic_idx]
        best["selection_method"] = "aic_minimum_at_lower_boundary"
    else:
        best = stable[min_aic_idx]
        best["selection_method"] = "aic_interior_minimum"
        log.info(f"AIC found interior minimum at k={best['k']}")

    return best


def main():
    parser = argparse.ArgumentParser(description="Run NMFk rank selection (GPU, HALS)")
    parser.add_argument("--input-dir", type=str, default="nmf_analysis/output")
    parser.add_argument("--k-min", type=int, default=10,
                        help="Minimum rank to test (should be well below expected optimum)")
    parser.add_argument("--k-max", type=int, default=100,
                        help="Maximum rank to test (should be well above expected optimum "
                             "so AIC can find its elbow)")
    parser.add_argument("--k-step", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of NMF runs per k for stability estimation")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algo", type=str, default="hals",
                        choices=["hals", "halsvar", "mu", "bpp"],
                        help="NMF solver algorithm (default: hals)")
    parser.add_argument("--perturb-std", type=float, default=0.01,
                        help="Std of bootstrap noise relative to mean(V)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    input_dir = Path(args.input_dir)
    V_np = np.load(input_dir / "V_matrix.npy")  # (f, T)
    log.info(f"Loaded V: {V_np.shape} ({V_np.nbytes / 1e9:.2f} GB)")
    log.info(f"Solver: {args.algo}, perturbation std: {args.perturb_std}")

    # k cannot exceed min(f, T) — NNDSVD init needs that many singular vectors
    f, T = V_np.shape
    max_k = min(args.k_max, f, T)
    if max_k < args.k_max:
        log.warning(f"Capping k_max from {args.k_max} to {max_k} (V has {f} frequency bins)")

    k_values = [k for k in range(args.k_min, max_k + 1, args.k_step)]
    total_fits = len(k_values) * args.n_runs
    log.info(f"Testing k={k_values} ({len(k_values)} values × {args.n_runs} runs = {total_fits} total NMF fits)")

    results = []
    overall_start = time.time()

    for ki, k in enumerate(k_values):
        log.info(f"\n{'='*50}")
        log.info(f"k={k} ({ki+1}/{len(k_values)}) — starting")
        log.info(f"{'='*50}")

        res = run_nmf_for_k(V_np, k, n_runs=args.n_runs,
                             max_iter=args.max_iter, seed=args.seed,
                             algo=args.algo, perturb_std=args.perturb_std)
        results.append({
            "k": k,
            "mean_relative_error": res["mean_relative_error"],
            "std_relative_error": res["std_relative_error"],
            "silhouette": res["silhouette"],
            "aic": res["aic"],
            "total_time_s": res["total_time_s"],
        })

        # Save best W for this k
        np.save(input_dir / f"W_k{k}.npy", res["best_W"])

        # Running summary
        elapsed = time.time() - overall_start
        remaining_ks = len(k_values) - (ki + 1)
        avg_per_k = elapsed / (ki + 1)
        eta = avg_per_k * remaining_ks
        log.info(f"  Elapsed: {elapsed/60:.1f}min, ETA for remaining {remaining_ks} k-values: {eta/60:.1f}min")

    # Final summary
    log.info(f"\n{'='*60}")
    log.info("RESULTS SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"{'k':>5} | {'Error':>12} | {'Silhouette':>12} | {'AIC':>14} | {'Time':>8}")
    log.info("-" * 65)
    for r in results:
        log.info(f"{r['k']:>5} | {r['mean_relative_error']:>12.4f} | "
                 f"{r['silhouette']:>12.4f} | {r['aic']:>14.1f} | {r['total_time_s']:>7.1f}s")

    # Select optimal k
    best_result = select_k(results)
    best_k = best_result["k"]
    log.info(f"\nSelected k={best_k} "
             f"(silhouette={best_result['silhouette']:.4f}, "
             f"AIC={best_result['aic']:.1f}, "
             f"error={best_result['mean_relative_error']:.4f})")

    # Save the selected W as the global dictionary
    best_W = np.load(input_dir / f"W_k{best_k}.npy")
    np.save(input_dir / "W_global.npy", best_W)

    # Save diagnostics
    total_time = time.time() - overall_start
    diagnostics = {
        "k_values_tested": k_values,
        "results": results,
        "selected_k": best_k,
        "selection_criteria": {
            "silhouette": best_result["silhouette"],
            "aic": best_result["aic"],
            "error": best_result["mean_relative_error"],
            "method": best_result.get("selection_method", "unknown"),
        },
        "n_runs_per_k": args.n_runs,
        "algo": args.algo,
        "perturb_std": args.perturb_std,
        "total_time_minutes": round(total_time / 60, 1),
        "device": device,
    }
    with open(input_dir / "nmfk_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    log.info(f"\nSaved W_global.npy ({best_W.shape}) and nmfk_diagnostics.json")
    log.info(f"Total wall time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
