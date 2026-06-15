import copy
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from content_influence import gen_streams
from network import create_edges, create_nodes, create_edges_for_hubs
from restart_linUCB import restart_topk_linucb


FEATURE_NAMES = [
    "abs_z_user",
    "out_deg_norm",
    "local_abs_polarization",
    "mean_abs_neigh_z",
    "dep_proxy",
]


BASE_THETA_VARIANTS = {
    "default": [0.1,0.1, 0.1, 0.1, 0.1],
    "depol_heavy": [0.1,0.1, 0.1, 0.1, 8.0], 
    "network_aware": [8.0,0.1, 0.1, 0.1, 0.1], # network aware
    #"no_network": [0.0, 0.0, 0.0, 1.0],   # only depolarization proxy matters
    #"no_cost": [0.0, 0.3, 0.4, 0.9],
}


def normalize_theta(theta):
    theta = np.asarray(theta, dtype=float)
    norm = np.linalg.norm(theta)
    if norm <= 0:
        raise ValueError("theta must have non-zero norm")
    return theta / norm


def build_base_graph(N=50, seed=123):
    np.random.seed(seed)
    G = create_nodes(N, case="pol", s_minus=-0.85, s_plus=0.85, p_mod=0.05)
    G, _, _, _, d_target = create_edges_for_hubs(G, N, kappa=14, dtarget=8, dmin=3)
    return G, d_target


def summarize_run(res):
    pol_values = np.asarray(res["pol_values"], dtype=float)
    regret = np.asarray(res["regret_list"], dtype=float)
    cum_costs = np.asarray(res["cum_costs"], dtype=float)
    true_depol = np.asarray(res.get("alg_true_rewards", []), dtype=float)
    theta_err = np.asarray(res["theta_hat_err_list"], dtype=float)
    theta_cos = np.asarray(res["theta_hat_cos_list"], dtype=float)
    k_users = int(res.get("k_users", 1))

    initial_pol = float(pol_values[0]) if len(pol_values) else np.nan
    final_pol = float(pol_values[-1]) if len(pol_values) else np.nan

    return {
        "initial_pol": initial_pol,
        "final_pol": final_pol,
        "pol_reduction": initial_pol - final_pol,
        "relative_pol_reduction": 100.0 * (initial_pol - final_pol) / max(initial_pol, 1e-12),
        "final_regret": float(regret[-1]) if len(regret) else np.nan,
        "avg_regret": float(regret[-1] / len(regret)) if len(regret) else np.nan,
        "total_cost": float(cum_costs[-1]) if len(cum_costs) else 0.0,
        "avg_cost_per_selected_user": (
            float(cum_costs[-1] / max(1, len(pol_values) * k_users)) if len(cum_costs) else 0.0
        ),
        "cumulative_true_depol": float(np.sum(true_depol)) if len(true_depol) else 0.0,
        "final_theta_err": float(theta_err[-1]) if len(theta_err) else np.nan,
        "final_theta_cos": float(theta_cos[-1]) if len(theta_cos) else np.nan,
        "pol_values": pol_values,
        "pol_reduction_values": initial_pol - pol_values,
        "relative_pol_reduction_values": 100.0 * (initial_pol - pol_values) / max(initial_pol, 1e-12),
    }


def aggregate_runs(runs):
    scalar_keys = [
        "initial_pol",
        "final_pol",
        "pol_reduction",
        "relative_pol_reduction",
        "final_regret",
        "avg_regret",
        "total_cost",
        "avg_cost_per_selected_user",
        "cumulative_true_depol",
        "final_theta_err",
        "final_theta_cos",
    ]
    aggregated = {}

    for key in scalar_keys:
        values = np.asarray([run[key] for run in runs], dtype=float)
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    curves = [run["pol_values"] for run in runs]
    aggregated["pol_values_mean"] = np.mean(curves, axis=0)
    aggregated["pol_values_std"] = np.std(curves, axis=0)

    reduction_curves = [run["pol_reduction_values"] for run in runs]
    aggregated["pol_reduction_values_mean"] = np.mean(reduction_curves, axis=0)
    aggregated["pol_reduction_values_std"] = np.std(reduction_curves, axis=0)

    relative_reduction_curves = [run["relative_pol_reduction_values"] for run in runs]
    aggregated["relative_pol_reduction_values_mean"] = np.mean(relative_reduction_curves, axis=0)
    aggregated["relative_pol_reduction_values_std"] = np.std(relative_reduction_curves, axis=0)
    return aggregated


def save_summary_csv(aggregated, output_path):
    scalar_fields = [
        "variant",
        "base_theta",
        "final_pol_mean",
        "final_pol_std",
        "pol_reduction_mean",
        "pol_reduction_std",
        "relative_pol_reduction_mean",
        "relative_pol_reduction_std",
        "cumulative_true_depol_mean",
        "cumulative_true_depol_std",
        "avg_cost_per_selected_user_mean",
        "avg_cost_per_selected_user_std",
        "final_regret_mean",
        "final_regret_std",
        "final_theta_err_mean",
        "final_theta_err_std",
        "final_theta_cos_mean",
        "final_theta_cos_std",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_fields)
        writer.writeheader()
        for name, stats in aggregated.items():
            row = {
                "variant": name,
                "base_theta": np.array2string(stats["base_theta"], precision=4),
            }
            for field in scalar_fields:
                if field not in row and field in stats:
                    row[field] = stats[field]
            writer.writerow(row)


def plot_polarization_curves(aggregated, output_path):
    plt.figure(figsize=(10, 6))

    for name, stats in aggregated.items():
        mean_curve = stats["relative_pol_reduction_values_mean"]
        std_curve = stats["relative_pol_reduction_values_std"]
        x = np.arange(len(mean_curve))

        plt.plot(x, mean_curve, linewidth=2, label=name)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.12)

    plt.xlabel("Time step")
    plt.ylabel("Relative polarization reduction (%)")
    plt.title("Restart LinUCB Base-Theta Ablation")
    plt.grid(True, alpha=0.3)
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_base_theta_ablation(N=50,T=1000,k=5,k_f=3,k_g=2,M_g=10,M_f=5,k_users=1,restart_period=250,gamma=0.4,cost_budget=3.0,seeds=None,theta_variants=None,output_dir="plots",):
    if seeds is None:
        seeds = [1]
    if theta_variants is None:
        theta_variants = BASE_THETA_VARIANTS

    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Creating shared base graph for Restart LinUCB base-theta ablation...")
    G_base, d_target = build_base_graph(N)

    all_results = {}

    for name, theta in theta_variants.items():
        theta = normalize_theta(theta)
        variant_runs = []

        print("\n" + "=" * 90)
        print(f"Variant: {name}")
        print("theta order:", ", ".join(FEATURE_NAMES))
        print(f"normalized theta: {np.array2string(theta, precision=4)}")
        print("=" * 90)

        for seed in seeds:
            print(f"  seed={seed}")
            G = copy.deepcopy(G_base)

            np.random.seed(seed + 10_000)
            v_vec, e_vec = gen_streams(T, N, sigma_post=0.02, sigma_q=0.02)

            res = restart_topk_linucb(G,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,T=T,
                k_users=k_users,K=1,d_target=d_target,gamma=gamma,restart_period=restart_period,lambda_reg=1.0,
                sigma_reward=0.03,theta_path=None,drift=True,delta=0.05,seed=seed,cache_fraction=0.5,cost_budget=cost_budget,base_theta=theta,cost_ratio=3.0)

            variant_runs.append(summarize_run(res))

        all_results[name] = variant_runs

    aggregated = {}
    for name, runs in all_results.items():
        aggregated[name] = aggregate_runs(runs)
        aggregated[name]["base_theta"] = normalize_theta(theta_variants[name])

    print("\nRestart LinUCB base-theta ablation summary")
    print("-" * 120)
    print(
        "variant | final_pol(mean+-std) | pol_reduction(mean) | "
        "true_depol(mean) | avg_cost/user(mean) | regret(mean)"
    )
    print("-" * 120)

    best_variant = None
    best_final_pol = np.inf
    for name, stats in aggregated.items():
        if stats["final_pol_mean"] < best_final_pol:
            best_final_pol = stats["final_pol_mean"]
            best_variant = name

        print(
            f"{name:18s} | "
            f"{stats['final_pol_mean']:.4f}+-{stats['final_pol_std']:.4f} | "
            f"{stats['pol_reduction_mean']:.4f} | "
            f"{stats['cumulative_true_depol_mean']:.4f} | "
            f"{stats['avg_cost_per_selected_user_mean']:.4f} | "
            f"{stats['final_regret_mean']:.4f}"
        )

    print("-" * 120)
    print(f"Best variant by lowest mean final polarization: {best_variant}")

    csv_path = os.path.join(output_dir, "restart_linucb_base_theta_ablation.csv")
    plot_path = os.path.join(output_dir, "restart_linucb_base_theta_ablation.png")

    save_summary_csv(aggregated, csv_path)
    plot_polarization_curves(aggregated, plot_path)

    print(f"Saved CSV summary to: {csv_path}")
    print(f"Saved polarization plot to: {plot_path}")

    return {
        "all_results": all_results,
        "aggregated": aggregated,
        "best_variant": best_variant,
        "csv_path": csv_path,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    run_base_theta_ablation()
