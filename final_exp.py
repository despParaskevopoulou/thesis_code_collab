# this is the final experiment
import copy
import os
import re
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from torch import seed

from gnn_ensemble_ucb import gnn_ucb_topk
from gnn_try2 import gnn_ucb_topk2
from restart_linUCB import restart_topk_linucb
from exp3s import exp3s_topk
from vanilla_linucb import linucb_s
from network import create_nodes, create_edges, create_edges_for_hubs
from content_influence import gen_posts, generic_posts, gen_streams
from utils import (plot_polarization_comparison, plot_shared_initial_and_algorithm_finals, 
                   plot_regret_comparison, plot_graph, plot_rewards_comparison, 
                   plot_round_rewards_comparison)
from greedy import greedy_multi_objective_user_selection

def summarize_result(result, method_name, graph_idx, seed, T):
    pol = np.asarray(result["pol_values"], dtype=float)

    if len(pol) == 0:
        final_pol = np.nan
        best_pol = np.nan
        best_round = np.nan
        avg_pol = np.nan
    else:
        final_pol = float(pol[-1])
        best_pol = float(np.min(pol))
        best_round = int(np.argmin(pol))
        avg_pol = float(np.trapezoid(pol) / len(pol))

    runtime = float(result.get("runtime_seconds", np.nan))
    cumulative_true_depol = float(result.get("cumulative_true_depol", np.nan))
    cumulative_cost = float(result.get("cumulative_cost", np.nan))

    if runtime > 0:
        depol_per_second = cumulative_true_depol / runtime
    else:
        depol_per_second = np.nan

    if len(pol) > 0 and runtime > 0:
        runtime_per_round = runtime / len(pol)
    else:
        runtime_per_round = np.nan

    return {
        "method": method_name,
        "graph_idx": graph_idx,
        "seed": seed,
        "T": T,
        "final_pol": final_pol,
        "best_pol": best_pol,
        "best_round": best_round,
        "avg_pol": avg_pol,
        "cumulative_true_depol": cumulative_true_depol,
        "cumulative_cost": cumulative_cost,
        "avg_cost_per_selected_user": float(result.get("avg_cost_per_selected_user", np.nan)),
        "runtime_seconds": runtime,
        "runtime_per_round": runtime_per_round,
        "depol_per_second": depol_per_second,
    }

def run_algorithms_on_graph(G, graph_idx, seed=42, T=1000, k=5,k_f=3, k_g=2, M_g=10, M_f=5,lambda_cost=0.0, d_target=None):
    """
    Apply EXP3.S and Restart-LinUCB to copies of the same graph.
    Both algorithms receive the same content streams and the same reward setting.
    """
    N = len(G.nodes)

    np.random.seed(seed + 10_000)
    v_vec, e_vec = gen_streams(T, N, sigma_post=0.02, sigma_q=0.02)

    results = {}

    greedy_start = time.perf_counter()
    print(f"  Graph {graph_idx} - Running Greedy (seed={seed})...")
    G_greedy = copy.deepcopy(G)

    result_greedy = greedy_multi_objective_user_selection(G_greedy,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
         T=T,k_users=5,K=5,gamma=0.7,cache_fraction=0.5,seed=seed,lambda_cost=lambda_cost,drift=True,cost_ratio=2,cost_budget=None)

    results["greedy_multi_objective"] = result_greedy
    result_greedy["runtime_seconds"] = time.perf_counter() - greedy_start
    print(f"Greedy runtime: {result_greedy['runtime_seconds']:.2f}s")

    """
    
    """

    print(f"Graph {graph_idx} - Running EXP3.S (seed={seed})...")
    G_exp3 = copy.deepcopy(G)

    exp3_start = time.perf_counter()
    result_exp3s = exp3s_topk(G_exp3,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
        T=T,k_users=5,K=5,gamma=0.7,eta=0.05,exp3_gamma=0.1,alpha_share=0.01,seed=seed,lambda_cost=lambda_cost,cost_ratio=2,drift=True,cost_budget=None)
    result_exp3s["runtime_seconds"] = time.perf_counter() - exp3_start

    results["exp3s_topk"] = result_exp3s
    print(f"EXP3.S runtime: {result_exp3s['runtime_seconds']:.2f}s")

    
    print(f"  Graph {graph_idx} - Running Restart LinUCB (seed={seed})...")
    G_restart = copy.deepcopy(G)

    linucb_start = time.perf_counter()
    result_restart = restart_topk_linucb(G_restart,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,
        M_g=M_g,M_f=M_f,T=T,k_users=5,K=5,gamma=0.7,c=1.0,restart_period=250,lambda_reg=1.0,
        sigma_reward=0.03,theta_path=None,drift=True,delta=0.05,seed=seed,lambda_cost=lambda_cost,cost_ratio=2,cost_budget=None)

    results["restart_topk_linucb"] = result_restart
    result_restart["runtime_seconds"] = time.perf_counter() - linucb_start
    print(f"Restart LinUCB runtime: {result_restart['runtime_seconds']:.2f}s")

    print(f"Graph {graph_idx} - Running GNN Ensemble UCB (seed={seed})...")
    G_gnn = copy.deepcopy(G)
    warmup_gnn = min(300, T // 4)
    gnn_start = time.perf_counter()
    result_gnn = gnn_ucb_topk2(G_gnn,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
         T=T,k_users=5,K=5,d_target=d_target,gamma=0.7,seed=seed,
         drift=True,cache_fraction=0.5,hidden_dim=32,embedding_dim=16, n_models = 3,
         alpha=1.0,lr=5e-4,buffer_size=500,batch_size=32,train_epochs=10,
         warmup=warmup_gnn,epsilon=0.2,device="cpu", cost_ratio=2, train_every =5,cost_budget=None)
    result_gnn["runtime_seconds"] = time.perf_counter() - gnn_start
     
    results['gnn_ucb_topk2'] = result_gnn
    print(f"GNN runtime: {result_gnn['runtime_seconds']:.2f}s")


    return results


def run_final_horizon_experiment(graphs,T_values=[500, 1000, 2000, 4000],seeds=[1, 2, 3, 4, 5],k=5,k_f=3,k_g=2,M_g=10,M_f=5,lambda_cost=0.0,d_target=None,):
    all_summaries = []
    all_results = {}

    for graph_idx, G in enumerate(graphs):

        for T in T_values:

            for seed in seeds:

                print("=" * 80)
                print(f"Graph {graph_idx} | T={T} | seed={seed}")
                print("=" * 80)

                results = run_algorithms_on_graph(G=G,graph_idx=graph_idx,seed=seed,T=T,k=k,k_f=k_f,k_g=k_g,M_g=M_g,M_f=M_f,lambda_cost=lambda_cost,d_target=d_target,)

                all_results[(graph_idx, T, seed)] = results

                for method_name, result in results.items():
                    summary = summarize_result(
                        result=result,
                        method_name=method_name,
                        graph_idx=graph_idx,
                        seed=seed,
                        T=T,
                    )

                    all_summaries.append(summary)

    df = pd.DataFrame(all_summaries)

    return df, all_results

def make_summary_table(df_summary, save_path=None):
    """
    Aggregate results over graph instances and seeds.
    """

    summary_table = (
        df_summary
        .groupby(["method", "T"])
        .agg(
            final_pol_mean=("final_pol", "mean"),
            final_pol_std=("final_pol", "std"),
            best_pol_mean=("best_pol", "mean"),
            best_pol_std=("best_pol", "std"),
            avg_pol_mean=("avg_pol", "mean"),
            avg_pol_std=("avg_pol", "std"),
            cumulative_true_depol_mean=("cumulative_true_depol", "mean"),
            cumulative_true_depol_std=("cumulative_true_depol", "std"),
            cumulative_cost_mean=("cumulative_cost", "mean"),
            cumulative_cost_std=("cumulative_cost", "std"),
            avg_cost_per_selected_user_mean=("avg_cost_per_selected_user", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            runtime_seconds_std=("runtime_seconds", "std"),
            runtime_per_round_mean=("runtime_per_round", "mean"),
            depol_per_second_mean=("depol_per_second", "mean"),
            depol_per_second_std=("depol_per_second", "std"),
        )
        .reset_index()
    )

    if save_path is not None:
        summary_table.to_csv(save_path, index=False)

    return summary_table


def plot_final_pol_vs_horizon(df_summary, save_path=None):
    grouped = (
        df_summary
        .groupby(["method", "T"])
        .agg(
            mean_final_pol=("final_pol", "mean"),
            std_final_pol=("final_pol", "std"),
        )
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for method in grouped["method"].unique():
        df_m = grouped[grouped["method"] == method]

        plt.errorbar(
            df_m["T"],
            df_m["mean_final_pol"],
            yerr=df_m["std_final_pol"],
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xlabel("Horizon T")
    plt.ylabel("Final polarization")
    plt.title("Final polarization as a function of horizon")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_avg_pol_vs_horizon(df_summary, save_path=None):
    grouped = (
        df_summary
        .groupby(["method", "T"])
        .agg(
            mean_avg_pol=("avg_pol", "mean"),
            std_avg_pol=("avg_pol", "std"),
        )
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for method in grouped["method"].unique():
        df_m = grouped[grouped["method"] == method]

        plt.errorbar(
            df_m["T"],
            df_m["mean_avg_pol"],
            yerr=df_m["std_avg_pol"],
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xlabel("Horizon T")
    plt.ylabel("Average polarization")
    plt.title("Average polarization over the horizon")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_runtime_vs_horizon(df_summary, save_path=None):
    grouped = (
        df_summary
        .groupby(["method", "T"])
        .agg(
            mean_runtime=("runtime_seconds", "mean"),
            std_runtime=("runtime_seconds", "std"),
        )
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for method in grouped["method"].unique():
        df_m = grouped[grouped["method"] == method]

        plt.errorbar(
            df_m["T"],
            df_m["mean_runtime"],
            yerr=df_m["std_runtime"],
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xlabel("Horizon T")
    plt.ylabel("Runtime seconds")
    plt.title("Runtime as a function of horizon")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_depol_per_second_vs_horizon(df_summary, save_path=None):
    grouped = (
        df_summary
        .groupby(["method", "T"])
        .agg(
            mean_depol_per_second=("depol_per_second", "mean"),
            std_depol_per_second=("depol_per_second", "std"),
        )
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for method in grouped["method"].unique():
        df_m = grouped[grouped["method"] == method]

        plt.errorbar(
            df_m["T"],
            df_m["mean_depol_per_second"],
            yerr=df_m["std_depol_per_second"],
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xlabel("Horizon T")
    plt.ylabel("Depolarization per second")
    plt.title("Runtime-normalized depolarization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_cost_vs_horizon(df_summary, save_path=None):
    grouped = (
        df_summary
        .groupby(["method", "T"])
        .agg(
            mean_cost=("cumulative_cost", "mean"),
            std_cost=("cumulative_cost", "std"),
        )
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for method in grouped["method"].unique():
        df_m = grouped[grouped["method"] == method]

        plt.errorbar(
            df_m["T"],
            df_m["mean_cost"],
            yerr=df_m["std_cost"],
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xlabel("Horizon T")
    plt.ylabel("Cumulative cost")
    plt.title("Cumulative intervention cost as a function of horizon")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_one_run_polarization_curves(all_results, graph_idx, T, seed, save_path=None):
    """
    Plot polarization curves for one graph, one horizon, and one seed.
    Useful for thesis figures.
    """

    results = all_results[(graph_idx, T, seed)]

    plt.figure(figsize=(8, 5))

    for method_name, result in results.items():
        pol = np.asarray(result["pol_values"], dtype=float)

        if len(pol) == 0:
            continue

        plt.plot(pol, label=method_name)

    plt.xlabel("Round")
    plt.ylabel("Polarization")
    plt.title(f"Polarization trajectories | graph={graph_idx}, T={T}, seed={seed}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_mean_polarization_curve_for_T(all_results, graph_idx, T, seeds, save_path=None):
    """
    For a fixed graph and horizon, plot mean polarization trajectory over seeds.
    Curves with different lengths are padded with NaN.
    """

    method_names = None

    for seed in seeds:
        key = (graph_idx, T, seed)
        if key in all_results:
            method_names = list(all_results[key].keys())
            break

    if method_names is None:
        print("No results found for this graph and T.")
        return

    plt.figure(figsize=(8, 5))

    for method_name in method_names:
        curves = []

        for seed in seeds:
            key = (graph_idx, T, seed)

            if key not in all_results:
                continue

            result = all_results[key][method_name]
            pol = np.asarray(result["pol_values"], dtype=float)

            if len(pol) == 0:
                continue

            curves.append(pol)

        if len(curves) == 0:
            continue

        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)

        for i, c in enumerate(curves):
            padded[i, :len(c)] = c

        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)

        x = np.arange(len(mean_curve))

        plt.plot(x, mean_curve, label=method_name)
        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15,
        )

    plt.xlabel("Round")
    plt.ylabel("Polarization")
    plt.title(f"Mean polarization trajectory | graph={graph_idx}, T={T}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def print_best_methods(summary_table):
    """
    Print the best method per horizon according to several criteria.
    """

    print("\nBest method per horizon by final polarization:")
    for T in sorted(summary_table["T"].unique()):
        df_T = summary_table[summary_table["T"] == T]
        best_row = df_T.loc[df_T["final_pol_mean"].idxmin()]
        print(
            f"T={T}: {best_row['method']} "
            f"(final_pol_mean={best_row['final_pol_mean']:.4f})"
        )

    print("\nBest method per horizon by average polarization:")
    for T in sorted(summary_table["T"].unique()):
        df_T = summary_table[summary_table["T"] == T]
        best_row = df_T.loc[df_T["avg_pol_mean"].idxmin()]
        print(
            f"T={T}: {best_row['method']} "
            f"(avg_pol_mean={best_row['avg_pol_mean']:.4f})"
        )

    print("\nBest method per horizon by depolarization per second:")
    for T in sorted(summary_table["T"].unique()):
        df_T = summary_table[summary_table["T"] == T]
        best_row = df_T.loc[df_T["depol_per_second_mean"].idxmax()]
        print(
            f"T={T}: {best_row['method']} "
            f"(depol_per_second_mean={best_row['depol_per_second_mean']:.4f})"
        )

def generate_three_graphs(N=100):
    """Generate 2 graphs with different configurations."""
    graphs = []
    
    # Create shared nodes for both stable graphs
    #print("Creating shared nodes for Stable graphs...")
    #base_nodes = create_nodes(N, case='mild', s_minus=-0.75, s_plus=0.75, p_mod=0.03)

    """
    # Graph 2: Stable case with p_mod=0.1 (same nodes)
    print("Creating Graph 2 (Depol case, p_mod=0.1)...")
    G2 = create_nodes(N, case='depol', s_minus=-0.85, s_plus=0.85, p_mod=0.05)
    G2, s0_2, A2, W2, d_target2 = create_edges(G2, N, kappa=2, dtarget=20, dmin=10)
    graphs.append(G2)
    plot_graph(G2, title="Graph 2: Stable Case (p_mod=0.1)")

    print("Creating Graph 1 (Polarizing case, p_mod=0.01)...")
    G1 = create_nodes(N, case='pol', s_minus=-0.85, s_plus=0.85, p_mod=0.01)
    G1, s0_1, A1, W1, d_target1 = create_edges_for_hubs(G1, N, kappa=6, dtarget=5, dmin=2)
    graphs.append(G1)
    plot_graph(G1, title="Graph 1: Polarizing Case (p_mod=0.01)")

    print("Creating Graph 1 Depolarizing Case...")
    G1 = create_nodes(N, case='depol', s_minus=-0.65, s_plus=0.65, p_mod=0.01)
    G1, s0_1, A1, W1, d_target1 = create_edges(G1, N, kappa=1.4, dtarget=20, dmin=14)
    graphs.append(G1)
    plot_graph(G1, title="Graph 1: Depolarization Case (p_mod=0.01)")

    # Graph 1: Stable case with p_mod=0.01
    print("Creating Graph 2 Stable Case...")
    G2 = create_nodes(N, case='stable', s_minus=-0.65, s_plus=0.65, p_mod=0.01)
    G2, s0_2, A2, W2, d_target2 = create_edges(G2, N, kappa=6, dtarget=14, dmin=10)
    graphs.append(G2)
    plot_graph(G2, title="Graph 2: Stable Case (p_mod=0.01)")
    """
    
    # Graph 3: Hyperpolarization case
    print("Creating Graph 3 Polarizing Case...")
    G3 = create_nodes(N, case='pol', s_minus=-0.75, s_plus=0.75, p_mod=0.05)
    G3, s0_3, A3, W3, d_target3 = create_edges_for_hubs(G3, N, kappa=15, dtarget=14, dmin=1)
    graphs.append(G3)
    plot_graph(G3, title="Graph 3: Polarizing Case (p_mod=0.01)")

    return graphs, [ d_target3]

if __name__ == "__main__":

    # --------------------------------------------------
    # Output folder
    # --------------------------------------------------
    output_dir = "final_experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Experiment parameters
    # --------------------------------------------------
    T_values = [500, 2500, 4000]
    seeds = [1, 2, 3]

    N = 50
    k = 5
    k_f = 3
    k_g = 2
    M_g = 10
    M_f = 5
    lambda_cost = 0.0
    d_target = None

    # --------------------------------------------------
    # Create graph regimes
    # --------------------------------------------------
    graphs, d_targets = generate_three_graphs(N=N)

    # Since your function currently returns one graph and one d_target:
    graphs = [graphs[0]]
    d_target = d_targets[0]

    print(f"Number of graphs used: {len(graphs)}")
    print(f"d_target used: {d_target}")

    # --------------------------------------------------
    # Run final experiment
    # --------------------------------------------------
    df_summary, all_results = run_final_horizon_experiment(
        graphs=graphs,
        T_values=T_values,
        seeds=seeds,
        k=k,
        k_f=k_f,
        k_g=k_g,
        M_g=M_g,
        M_f=M_f,
        lambda_cost=lambda_cost,
        d_target=d_target,
    )

    # --------------------------------------------------
    # Save raw summary
    # --------------------------------------------------
    raw_summary_path = os.path.join(output_dir, "final_horizon_experiment_raw_summary.csv")
    df_summary.to_csv(raw_summary_path, index=False)

    print("\nRaw summary saved to:")
    print(raw_summary_path)

    # --------------------------------------------------
    # Aggregate and save table
    # --------------------------------------------------
    summary_table_path = os.path.join(output_dir, "final_horizon_experiment_table.csv")

    summary_table = make_summary_table(
        df_summary,
        save_path=summary_table_path,
    )

    print("\nAggregated summary table:")
    print(summary_table)

    print("\nSummary table saved to:")
    print(summary_table_path)

    print_best_methods(summary_table)

    # --------------------------------------------------
    # Save plots
    # --------------------------------------------------
    plot_final_pol_vs_horizon(
        df_summary,
        save_path=os.path.join(output_dir, "final_polarization_vs_horizon.png"),
    )

    plot_avg_pol_vs_horizon(
        df_summary,
        save_path=os.path.join(output_dir, "average_polarization_vs_horizon.png"),
    )

    plot_runtime_vs_horizon(
        df_summary,
        save_path=os.path.join(output_dir, "runtime_vs_horizon.png"),
    )

    plot_depol_per_second_vs_horizon(
        df_summary,
        save_path=os.path.join(output_dir, "depolarization_per_second_vs_horizon.png"),
    )

    plot_cost_vs_horizon(
        df_summary,
        save_path=os.path.join(output_dir, "cumulative_cost_vs_horizon.png"),
    )

    # --------------------------------------------------
    # Example trajectory plots for thesis
    # --------------------------------------------------
    plot_one_run_polarization_curves(
        all_results,
        graph_idx=0,
        T=max(T_values),
        seed=seeds[0],
        save_path=os.path.join(output_dir, "example_polarization_trajectories.png"),
    )

    plot_mean_polarization_curve_for_T(
        all_results,
        graph_idx=0,
        T=max(T_values),
        seeds=seeds,
        save_path=os.path.join(output_dir, "mean_polarization_trajectory_graph0.png"),
    )

    print("\nFinal experiment completed.")