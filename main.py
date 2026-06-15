"""
This is the main function. All the tests happen here.
"""
import copy
import os
import re
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from gnn_try2 import gnn_ucb_topk2
from restart_linUCB import restart_topk_linucb
from exp3s import exp3s_topk
from network import create_nodes, create_edges, create_edges_for_hubs, pol_L1
from content_influence import gen_posts, generic_posts, gen_streams
from utils import (plot_polarization_comparison, plot_shared_initial_and_algorithm_finals, 
                   plot_regret_comparison, plot_graph, plot_rewards_comparison, 
                   plot_round_rewards_comparison)
from greedy import greedy_multi_objective_user_selection
from naive_greedy import greedy_naive
from random_baseline import random_baseline


PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
SAVE_FIGURES = True
_ORIGINAL_PLT_SHOW = plt.show
_MAIN_FIG_COUNTER = 0
RUN_ID = time.strftime("%Y%m%d_%H%M%S")


def _figure_name(fig, counter):
    title = ""
    if getattr(fig, "_suptitle", None) is not None:
        title = fig._suptitle.get_text()
    if not title and fig.axes:
        title = fig.axes[0].get_title()
    if not title:
        title = "figure"

    title = re.sub(r"[^A-Za-z0-9_.-]+", "_", title).strip("_").lower()
    if not title:
        title = "figure"
    return f"{RUN_ID}_main_{counter:03d}_{title[:80]}.png"

def _save_figures_before_show(*args, **kwargs):
    global _MAIN_FIG_COUNTER

    if SAVE_FIGURES:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            if getattr(fig, "_saved_from_main", False):
                continue
            _MAIN_FIG_COUNTER += 1
            fig.savefig(
                os.path.join(PLOTS_DIR, _figure_name(fig, _MAIN_FIG_COUNTER)),
                dpi=200,
                bbox_inches="tight",
            )
            fig._saved_from_main = True

    return _ORIGINAL_PLT_SHOW(*args, **kwargs)

plt.show = _save_figures_before_show

def save_and_close_current_fig(name):
    """
    Save current matplotlib figure and close it so experiments continue.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()
    path = os.path.join(PLOTS_DIR, f"{RUN_ID}_{safe_name}.png")

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {path}")

def generate_three_graphs(N=100):
    """Generate 2 graphs with different configurations.

    print("Creating Graph 1 (Stable case, p_mod=0.01)...")
    G1 = create_nodes(N, case='stable', s_minus=-0.65, s_plus=0.65, p_mod=0.01)
    G1, s0_1, A1, W1, d_target1 = create_edges(G1, N, kappa=6, dtarget=14, dmin=10)
    graphs.append(G1)
    plot_graph(G1, title="Graph 1: Stable Case (p_mod=0.01)")
    
    print("Creating Graph 3 (Polarizing case, p_mod=0.01)...")
    G3 = create_nodes(N, case='pol', s_minus=-0.85, s_plus=0.85, p_mod=0.01)
    G3, s0_3, A3, W3, d_target3 = create_edges_for_hubs(G3, N, kappa=8, dtarget=4, dmin=2)
    graphs.append(G3)
    plot_graph(G3, title="Graph 3: Polarizing Case (p_mod=0.01)")

    """
    graphs = []

    """
    print("Creating Graph 1 (Stable case, p_mod=0.01)...")
    G1 = create_nodes(N, case='stable', s_minus=-0.65, s_plus=0.65, p_mod=0.01)
    G1, s0_1, A1, W1, d_target1 = create_edges(G1, N, kappa=6, dtarget=14, dmin=10)
    graphs.append(G1)
    plot_graph(G1, title="Graph 1: Stable Case (p_mod=0.01)")

    print("Creating Graph 3 (Depolarizing case, p_mod=0.01)...")
    G3 = create_nodes(N, case='depol', s_minus=-0.65, s_plus=0.65, p_mod=0.01)
    G3, s0_3, A3, W3, d_target3 = create_edges(G3, N, kappa=1.4, dtarget=20, dmin=14)
    graphs.append(G3)
    plot_graph(G3, title="Graph 3: Depolarizing Case (p_mod=0.01)")

    """
    

    print("Creating Graph 2 (Polarizing case, p_mod=0.01)...")
    G2 = create_nodes(N, case='pol', s_minus=-0.85, s_plus=0.85, p_mod=0.01)
    G2, s0_2, A2, W2, d_target2 = create_edges_for_hubs(G2, N, kappa=8, dtarget=8, dmin=2)
    graphs.append(G2)
    plot_graph(G2, title="Graph 2: Polarizing Case (p_mod=0.01)")

    

    return graphs, [d_target2]

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
        avg_pol = float(np.trapz(pol) / len(pol))

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
    N = len(G.nodes)

    np.random.seed(seed + 10_000)
    v_vec, e_vec = gen_streams(T, N, sigma_post=0.02, sigma_q=0.02)

    results = {}

    greedy_start = time.perf_counter()
    print(f"  Graph {graph_idx} - Running Greedy (seed={seed})...")
    G_greedy = copy.deepcopy(G)

    result_greedy = greedy_multi_objective_user_selection(G_greedy,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
         T=T,k_users=5,K=5,gamma=0.4,cache_fraction=0.5,seed=seed,lambda_cost=lambda_cost,drift=True,cost_ratio=0.8,cost_budget=None)

    results["greedy_multi_objective"] = result_greedy
    result_greedy["runtime_seconds"] = time.perf_counter() - greedy_start
    print(f"Greedy runtime: {result_greedy['runtime_seconds']:.2f}s")

    """
    print(f"Graph {graph_idx} - Running EXP3.S (seed={seed})...")
    G_exp3 = copy.deepcopy(G)

    exp3_start = time.perf_counter()
    result_exp3s = exp3s_topk(G_exp3,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
        T=T,k_users=10,K=10,gamma=0.7,eta=0.05,exp3_gamma=0.1,alpha_share=0.01,seed=seed,lambda_cost=lambda_cost,cost_ratio=1.0,drift=True,cost_budget=None)
    result_exp3s["runtime_seconds"] = time.perf_counter() - exp3_start

   
    results["exp3s_topk"] = result_exp3s
    print(f"EXP3.S runtime: {result_exp3s['runtime_seconds']:.2f}s")
    """

    print(f"Graph {graph_idx} - Running GNN Ensemble UCB (seed={seed})...")
    G_gnn = copy.deepcopy(G)

    gnn_start = time.perf_counter()
    result_gnn = gnn_ucb_topk2(G_gnn,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
         T=T,k_users=5,K=5,d_target=d_target,gamma=0.4,seed=seed,
         drift=True,cache_fraction=0.5,hidden_dim=32,embedding_dim=16, n_models = 1,
         alpha=1.0,lr=5e-4,buffer_size=500,batch_size=64,train_epochs=10,
         warmup=500,epsilon=0.2,device="cpu", cost_ratio=0.8, train_every = 3 ,cost_budget=None)
    
    result_gnn["runtime_seconds"] = time.perf_counter() - gnn_start
     
    results['gnn_ucb_topk2'] = result_gnn
    print(f"GNN runtime: {result_gnn['runtime_seconds']:.2f}s")

    print(f"  Graph {graph_idx} - Running Restart LinUCB (seed={seed})...")
    G_restart = copy.deepcopy(G)
    linucb_start = time.perf_counter()
    result_restart = restart_topk_linucb(G_restart,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,
        M_g=M_g,M_f=M_f,T=T,k_users=5,K=5,gamma=0.4,c=1.0,restart_period=500,lambda_reg=1.0,cache_fraction=0.5,
        sigma_reward=0.03,theta_path=None,drift=True,delta=0.05,seed=seed,lambda_cost=lambda_cost,cost_ratio=0.8,cost_budget=None)

    results["restart_topk_linucb"] = result_restart
    result_restart["runtime_seconds"] = time.perf_counter() - linucb_start
    print(f"Restart LinUCB runtime: {result_restart['runtime_seconds']:.2f}s")

    return results

def _stack_seed_series(results_list, key):
    """
    Stack per-seed time series after padding shorter runs with their last value.
    This keeps early-stopped runs in the seed average instead of dropping them.
    """
    series = [np.asarray(r[key], dtype=float) for r in results_list if key in r and len(r[key]) > 0]
    if len(series) == 0:
        return None

    max_len = max(len(values) for values in series)
    padded = []
    for values in series:
        if len(values) < max_len:
            values = np.pad(values, (0, max_len - len(values)), mode="edge")
        padded.append(values)
    return np.vstack(padded)

def run_experiments_with_seeds(G, graph_idx, graph_name, seeds, T=5000, k=5, k_f=3, k_g=2, M_g=10, M_f=5, d_target=None):
    """
    Run experiments for multiple seeds and aggregate results.
    Returns aggregated results with mean and std across seeds.
    """
    all_seed_results = {
        'greedy_multi_objective': [],
        'gnn_ucb_topk2': [],
        'exp3s_topk': [],
        'restart_topk_linucb': [],
        'random_baseline': [],
        'greedy_naive': []
     }
    
    print(f"\n{'='*60}")
    print(f"Running experiments for {graph_name} Graph (Graph {graph_idx})")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}\n")
    
    for seed in seeds:
        print(f"Seed {seed}:")
        results = run_algorithms_on_graph(
            G, graph_idx, seed=seed, T=T, k=k, k_f=k_f, k_g=k_g, M_g=M_g, M_f=M_f, lambda_cost=0.0, d_target=d_target
        )
        for algo in all_seed_results.keys():
            if algo in results:
                all_seed_results[algo].append(results[algo])
    
    # Aggregate results: compute mean and std across seeds
    aggregated_results = {}
    for algo, results_list in all_seed_results.items():
        if len(results_list) > 0:
            pol_values = _stack_seed_series(results_list, "pol_values")
            aggregated_results[algo] = {
                'results': results_list,
                'pol_values_mean': np.mean(pol_values, axis=0),
                'pol_values_std': np.std(pol_values, axis=0),
            }
            
            # Handle reward aggregations - different algorithms track rewards differently
            # EXP3.S has cum_scaled_rewards and cum_true_depol_rewards
            if 'cum_scaled_rewards' in results_list[0]:
                cum_scaled_rewards = _stack_seed_series(results_list, "cum_scaled_rewards")
                if cum_scaled_rewards is not None:
                    aggregated_results[algo]['cum_scaled_rewards_mean'] = np.mean(cum_scaled_rewards, axis=0)
                    aggregated_results[algo]['cum_scaled_rewards_std'] = np.std(cum_scaled_rewards, axis=0)
            if 'cum_true_depol_rewards' in results_list[0]:
                cum_true_depol_rewards = _stack_seed_series(results_list, "cum_true_depol_rewards")
                if cum_true_depol_rewards is not None:
                    aggregated_results[algo]['cum_true_depol_rewards_mean'] = np.mean(cum_true_depol_rewards, axis=0)
                    aggregated_results[algo]['cum_true_depol_rewards_std'] = np.std(cum_true_depol_rewards, axis=0)
            
            # Restart-LinUCB has per-round rewards - aggregate directly
            if 'alg_true_rewards' in results_list[0]:
                alg_true_rewards = _stack_seed_series(results_list, "alg_true_rewards")
                if alg_true_rewards is not None:
                    aggregated_results[algo]['alg_true_rewards_mean'] = np.mean(alg_true_rewards, axis=0)
                    aggregated_results[algo]['alg_true_rewards_std'] = np.std(alg_true_rewards, axis=0)
            
            # Add cumulative costs if available
            if 'cum_costs' in results_list[0]:
                cum_costs = _stack_seed_series(results_list, "cum_costs")
                if cum_costs is not None:
                    aggregated_results[algo]['cum_costs_mean'] = np.mean(cum_costs, axis=0)
                    aggregated_results[algo]['cum_costs_std'] = np.std(cum_costs, axis=0)
    
    return aggregated_results
    """
    print(f"  Graph {graph_idx} - Running gnn_ensemble_ucb_topk...")
    res_gnn = gnn_ensemble_ucb_topk(G,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,
        M_f=M_f,T=T,k_users=5,K=1,gamma=0.5,seed=42,lambda_cost=0.1,drift=True,cache_fraction=0.5,
        n_models=5,hidden_dim=32,embedding_dim=16,alpha_ucb=1.0,lr=1e-3,buffer_size=1000,batch_size=32,
        train_epochs=5,train_every=1,epsilon=0.05,device="cpu",
        opinion_key="z",prejudice_key="z",)
    
    results['gnn_ensemble_ucb_topk'] = res_gnn

    print(f"  Graph {graph_idx} - Running linucb_s...")
    G3_copy = copy.deepcopy(G)
    result_linucb_s = linucb_s(
        G3_copy, k=k, k_f=k_f, k_g=k_g, v_vec=v_vec, e_vec=e_vec,
        M_g=M_g, M_f=M_f, T=T, k_users=1, K=1, gamma=0.3, c=1.0,
        window_size=None, lambda_reg=1.0, drift=True, delta=0.7, seed=42
    )
    results['linucb_s'] = result_linucb_s


    
    """
    
    
    return results

def pareto_exp3s(G,d_target,cost_ratio_values,seeds,exp3s_topk,k,k_f,k_g,v_vec,e_vec,
                 M_g,M_f,T=2000,k_users=5,K=1,gamma=0.5,seed=2,**kwargs):
    all_points = []

    for cost_ratio in cost_ratio_values:
        budget_costs = []
        budget_reductions = []
        budget_relative_reductions = []

        for seed in seeds:
            print(f"Running exp3.s with cost_ratio={cost_ratio}, seed={seed}...")
            G_run = copy.deepcopy(G)

            res = exp3s_topk(G_run,k=k,k_f=k_f,k_g=k_g,
                v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,T=T,
                k_users=k_users,K=K,d_target=d_target,gamma=gamma,seed=seed,cost_ratio=cost_ratio,drift=True,cache_fraction=0.5)

            # Average cost per selected user per round
            avg_cost = res["cum_costs"][-1] / (T * k_users)

            initial_pol = res["pol_values"][0]
            final_pol = res["pol_values"][-1]

            pol_reduction = initial_pol - final_pol
            rel_pol_reduction = 100 * (pol_reduction / max(initial_pol, 1e-12))

            budget_costs.append(avg_cost)
            budget_reductions.append(pol_reduction)
            budget_relative_reductions.append(rel_pol_reduction)

        all_points.append({
            "cost_ratio": cost_ratio,

            "avg_cost_mean": float(np.mean(budget_costs)),
            "avg_cost_std": float(np.std(budget_costs)),

            "pol_reduction_mean": float(np.mean(budget_reductions)),
            "pol_reduction_std": float(np.std(budget_reductions)),

            "rel_pol_reduction_mean": float(np.mean(budget_relative_reductions)),
            "rel_pol_reduction_std": float(np.std(budget_relative_reductions)),
        })

    all_points = sorted(all_points, key=lambda p: p["avg_cost_mean"])

    costs = [p["avg_cost_mean"] for p in all_points]
    cost_err = [p["avg_cost_std"] for p in all_points]

    reductions = [p["rel_pol_reduction_mean"] for p in all_points]
    reduction_err = [p["rel_pol_reduction_std"] for p in all_points]

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        costs,
        reductions,
        xerr=cost_err,
        yerr=reduction_err,
        marker="o",
        capsize=3
    )

    for p in all_points:
        plt.annotate(
            f'{p["cost_ratio"]:g}',
            (p["avg_cost_mean"], p["rel_pol_reduction_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    plt.xlabel("Average network cost per selected user")
    plt.ylabel("Relative polarization reduction")
    plt.title("EXP3.S Pareto curve by cost ratio averaged over seeds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_and_close_current_fig("exp3s_pareto")

    return all_points

def greedy_pareto(G, d_target, cost_ratio_values, seeds, greedy_multi_objective_user_selection, gamma,K,
                  k, k_f, k_g, v_vec, e_vec, M_g, M_f, T=2000, k_users=5,):
    all_points = []
    for cost_ratio in cost_ratio_values:
        budget_costs = []
        budget_reductions = []
        budget_relative_reductions = []

        for seed in seeds:
            print(f"Running Greedy Multi-Objective User Selection with cost_ratio={cost_ratio}, seed={seed}...")
            G_run = copy.deepcopy(G)

            res = greedy_multi_objective_user_selection(G_run,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
                T=T,k_users=5,K=5,gamma=0.4,cache_fraction=0.5,seed=seed,drift=True,cost_ratio=cost_ratio,cost_budget=None)

            avg_cost = res["cum_costs"][-1] / (T * k_users)

            initial_pol = res["pol_values"][0]
            final_pol = res["pol_values"][-1]

            pol_reduction = initial_pol - final_pol
            rel_pol_reduction = 100 * (pol_reduction / max(initial_pol, 1e-12))

            budget_costs.append(avg_cost)
            budget_reductions.append(pol_reduction)
            budget_relative_reductions.append(rel_pol_reduction)

        all_points.append({
            "cost_ratio": cost_ratio,

            "avg_cost_mean": float(np.mean(budget_costs)),
            "avg_cost_std": float(np.std(budget_costs)),

            "pol_reduction_mean": float(np.mean(budget_reductions)),
            "pol_reduction_std": float(np.std(budget_reductions)),

            "rel_pol_reduction_mean": float(np.mean(budget_relative_reductions)),
            "rel_pol_reduction_std": float(np.std(budget_relative_reductions)),
        })

    all_points = sorted(all_points, key=lambda p: p["avg_cost_mean"])

    costs = [p["avg_cost_mean"] for p in all_points]
    cost_err = [p["avg_cost_std"] for p in all_points]

    reductions = [p["rel_pol_reduction_mean"] for p in all_points]
    reduction_err = [p["rel_pol_reduction_std"] for p in all_points]

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        costs,
        reductions,
        xerr=cost_err,
        yerr=reduction_err,
        marker="o",
        capsize=3
    )

    for p in all_points:
        plt.annotate(
            f'{p["cost_ratio"]:g}',
            (p["avg_cost_mean"], p["rel_pol_reduction_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    plt.xlabel("Average network cost per selected user")
    plt.ylabel("Relative polarization reduction")
    plt.title("Greedy Pareto curve by cost ratio averaged over seeds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_and_close_current_fig("greedy_pareto")

    return all_points

def rlinucb_pareto(G, d_target, cost_ratio_values, seeds, restart_topk_linucb, k, k_f, k_g, v_vec, e_vec, M_g, M_f, T=2000, k_users=5, 
                   K=1, gamma=0.5, seed=2, **kwargs):
    all_points = []

    for cost_ratio in cost_ratio_values:
        budget_costs = []
        budget_reductions = []
        budget_relative_reductions = []

        for seed in seeds:
            print(f"Running Restart LinUCB with cost_ratio={cost_ratio}, seed={seed}...")
            G_run = copy.deepcopy(G)

            res = restart_topk_linucb(G_run,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,
        M_g=M_g,M_f=M_f,T=T,k_users=5,K=5,gamma=0.4,c=1.0,restart_period=500,lambda_reg=1.0,cache_fraction=0.5,
        sigma_reward=0.03,theta_path=None,drift=True,delta=0.05,seed=seed,cost_ratio=cost_ratio,cost_budget=None)

            avg_cost = res["cum_costs"][-1] / (T * k_users)

            initial_pol = res["pol_values"][0]
            final_pol = res["pol_values"][-1]

            pol_reduction = initial_pol - final_pol
            rel_pol_reduction = 100 * (pol_reduction / max(initial_pol, 1e-12))

            budget_costs.append(avg_cost)
            budget_reductions.append(pol_reduction)
            budget_relative_reductions.append(rel_pol_reduction)

        all_points.append({
            "cost_ratio": cost_ratio,

            "avg_cost_mean": float(np.mean(budget_costs)),
            "avg_cost_std": float(np.std(budget_costs)),

            "pol_reduction_mean": float(np.mean(budget_reductions)),
            "pol_reduction_std": float(np.std(budget_reductions)),

            "rel_pol_reduction_mean": float(np.mean(budget_relative_reductions)),
            "rel_pol_reduction_std": float(np.std(budget_relative_reductions)),
        })

    all_points = sorted(all_points, key=lambda p: p["avg_cost_mean"])

    costs = [p["avg_cost_mean"] for p in all_points]
    cost_err = [p["avg_cost_std"] for p in all_points]

    reductions = [p["rel_pol_reduction_mean"] for p in all_points]
    reduction_err = [p["rel_pol_reduction_std"] for p in all_points]

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        costs,
        reductions,
        xerr=cost_err,
        yerr=reduction_err,
        marker="o",
        capsize=3
    )

    for p in all_points:
        plt.annotate(
            f'{p["cost_ratio"]:g}',
            (p["avg_cost_mean"], p["rel_pol_reduction_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    plt.xlabel("Average network cost per selected user")
    plt.ylabel("Relative polarization reduction")
    plt.title("Restart LinUCB Pareto curve by cost ratio averaged over seeds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_and_close_current_fig("restart_linucb_pareto")

    return all_points

def gnn_pareto(G, d_target, cost_ratio_values, seeds, gnn_ensemble_ucb_topk, k, k_f,k_g, v_vec, e_vec, M_g, M_f, T=2000,
               k_users = 5, gamma=0.5, seed=2, **kwargs):
    all_points = []

    for cost_ratio in cost_ratio_values:
        budget_costs = []
        budget_reductions = []
        budget_relative_reductions = []

        for seed in seeds:
            print(f"Running GNN Ensemble UCB with cost_ratio={cost_ratio}, seed={seed}...")
            G_run = copy.deepcopy(G)

            res = gnn_ucb_topk2(G_run,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
                    T=T,k_users=5,K=5,d_target=d_target,gamma=0.4,seed=seed,
                    drift=True,cache_fraction=0.5,hidden_dim=32,embedding_dim=16, n_models = 1,
                    alpha=1.0,lr=5e-4,buffer_size=500,batch_size=64,train_epochs=10,
                    warmup=500,epsilon=0.2,device="cpu", cost_ratio=cost_ratio, train_every = 3 ,cost_budget=None)

            # Average cost per selected user per round
            avg_cost = res["cum_costs"][-1] / (T * k_users)

            initial_pol = res["pol_values"][0]
            final_pol = res["pol_values"][-1]

            pol_reduction = initial_pol - final_pol
            rel_pol_reduction = 100 * (pol_reduction / max(initial_pol, 1e-12))

            budget_costs.append(avg_cost)
            budget_reductions.append(pol_reduction)
            budget_relative_reductions.append(rel_pol_reduction)

        all_points.append({
            "cost_ratio": cost_ratio,

            "avg_cost_mean": float(np.mean(budget_costs)),
            "avg_cost_std": float(np.std(budget_costs)),

            "pol_reduction_mean": float(np.mean(budget_reductions)),
            "pol_reduction_std": float(np.std(budget_reductions)),

            "rel_pol_reduction_mean": float(np.mean(budget_relative_reductions)),
            "rel_pol_reduction_std": float(np.std(budget_relative_reductions)),
        })

    all_points = sorted(all_points, key=lambda p: p["avg_cost_mean"])

    costs = [p["avg_cost_mean"] for p in all_points]
    cost_err = [p["avg_cost_std"] for p in all_points]

    reductions = [p["rel_pol_reduction_mean"] for p in all_points]
    reduction_err = [p["rel_pol_reduction_std"] for p in all_points]

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        costs,
        reductions,
        xerr=cost_err,
        yerr=reduction_err,
        marker="o",
        capsize=3
    )

    for p in all_points:
        plt.annotate(
            f'{p["cost_ratio"]:g}',
            (p["avg_cost_mean"], p["rel_pol_reduction_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    plt.xlabel("Average network cost per selected user")
    plt.ylabel("Relative polarization reduction")
    plt.title("GNN UCB Pareto curve by cost ratio averaged over seeds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_and_close_current_fig("gnn_ucb_pareto")

    return all_points

def plot_combined_pareto(pareto_exp3s_points, pareto_linucb_points, pareto_gnn_points=None, pareto_greedy_points=None):
    """
    Plot Pareto curves in a single figure with error bars and legend.

    Parameters:
    - pareto_exp3s_points: list of dicts from pareto_exp3s()
    - pareto_linucb_points: list of dicts from rlinucb_pareto()
    - pareto_gnn_points: optional list of dicts from gnn_pareto()
    - pareto_greedy_points: optional list of dicts from greedy_pareto()
    """
    plt.figure(figsize=(10, 7))
    
    # Plot EXP3.S
    costs_exp3s = [p["avg_cost_mean"] for p in pareto_exp3s_points]
    cost_err_exp3s = [p["avg_cost_std"] for p in pareto_exp3s_points]
    reductions_exp3s = [p["rel_pol_reduction_mean"] for p in pareto_exp3s_points]
    reduction_err_exp3s = [p["rel_pol_reduction_std"] for p in pareto_exp3s_points]
    
    plt.errorbar(
        costs_exp3s,
        reductions_exp3s,
        xerr=cost_err_exp3s,
        yerr=reduction_err_exp3s,
        marker="o",
        capsize=3,
        label="EXP3.S",
        linewidth=2,
        markersize=8
    )
    
    # Plot Restart LinUCB
    costs_linucb = [p["avg_cost_mean"] for p in pareto_linucb_points]
    cost_err_linucb = [p["avg_cost_std"] for p in pareto_linucb_points]
    reductions_linucb = [p["rel_pol_reduction_mean"] for p in pareto_linucb_points]
    reduction_err_linucb = [p["rel_pol_reduction_std"] for p in pareto_linucb_points]
    
    plt.errorbar(
        costs_linucb,
        reductions_linucb,
        xerr=cost_err_linucb,
        yerr=reduction_err_linucb,
        marker="s",
        capsize=3,
        label="Restart-LinUCB",
        linewidth=2,
        markersize=8
    )

    if pareto_gnn_points is not None:
        costs_gnn = [p["avg_cost_mean"] for p in pareto_gnn_points]
        cost_err_gnn = [p["avg_cost_std"] for p in pareto_gnn_points]
        reductions_gnn = [p["rel_pol_reduction_mean"] for p in pareto_gnn_points]
        reduction_err_gnn = [p["rel_pol_reduction_std"] for p in pareto_gnn_points]

        plt.errorbar(
            costs_gnn,
            reductions_gnn,
            xerr=cost_err_gnn,
            yerr=reduction_err_gnn,
            marker="^",
            capsize=3,
            label="GNN-UCB",
            linewidth=2,
            markersize=8
        )

    if pareto_greedy_points is not None:
        costs_greedy = [p["avg_cost_mean"] for p in pareto_greedy_points]
        cost_err_greedy = [p["avg_cost_std"] for p in pareto_greedy_points]
        reductions_greedy = [p["rel_pol_reduction_mean"] for p in pareto_greedy_points]
        reduction_err_greedy = [p["rel_pol_reduction_std"] for p in pareto_greedy_points]

        plt.errorbar(
            costs_greedy,
            reductions_greedy,
            xerr=cost_err_greedy,
            yerr=reduction_err_greedy,
            marker="D",
            capsize=3,
            label="Greedy",
            linewidth=2,
            markersize=8
        )
    
    
    plt.xlabel("Average network cost per selected user", fontsize=12)
    plt.ylabel("Relative polarization reduction", fontsize=12)
    plt.title("Pareto Frontiers by Cost Ratio - All Algorithms", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="best")
    plt.tight_layout()
    save_and_close_current_fig("combined_pareto")

def run_pareto_for_all_graphs(N=100, T=1000, k=5, k_f=3, k_g=2, M_g=10, M_f=5, 
                               seeds=None, cost_ratio_values=None):
    """
    Run pareto analysis for all 3 graphs and create plots per algorithm with all 3 curves.
    
    Creates 3 plots:
    - Plot 1: EXP3.S pareto curves for all 3 graphs
    - Plot 2: Restart-LinUCB pareto curves for all 3 graphs
    - Plot 3: GNN-UCB pareto curves for all 3 graphs
    """
    if seeds is None:
        seeds = [1]
    if cost_ratio_values is None:
        cost_ratio_values = [0.0, 0.1, 0.25, 0.5, 0.75]
    
    print("="*80)
    print("PARETO ANALYSIS: ALL 3 GRAPHS")
    print("="*80)
    
    # Generate 3 graphs with d_targets
    graphs, d_targets = generate_three_graphs(N)
    graph_names = [
      #  "Graph 1: Stable",
        "Graph 2: Polarizing", 
      #  "Graph 3: Depolarizing"
    ]
    
    # Generate streams (same for all graphs)
    v_vec, e_vec = gen_streams(T, N, sigma_post=0.02, sigma_q=0.02)
    
    # Collect pareto results per algorithm and per graph
    all_exp3s_results = []
    all_linucb_results = []
    all_gnn_results = []
    all_greedy_results = []
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Run pareto for each graph: EXP3.S, Restart-LinUCB, Greedy (GNN commented out)
    for graph_idx, (G, graph_name, d_target) in enumerate(zip(graphs, graph_names, d_targets)):
        print(f"\n{'='*60}")
        print(f"Processing {graph_name}")
        print(f"{'='*60}")

        # Run Restart-LinUCB pareto
        print(f"Running Restart-LinUCB pareto for {graph_name}...")
        pareto_linucb_pts = rlinucb_pareto(
            G, d_target=d_target, cost_ratio_values=cost_ratio_values, seeds=seeds,
            restart_topk_linucb=restart_topk_linucb, k=k, k_f=k_f, k_g=k_g,
            v_vec=v_vec, e_vec=e_vec, M_g=M_g, M_f=M_f, T=T,
            k_users=5, K=5, gamma=0.4
        )
        all_linucb_results.append((graph_name, pareto_linucb_pts))

        # Run EXP3.S pareto
        print(f"Running EXP3.S pareto for {graph_name}...")
        pareto_exp3s_pts = pareto_exp3s(
            G, d_target=d_target, cost_ratio_values=cost_ratio_values, seeds=seeds,
            exp3s_topk=exp3s_topk, k=k, k_f=k_f, k_g=k_g,
            v_vec=v_vec, e_vec=e_vec, M_g=M_g, M_f=M_f, T=T,
            k_users=5, K=5, gamma=0.4, exp3_gamma=0.1, alpha_share=0.01
        )
        all_exp3s_results.append((graph_name, pareto_exp3s_pts))

        print(f"Running GNN ensemble UCB pareto for {graph_name}...")
        pareto_gnn_pts = gnn_pareto(
            G, d_target=d_target, cost_ratio_values=cost_ratio_values, seeds=seeds,
            gnn_ensemble_ucb_topk=gnn_ucb_topk2, k=k, k_f=k_f, k_g=k_g,
            v_vec=v_vec, e_vec=e_vec, M_g=M_g, M_f=M_f, T=T,
            k_users=5, K=5, gamma=0.4
        )
        all_gnn_results.append((graph_name, pareto_gnn_pts))
        

        # Run Greedy pareto
        print(f"Running Greedy pareto for {graph_name}...")
        pareto_greedy_pts = greedy_pareto(
            G, d_target=d_target, cost_ratio_values=cost_ratio_values, seeds=seeds,
            greedy_multi_objective_user_selection=greedy_multi_objective_user_selection,
            gamma=0.4, K=5, k=k, k_f=k_f, k_g=k_g, v_vec=v_vec, e_vec=e_vec,
            M_g=M_g, M_f=M_f, T=T, k_users=8,
        )
        all_greedy_results.append((graph_name, pareto_greedy_pts))

        # GNN pareto disabled for now; uncomment to run later
        # print(f"Running GNN-UCB pareto for {graph_name}...")
        # pareto_gnn_pts = gnn_pareto(...)
        # all_gnn_results.append((graph_name, pareto_gnn_pts))

        # Plot combined Pareto for this graph (include Greedy)
        plot_combined_pareto(pareto_exp3s_pts, pareto_linucb_pts, pareto_gnn_points=pareto_gnn_pts, pareto_greedy_points=pareto_greedy_pts)
        
    # # Plot 2: Restart-LinUCB Pareto curves for all 3 graphs
    print("Plotting Restart-LinUCB Pareto curves...")
    plt.figure(figsize=(10, 7))
    # 
    for (graph_name, points), marker, color in zip(all_linucb_results, markers, colors):
        costs = [p["avg_cost_mean"] for p in points]
        cost_err = [p["avg_cost_std"] for p in points]
        reductions = [p["rel_pol_reduction_mean"] for p in points]
        reduction_err = [p["rel_pol_reduction_std"] for p in points]
    #     
        plt.errorbar(costs, reductions, xerr=cost_err, yerr=reduction_err,
                      marker=marker, capsize=3, label=graph_name, linewidth=2, markersize=8, color=color)
     
    plt.xlabel("Average network cost per selected user", fontsize=12)
    plt.ylabel("Relative polarization reduction (%)", fontsize=12)
    plt.title("Restart-LinUCB Pareto Frontiers - All Graphs", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="best")
    plt.tight_layout()
    plt.show()

    # # Plot 1: EXP3.S Pareto curves for all 3 graphs
    print("\nPlotting EXP3.S Pareto curves...")
    plt.figure(figsize=(10, 7))
    # 
    for (graph_name, points), marker, color in zip(all_exp3s_results, markers, colors):
        costs = [p["avg_cost_mean"] for p in points]
        cost_err = [p["avg_cost_std"] for p in points]
        reductions = [p["rel_pol_reduction_mean"] for p in points]
        reduction_err = [p["rel_pol_reduction_std"] for p in points]
    #     
        plt.errorbar(costs, reductions, xerr=cost_err, yerr=reduction_err,
                      marker=marker, capsize=3, label=graph_name, linewidth=2, markersize=8, color=color)
    # 
    plt.xlabel("Average network cost per selected user", fontsize=12)
    plt.ylabel("Relative polarization reduction (%)", fontsize=12)
    plt.title("EXP3.S Pareto Frontiers - All Graphs", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="best")
    plt.tight_layout()
    plt.show()
    
    # Plot 3: GNN-UCB Pareto curves for all 3 graphs
    print("Plotting GNN-UCB Pareto curves...")
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for (graph_name, points), marker, color in zip(all_gnn_results, markers, colors):
        costs = [p["avg_cost_mean"] for p in points]
        cost_err = [p["avg_cost_std"] for p in points]
        reductions = [p["rel_pol_reduction_mean"] for p in points]
        reduction_err = [p["rel_pol_reduction_std"] for p in points]
        
        plt.errorbar(costs, reductions, xerr=cost_err, yerr=reduction_err,
                     marker=marker, capsize=3, label=graph_name, linewidth=2, markersize=8, color=color)
    
    plt.xlabel("Average network cost per selected user", fontsize=12)
    plt.ylabel("Relative polarization reduction (%)", fontsize=12)
    plt.title("GNN-UCB Pareto Frontiers - All Graphs", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="best")
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("PARETO ANALYSIS COMPLETE")
    print("="*80)

def main(N,T,k,k_f,k_g,M_g,M_f):
    """
    k_f : int
        Number of friend posts in slate
    k_g : int
        Number of generic posts in slate
    M_g : int
        Number of generic posts available
    M_f : int
        Number of friend posts to consider
    """
    
    print("="*80)
    print("GRAPH GENERATION AND ALGORITHM COMPARISON")
    print("="*80)
    
    # Generate 2 graphs
    graphs, d_target = generate_three_graphs(N)
    graph_names = ["Sparse strong-homophily stable graph"]
    
    # Define seeds for multiple runs
    seeds = [1,2,3,4,5]
    
    # ========================================================================
    # Run experiments with multiple seeds (NEW VERSION)
    # ========================================================================
    """
    all_results = []
    
    for graph_idx, (G, name) in enumerate(zip(graphs, graph_names)):
        aggregated_results = run_experiments_with_seeds(
            G, graph_idx + 1, name, seeds, T=T, k=k, k_f=k_f, 
            k_g=k_g, M_g=M_g, M_f=M_f
        )
        all_results.append((name, aggregated_results))
    
    """
    
    
    # ========================================================================
    # OLD CODE (COMMENTED OUT - kept for reference)
    # ========================================================================
    # Run Pareto plot for EXP3.S
    v_vec, e_vec = gen_streams(T, N, sigma_post=0.02, sigma_q=0.02)
    seeds = [1,2,3,4]
    cost_ratio_values =  [0.0, 0.2, 0.5, 0.7, 1.0]
    
    # Use first graph for pareto analysis
    G = graphs[0]

    #pareto_gnn_points = gnn_pareto(G,d_target=d_target,cost_ratio_values=cost_ratio_values, seeds=seeds, gnn_ensemble_ucb_topk=gnn_ucb_topk, k=k, k_f=k_f, k_g=k_g, v_vec=v_vec, e_vec=e_vec, M_g=M_g, M_f=M_f, T=T, k_users=1, K=1, gamma=0.3, seed=42)
    pareto_points_linucb = rlinucb_pareto(G, d_target=d_target, cost_ratio_values=cost_ratio_values, seeds=seeds, restart_topk_linucb=restart_topk_linucb, k=k, k_f=k_f, k_g=k_g, v_vec=v_vec, e_vec=e_vec, M_g=M_g, M_f=M_f, T=T, k_users=1, K=1, gamma=0.3, seed=42)
    pareto_points = pareto_exp3s(G, d_target=d_target, cost_ratio_values=cost_ratio_values, seeds=seeds, exp3s_topk=exp3s_topk, k=k, k_f=k_f, k_g=k_g, v_vec=v_vec, e_vec=e_vec, M_g=M_g, M_f=M_f, T=T, k_users=5, K=1, gamma=0.5, exp3_gamma=0.1, alpha_share=0.01)

    #plot_combined_pareto(pareto_points, pareto_points_linucb, pareto_gnn_points)

    # (Commented out algorithm runs)
    all_results = []
    
    for graph_idx, (G, name) in enumerate(zip(graphs, graph_names)):
        print(f"\n{'='*80}")
        print(f"Processing {name} Graph (Graph {graph_idx + 1})")
        print(f"{'='*80}")
         
        results = run_algorithms_on_graph(G, graph_idx + 1, seed=seeds ,T=T, k=k, k_f=k_f, 
                                           k_g=k_g, M_g=M_g, M_f=M_f)
        all_results.append((name, results))
    
    
    
    # ========================================================================
    # PLOTTING COMPARISONS FOR EACH GRAPH (Multi-seed version with aggregated results)
    # ========================================================================
    """
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*80}\n")
    
    for graph_name, aggregated_results in all_results:
        print(f"Plotting for {graph_name} graph...")
        
        algo_names = list(aggregated_results.keys())
        # Map algorithm names to proper display labels
        label_mapping = {
            'exp3s_topk': 'EXP3.S',
            'restart_topk_linucb': 'Restart-LinUCB',
            'linucb_s': 'LinUCB-S',
            'gnn_ensemble_ucb_topk': 'GNN-Ensemble-UCB',
            'greedy_multi_objective': 'Greedy Multi-Objective'
        }
        labels = [label_mapping.get(name, name) for name in algo_names]
        
        # Plot 1: Polarization Comparison with Error Bars (Multi-seed version)
        pol_series_mean = [aggregated_results[algo]['pol_values_mean'] for algo in algo_names]
        pol_series_std = [aggregated_results[algo]['pol_values_std'] for algo in algo_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (mean, std, label) in enumerate(zip(pol_series_mean, pol_series_std, labels)):
            ax.plot(range(len(mean)), mean, linewidth=2.0, label=label)
            ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
        
        ax.set_title(f"Polarization Over Time - {graph_name} Graph (Mean ± Std over {len(seeds)} seeds)")
        ax.set_xlabel("Time step")
        ax.set_ylabel(r"$\pi(z) = \|z\| / N$")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()
        
        # Plot 3: Initial vs Final Opinion Distribution (using first seed results)
        histories = [aggregated_results[algo]['results'][0]['history'] for algo in algo_names]
        plot_shared_initial_and_algorithm_finals(
            histories,
            labels,
            bins=20,
            title=f"Opinion Distribution - {graph_name} Graph (First Seed)"
        )
        
        
        # ========================================================================
        # OLD PLOTTING CODE (COMMENTED OUT - kept for reference)
        # ========================================================================
        
        # Original single-seed plotting code:
        pol_series = [results[algo]['pol_values'] for algo in algo_names]
        plot_polarization_comparison(
             pol_series, 
             labels, 
             f"Polarization Over Time - {graph_name} Graph",
             T=T
        )
    
        reward_results = [results[algo] for algo in algo_names]
        plot_rewards_comparison(
            reward_results,
            labels,
            f"Cumulative Rewards - {graph_name} Graph"
        )
        
        plot_round_rewards_comparison(
            reward_results,
            labels,
            f"Per-Round Rewards - {graph_name} Graph"
        )

          
        histories = [results[algo]['history'] for algo in algo_names]
        plot_shared_initial_and_algorithm_finals(
            histories,
            labels,
            bins=20,
            title=f"Opinion Distribution - {graph_name} Graph"
        )
        
    # 
    # ========================================================================
    # CROSS-GRAPH COMPARISON (Commented out)
    # ========================================================================
    # print(f"\n{'='*80}")
    # print("CROSS-GRAPH POLARIZATION COMPARISON")
    # print(f"{'='*80}\n")
    
    
    print("\n" + "="*80)
    print("COMPARISON STUDY COMPLETE")
    print("="*80)
    """

def run_simple_comparison(N=100, T=1000, k=5, k_f=3, k_g=2, M_g=2, M_f=5, seed=42, seeds=None):
    """
    Simple comparison: Generate 3 graphs, run 4 algorithms on each,
    create average polarization comparison plots and opinion distribution plots.
    
    Algorithms: Greedy, GNN-UCB, EXP3.S and Restart-LinUCB
    
    Output:
    - 3 polarization plots (one per graph, all algorithms overlaid)
    - 3 opinion distribution plots (one per graph, initial vs final from the first seed)
    """
    if seeds is None:
        seeds = [seed, seed + 1, seed + 2]
    
    print("="*80)
    print("SIMPLE COMPARISON: 3 GRAPHS x 4 ALGORITHMS")
    print(f"Seeds: {seeds}")
    print("="*80)
    
    # Generate 3 graphs (also returns d_targets)
    graphs, d_targets = generate_three_graphs(N)
    graph_names = [
        "Graph 1: Stable Case",
        "Graph 2: Polarizing Case",
        "Graph 3: Depolarizing Case"
    ]
    
    # Algorithm names for plotting (must match keys returned by `run_algorithms_on_graph`)
    algo_names = ['greedy_multi_objective', 'gnn_ucb_topk2', 'exp3s_topk', 'restart_topk_linucb', 'random_baseline','greedy_naive']
    label_mapping = {
        'random_baseline': 'Random',
        'greedy_naive': 'Greedy Naive',
        'greedy_multi_objective': 'Greedy Multi-Objective',
        'gnn_ucb_topk2': 'GNN-UCB',
        'exp3s_topk': 'EXP3.S',
        'restart_topk_linucb': 'Restart-LinUCB'
    }
    labels = [label_mapping[name] for name in algo_names]
    
    # Process each graph
    for graph_idx, (G, graph_name, d_target) in enumerate(zip(graphs, graph_names, d_targets)):
        print(f"\n{'='*80}")
        print(f"Processing {graph_name} (Graph {graph_idx + 1})")
        print(f"{'='*80}")
        
        # Run algorithms on this graph and average each time step over seeds.
        results = run_experiments_with_seeds(
            G, graph_idx + 1, graph_name, seeds, T=T, k=k, k_f=k_f,
            k_g=k_g, M_g=M_g, M_f=M_f, d_target=d_target
        )

        # Use a single reference initial polarization for all algorithms
        # to ensure improvement percentages are comparable.
        initial_pol_ref = pol_L1(G)

        print(f"  Mean final polarization improvement summary for {graph_name}:")
        for algo in algo_names:
            if algo not in results:
                continue

            pol_values = np.asarray(results[algo]["pol_values_mean"], dtype=float)
            pol_std = np.asarray(results[algo]["pol_values_std"], dtype=float)
            if len(pol_values) == 0:
                continue

            # Use reference initial polarization instead of per-algorithm first value
            initial_pol = float(initial_pol_ref)
            final_pol = float(pol_values[-1])
            final_pol_std = float(pol_std[-1]) if len(pol_std) > 0 else 0.0
            pol_change = initial_pol - final_pol
            pol_change_pct = 100.0 * pol_change / max(initial_pol, 1e-12)
            runtime_seconds = np.mean(
                [
                    r.get("runtime_seconds", 0.0)
                    for r in results[algo]["results"]
                    if r.get("runtime_seconds", None) is not None
                ]
            ) if any("runtime_seconds" in r for r in results[algo]["results"]) else None
            label = label_mapping.get(algo, algo)

            if runtime_seconds is None:
                print(
                    f"    {label}: final pol={final_pol:.4f} +/- {final_pol_std:.4f}, "
                    f"improvement={pol_change_pct:.2f}%"
                )
            else:
                print(
                    f"    {label}: final pol={final_pol:.4f} +/- {final_pol_std:.4f}, "
                    f"improvement={pol_change_pct:.2f}%, "
                    f"mean runtime={runtime_seconds:.2f}s"
                )
        
        # Extract polarization series for plotting
        present_algos = [algo for algo in algo_names if algo in results]
        pol_series = [results[algo]['pol_values_mean'] for algo in present_algos]
        filtered_labels = [label_mapping[algo] for algo in present_algos]

        # Debug check: print mapping between labels and algorithm keys to ensure alignment
        print("  Plot label mapping:")
        for lbl, algo in zip(filtered_labels, present_algos):
            print(f"    '{lbl}' -> {algo}")
        
        # Plot 1: Polarization Over Time
        print(f"  Creating polarization comparison plot for {graph_name}...")
        plot_polarization_comparison(
            pol_series,
            filtered_labels,
            f"Mean Polarization Over Time - {graph_name} ({len(seeds)} seeds)",
            T=T
        )
        
        # Plot 2: Opinion Distribution (Initial vs Final)
        print(f"  Creating opinion distribution plot for {graph_name} using the first seed...")
        histories = [results[algo]['results'][0]['history'] for algo in present_algos]
        plot_shared_initial_and_algorithm_finals(
            histories,
            filtered_labels,
            bins=20,
            title=f"Opinion Distribution - {graph_name} (seed {seeds[0]})"
        )

        base_pos = nx.spring_layout(G, seed=42)

        print(f"  Creating depolarization percentage bucket plot for {graph_name}...")
        plot_depolarization_percentage_buckets(
            results,
            algo_names=present_algos,
            label_mapping=label_mapping,
            title=f"Final Depolarization Percentage - {graph_name} ({len(seeds)} seeds)"
        )
    
    print("\n" + "="*80)
    print("SIMPLE COMPARISON COMPLETE")
    print("="*80)

def plot_depolarization_percentage_buckets(aggregated_results,algo_names=None,label_mapping=None,title="Depolarization Percentage by Algorithm",show_values=True,):
    import numpy as np
    import matplotlib.pyplot as plt

    if algo_names is None:
        algo_names = list(aggregated_results.keys())

    if label_mapping is None:
        label_mapping = {}

    labels = []
    means = []
    stds = []

    for algo in algo_names:
        if algo not in aggregated_results:
            continue

        seed_percentages = []

        # Use per-seed results, not only the mean trajectory
        for res in aggregated_results[algo]["results"]:
            pol = np.asarray(res["pol_values"], dtype=float)

            if len(pol) == 0:
                continue

            initial_pol = pol[0]
            final_pol = pol[-1]

            depol_pct = 100.0 * (initial_pol - final_pol) / max(initial_pol, 1e-12)
            seed_percentages.append(depol_pct)

        if len(seed_percentages) == 0:
            continue

        labels.append(label_mapping.get(algo, algo))
        means.append(np.mean(seed_percentages))
        stds.append(np.std(seed_percentages))

    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        edgecolor="black",
        alpha=0.85,
    )

    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Depolarization (%)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    if show_values:
        for bar, value in zip(bars, means):
            height = bar.get_height()
            offset = 1 if height >= 0 else -3

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{value:.1f}%",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
            )

    plt.tight_layout()
    save_and_close_current_fig("percentages")


if __name__ == "__main__":
    # Choose one of the following:
    
    # Option 1: Run the main function with pareto analysis for single graph
    # main(N=50, T=1000, k=5, k_f=3, k_g=2, M_g=10, M_f=5)
    
    # Option 2: Run simple comparison (3 graphs x 4 algorithms with polarization and distributions)
    #run_simple_comparison(N=20, T=2000, k=8, k_f=5, k_g=3, M_g=6, M_f=5, seeds=[1])

    # Option 3: Run pareto analysis for GNN-UCB (gnn_ucb_topk2)
    run_pareto_for_all_graphs(N=20, T=2000, k=6, k_f=3, k_g=3, M_g=14, M_f=5,seeds=[1,2], cost_ratio_values=[0.0,0.1,0.2,0.3,0.5,0.8,1.0,1.5, 2.0 ])

# run in this setup for 10000
# play with train_every in gnn
# a = 0.7 for the similarity in exp3s and gnn
