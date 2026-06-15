"""
This is the main function. All the tests happen here.
"""
import copy
import os
import pickle
import re
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from gnn_try2 import gnn_ucb_topk2
from restart_linUCB import restart_topk_linucb
from exp3s import exp3s_topk
from network import create_nodes, create_edges, create_edges_for_hubs, pol_L1, FJ_model_overT
from content_influence import gen_posts, generic_posts, gen_streams
from utils import (plot_polarization_comparison, plot_shared_initial_and_algorithm_finals, 
                   plot_regret_comparison, plot_graph, plot_rewards_comparison, 
                   plot_round_rewards_comparison, plot_depolarization_percentage_buckets,save_and_close_current_fig,_save_figures_before_show,
                   _figure_name, small)
from greedy import greedy_multi_objective_user_selection
from naive_greedy import greedy_naive
from random_baseline import random_baseline
from paretos import rlinucb_pareto, pareto_exp3s, gnn_pareto, greedy_pareto, plot_combined_pareto


def generate_three_graphs(N=100):
    """Generate 2 graphs with different configurations.
    """
    graphs = []
    
    print("Creating Graph 1 (Polarizing case, p_mod=0.01)...")
    G1 = create_nodes(N, case='pol', s_minus=-0.85, s_plus=0.85, p_mod=0.01)
    G1, s0_1, A1, W1, d_target1 = create_edges_for_hubs(G1, N, kappa=4, dtarget=14, dmin=10)
    graphs.append(G1)
    plot_graph(G1, title="Graph 1: Polarizing Case (p_mod=0.01)")

    print("Creating Graph 2 (Stable case, p_mod=0.01)...")
    G2 = create_nodes(N, case='stable', s_minus=-0.65, s_plus=0.65, p_mod=0.01)
    G2, s0_2, A2, W2, d_target2 = create_edges(G2, N, kappa=6, dtarget=14, dmin=10)
    graphs.append(G2)
    plot_graph(G2, title="Graph 2: Stable Case (p_mod=0.01)")
    

    return graphs, [d_target1, d_target2]

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

    # first we run FJ model over time
    fj_run_start = time.perf_counter()
    print(f"  Graph {graph_idx} - Running FJ model over T={T} rounds (seed={seed})...")
    G_fj = copy.deepcopy(G)

    raw_fj = FJ_model_overT(G_fj, T=T)
    # Ensure FJ result is a dict with keys `history` and `pol_values` (some implementations return a tuple)
    if isinstance(raw_fj, (tuple, list)):
        try:
            history_fj, pol_values_fj, z_vals_fj = raw_fj
        except Exception:
            # Fallback: wrap whole return value
            results_fj = {"raw": raw_fj}
        else:
            results_fj = {"history": history_fj, "pol_values": pol_values_fj, "z_vals": z_vals_fj}
    elif isinstance(raw_fj, dict):
        results_fj = raw_fj
    else:
        results_fj = {"raw": raw_fj}

    results_fj["runtime_seconds"] = time.perf_counter() - fj_run_start
    results["FJ_model_overT"] = results_fj
    print(f"FJ model runtime: {results_fj['runtime_seconds']:.2f}s")

    # GREEDY
    greedy_start = time.perf_counter()
    print(f"  Graph {graph_idx} - Running Greedy (seed={seed})...")
    G_greedy = copy.deepcopy(G)

    result_greedy = greedy_multi_objective_user_selection(G_greedy,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
         T=T,k_users=10,K=10,gamma=0.7,cache_fraction=0.5,seed=seed,lambda_cost=lambda_cost,drift=True,cost_ratio=0.6,cost_budget=None)

    results["greedy_multi_objective"] = result_greedy
    result_greedy["runtime_seconds"] = time.perf_counter() - greedy_start
    print(f"Greedy runtime: {result_greedy['runtime_seconds']:.2f}s")

    # RANDOM
    random_start = time.perf_counter()
    print(f"  Graph {graph_idx} - Running Random Baseline (seed={seed})...")
    G_random = copy.deepcopy(G)

    result_random = random_baseline(G_random,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
         T=T,k_users=10,K=10,gamma=0.7,cache_fraction=0.5,seed=seed,lambda_cost=lambda_cost,drift=True,cost_ratio=0.6,cost_budget=None)   
    results["random_baseline"] = result_random
    result_random["runtime_seconds"] = time.perf_counter() - random_start
    print(f"Random Baseline runtime: {result_random['runtime_seconds']:.2f}s")

    # EXP3.S
    print(f"Graph {graph_idx} - Running EXP3.S (seed={seed})...")
    G_exp3 = copy.deepcopy(G)

    exp3_start = time.perf_counter()
    result_exp3s = exp3s_topk(G_exp3,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
        T=T,k_users=10,K=10,gamma=0.7,eta=0.05,exp3_gamma=0.1,alpha_share=0.01,seed=seed,lambda_cost=lambda_cost,cost_ratio=0.6,drift=True,cost_budget=None)
    result_exp3s["runtime_seconds"] = time.perf_counter() - exp3_start

   
    results["exp3s_topk"] = result_exp3s
    print(f"EXP3.S runtime: {result_exp3s['runtime_seconds']:.2f}s")

    # GNN UCB
    print(f"Graph {graph_idx} - Running GNN Ensemble UCB (seed={seed})...")
    G_gnn = copy.deepcopy(G)

    gnn_start = time.perf_counter()
    result_gnn = gnn_ucb_topk2(G_gnn,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,M_g=M_g,M_f=M_f,
         T=T,k_users=10,K=10,d_target=d_target,gamma=0.7,seed=seed,
         drift=True,cache_fraction=0.5,hidden_dim=32,embedding_dim=16, n_models = 1,
         alpha=1.0,lr=5e-4,buffer_size=500,batch_size=64,train_epochs=10,
         warmup=500,epsilon=0.2,device="cpu", cost_ratio=0.6, train_every = 3 ,cost_budget=None)
    
    result_gnn["runtime_seconds"] = time.perf_counter() - gnn_start
     
    results['gnn_ucb_topk2'] = result_gnn
    print(f"GNN runtime: {result_gnn['runtime_seconds']:.2f}s")

    # Restart LinUCB
    print(f"  Graph {graph_idx} - Running Restart LinUCB (seed={seed})...")
    G_restart = copy.deepcopy(G)
    linucb_start = time.perf_counter()
    result_restart = restart_topk_linucb(G_restart,k=k,k_f=k_f,k_g=k_g,v_vec=v_vec,e_vec=e_vec,
        M_g=M_g,M_f=M_f,T=T,k_users=10,K=10,gamma=0.7,c=1.0,restart_period=500,lambda_reg=1.0,cache_fraction=0.5,
        sigma_reward=0.03,theta_path=None,drift=True,delta=0.05,seed=seed,lambda_cost=lambda_cost,cost_ratio=0.6,cost_budget=None)

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
        'FJ_model_overT': []
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

        # SAVE RESULTS IMMEDIATELY AFTER EACH SEED
        save_dir = "saved_results"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir,
            f"{graph_name.replace(' ', '_').replace(':', '')}_seed{seed}_T{T}_N{len(G.nodes)}.pkl"
        )

        with open(save_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Saved raw results to {save_path}")

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

def run_simple_comparison(N=100, T=1000, k=5, k_f=3, k_g=2, M_g=2, M_f=5, seed=42, seeds=None, show_fj_graph_idx=None):
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
        "Graph 1: Polarizing Case",
        "Graph 2: Stable Case",

        #"Graph 3: Depolarizing Case"
    ]
    
    # Algorithm names for plotting (must match keys returned by `run_algorithms_on_graph`)
    algo_names = ['greedy_multi_objective', 'gnn_ucb_topk2', 'exp3s_topk', 'restart_topk_linucb', 'random_baseline']
    label_mapping = {
        'random_baseline': 'Random',
        'greedy_multi_objective': 'Greedy Multi-Objective',
        'gnn_ucb_topk2': 'GNN-UCB',
        'exp3s_topk': 'EXP3.S',
        'restart_topk_linucb': 'Restart-LinUCB'
    }
    labels = [label_mapping[name] for name in algo_names]
    
    # (No combined FJ series collection) show FJ per-graph immediately instead

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

        # Show FJ model polarization-over-time immediately for this graph (single-graph view)
        fj_entry = results.get('FJ_model_overT', None)
        if fj_entry is not None:
            # Select pol series safely without relying on truthiness of numpy arrays
            fj_mean = None
            if 'pol_values_mean' in fj_entry and fj_entry['pol_values_mean'] is not None:
                fj_mean = fj_entry['pol_values_mean']
            elif 'pol_values' in fj_entry and fj_entry['pol_values'] is not None:
                fj_mean = fj_entry['pol_values']

            # Only show for the requested graph index (if provided)
            show_this = (show_fj_graph_idx is None) or (show_fj_graph_idx == graph_idx)
            if fj_mean is not None and show_this:
                print(f"  Showing FJ model polarization-over-time for {graph_name}...")
                small(fj_mean, T, label=f"FJ - {graph_name}")
                # Close the FJ plot immediately for the first graph
                if graph_idx == 0:
                    try:
                        plt.close('all')
                    except Exception:
                        pass
    
    print("\n" + "="*80)
    print("SIMPLE COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    # Choose one of the following:
    
    # Option 1: Run the main function with pareto analysis for single graph
    # main(N=50, T=1000, k=5, k_f=3, k_g=2, M_g=10, M_f=5)
    
    # Option 2: Run simple comparison (3 graphs x 4 algorithms with polarization and distributions)
    run_simple_comparison(N=50, T=3500, k=6, k_f=3, k_g=3, M_g=6, M_f=5, seeds=[1,2,3])

    # Option 3: Run pareto analysis for GNN-UCB (gnn_ucb_topk2)
    #run_pareto_for_all_graphs(N=20, T=2000, k=6, k_f=3, k_g=3, M_g=6, M_f=5,seeds=[1,42], cost_ratio_values=[0.0,0.2,0.3,0.4, 0.45,0.5,0.65,0.8,1.0,2.0])


