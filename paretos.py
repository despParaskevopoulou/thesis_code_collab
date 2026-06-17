"""
This is the Pareto file
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
from network import create_nodes, create_edges, create_edges_for_hubs, pol_L1
from content_influence import gen_posts, generic_posts, gen_streams
from utils import (plot_polarization_comparison, plot_shared_initial_and_algorithm_finals, 
                   plot_regret_comparison, plot_graph, plot_rewards_comparison, 
                   plot_round_rewards_comparison, plot_depolarization_percentage_buckets,save_and_close_current_fig,_save_figures_before_show,
                   _figure_name)
from greedy import greedy_multi_objective_user_selection
from naive_greedy import greedy_naive
from random_baseline import random_baseline

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