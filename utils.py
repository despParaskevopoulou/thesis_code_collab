import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

"""
In this file there are all the plotting functions we need.
"""

import os
import re
import time

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
SAVE_FIGURES = True
SHOW_FIGURES = False
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
_FIG_COUNTER = 0

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
SAVE_FIGURES = True
_ORIGINAL_PLT_SHOW = plt.show
_MAIN_FIG_COUNTER = 0
RUN_ID = time.strftime("%Y%m%d_%H%M%S")

def save_and_close_fig(name="figure"):
    """
    Save current matplotlib figure and close it.

    This prevents plt.show() from blocking long experiments.
    """
    global _FIG_COUNTER

    os.makedirs(PLOTS_DIR, exist_ok=True)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()
    if not safe_name:
        safe_name = "figure"

    _FIG_COUNTER += 1
    path = os.path.join(PLOTS_DIR, f"{RUN_ID}_{_FIG_COUNTER:03d}_{safe_name}.png")

    if SAVE_FIGURES:
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {path}")

    if SHOW_FIGURES:
        plt.show(block=False)
        plt.pause(0.5)

    plt.close()

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

def plot_regret_comparison(results, labels, title_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    has_inst = False
    has_cum = False
    has_avg = False

    for res, label in zip(results, labels):
        inst = res.get("inst_regret_list")
        cum = res.get("regret_list")
        avg = res.get("avg_regret_list")

        if inst is not None and len(inst) > 0:
            axes[0].plot(range(len(inst)), inst, linewidth=1.8, label=label)
            has_inst = True

        if cum is not None and len(cum) > 0:
            axes[1].plot(range(len(cum)), cum, linewidth=2.0, label=label)
            has_cum = True

        if avg is not None and len(avg) > 0:
            axes[2].plot(range(len(avg)), avg, linewidth=2.0, label=label)
            has_avg = True

    axes[0].set_title(f"{title_prefix} - Instant Regret")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Regret")
    axes[0].grid(True, alpha=0.3)
    if has_inst:
        axes[0].legend(loc="best")

    axes[1].set_title(f"{title_prefix} - Cumulative Regret")
    axes[1].set_xlabel("Time step")
    axes[1].grid(True, alpha=0.3)
    if has_cum:
        axes[1].legend(loc="best")

    axes[2].set_title(f"{title_prefix} - Average Regret")
    axes[2].set_xlabel("Time step")
    axes[2].grid(True, alpha=0.3)
    if has_avg:
        axes[2].legend(loc="best")

    plt.tight_layout()
    plt.show()

def plot_regret_over_sqrt_t(result, title=r"$R(T)/\sqrt{T}$ over time"):
    cum_regret = result.get("regret_list", [])

    if cum_regret is None or len(cum_regret) == 0:
        print("No cumulative regret data found in result.")
        return

    cum_regret = np.asarray(cum_regret, dtype=float)
    t = np.arange(1, len(cum_regret) + 1)
    regret_over_sqrt_t = cum_regret / np.sqrt(t)

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, regret_over_sqrt_t, linewidth=2.0, color="tab:red", label=r"$R(T)/\sqrt{T}$")
    plt.xlabel("Round")
    plt.ylabel(r"$R(T)/\sqrt{T}$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def plot_graph(G, title="skibidi", pos=None, node_attr="opinion"):
    opinions = [G.nodes[n].get(node_attr, G.nodes[n].get("z", 0.0)) for n in G.nodes]
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    vmin, vmax = -1, 1
    cmap = plt.cm.coolwarm

    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw(
        G, pos,
        node_color=opinions,
        cmap=cmap,
        with_labels=False,
        node_size=100,
        edge_color='gray',
        vmin=vmin, vmax=vmax,
        ax=ax
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f'{node_attr} Value')
    ax.set_title(title)
    save_and_close_fig(title if title is not None else "graph")

def plot_polarization_comparison(pol_series, labels, title, T=None, colors=None):
    fig, ax = plt.subplots(figsize=(9, 6.5))

    styles = [
        ("-", "o"),
        ("--", "s"),
        ("-.", "^"),
        (":", "D"),
        ((0, (3, 1, 1, 1)), "*"),
        ((0, (5, 2)), "x"),
    ]

    # Default color map for specific labels
    default_colors = {
        "exp3s": "orange",
        "gnn": "red",
        "gnn_ensemble_ucb_topk": "red",
    }

    for i, (pol, label) in enumerate(zip(pol_series, labels)):
        linestyle, marker = styles[i % len(styles)]
        markevery = max(1, len(pol) // 12)
        markersize = 8 if marker == "*" else 5
        
        # Determine color: use provided colors, or default map, or None for auto
        if colors is not None and i < len(colors):
            color = colors[i]
        else:
            color = default_colors.get(label, None)
        
        ax.plot(
            range(len(pol)),
            pol,
            label=label,
            linewidth=2.3,
            linestyle=linestyle,
            marker=marker,
            markevery=markevery,
            markersize=markersize,
            color=color,
        )

    ax.set_title(title, fontsize=15, pad=12)
    if T is not None:
        ax.set_xlim(0, T-1)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$\pi(z) = \|z\| / N$")
    ax.set_xlabel("Time step")
    ax.grid(True)
    ax.legend(loc="upper right")
    save_and_close_fig(title if title is not None else "pol_comparison")

def plot_shared_initial_and_algorithm_finals(histories, labels, bins=20, title=None):
    if len(histories) == 0:
        return
    if len(histories) != len(labels):
        raise ValueError("histories and labels must have the same length")

    def first_last_values(history_obj):
        if isinstance(history_obj, tuple) and len(history_obj) >= 1:
            history_obj = history_obj[0]
        if len(history_obj) == 0:
            return [], []

        first = history_obj[0]
        last = history_obj[-1]

        if isinstance(first, dict):
            z0 = [float(v) for v in first.values()]
            zf = [float(v) for v in last.values()]
        else:
            z0 = list(map(float, first))
            zf = list(map(float, last))
        return z0, zf

    z0_ref, _ = first_last_values(histories[0])

    n_alg = len(histories)
    n_cols = n_alg + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharex=True, sharey=True)

    axes[0].hist(
        z0_ref,
        bins=bins,
        range=(-1, 1),
        edgecolor='black',
        color='gray',
        alpha=0.75,
    )
    axes[0].set_title("Initial z")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("Count")

    for idx, (history_obj, label) in enumerate(zip(histories, labels), start=1):
        _, zf = first_last_values(history_obj)
        axes[idx].hist(
            zf,
            bins=bins,
            range=(-1, 1),
            edgecolor='black',
            alpha=0.8,
        )
        axes[idx].set_title(f"Final z\n{label}")
        axes[idx].set_xlabel("z")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    save_and_close_fig(title if title is not None else "initial_final_histograms")

def plot_initial_final_hist(history, bins=20, title=None, ax=None, mode="split", final_label="Final z", final_color="steelblue"):
    # Accept either a history list or a (history, pol_values) tuple
    if isinstance(history, tuple) and len(history) >= 1:
        history = history[0]
    if len(history) == 0:
        return
    # History can be a list of dicts (node -> z) or list of sequences
    first = history[0]
    last = history[-1]
    if isinstance(first, dict):
        z0 = [float(v) for v in first.values()]
        zf = [float(v) for v in last.values()]
    else:
        z0 = list(map(float, first))
        zf = list(map(float, last))
    if mode == "split":
        if ax is None:
            fig, axx = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
            axes = axx
            show_fig = True
        else:
            # ax is expected to be a tuple/list of 2 axes
            axes = ax
            show_fig = False

        axes[0].hist(z0, bins=bins, range=(-1, 1), edgecolor='black', color='gray', alpha=0.7, label="Initial z")
        axes[0].set_title("Initial z")
        axes[0].set_xlabel("z")
        axes[0].set_ylabel("Count")

        axes[1].hist(zf, bins=bins, range=(-1, 1), edgecolor='black', color=final_color, alpha=0.8, label=final_label)
        axes[1].set_title(final_label)
        axes[1].set_xlabel("z")

        if title is not None and ax is None:
            fig.suptitle(title)

def plot_rewards_comparison(results, labels, title, T=None, colors=None):
    """
    Plot cumulative and true rewards over time for different algorithms.
    
    Parameters:
    -----------
    results : list
        List of algorithm result dictionaries
    labels : list
        List of algorithm labels
    title : str
        Title for the plot
    T : int, optional
        Number of time steps
    colors : list, optional
        List of colors for each algorithm
    """
    if len(results) == 0:
        return
    
    # Determine how many metrics we have
    has_scaled = any('cum_scaled_rewards' in res for res in results)
    has_true_depol = any('cum_true_depol_rewards' in res for res in results)
    has_observed = any('alg_observed_rewards' in res for res in results)
    has_expected = any('alg_expected_rewards' in res for res in results)
    has_true = any('alg_true_rewards' in res for res in results)
    
    num_plots = sum([has_scaled, has_true_depol, has_observed or has_expected or has_true])
    
    if num_plots == 0:
        print("No reward data available in results")
        return
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 4))
    
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot 1: Cumulative scaled rewards (from EXP3.S)
    if has_scaled:
        ax = axes[plot_idx]
        for res, label in zip(results, labels):
            if 'cum_scaled_rewards' in res:
                rewards = res['cum_scaled_rewards']
                ax.plot(range(len(rewards)), rewards, linewidth=2.0, label=label)
        
        ax.set_title("Cumulative Scaled Rewards")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Cumulative Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plot_idx += 1
    
    # Plot 2: Cumulative true depolarization rewards (from EXP3.S)
    if has_true_depol:
        ax = axes[plot_idx]
        for res, label in zip(results, labels):
            if 'cum_true_depol_rewards' in res:
                rewards = res['cum_true_depol_rewards']
                ax.plot(range(len(rewards)), rewards, linewidth=2.0, label=label)
        
        ax.set_title("Cumulative True Depolarization Rewards")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Cumulative Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plot_idx += 1
    
    # Plot 3: Observed/Expected/True rewards (from LinUCB)
    if has_observed or has_expected or has_true:
        ax = axes[plot_idx]
        
        for res, label in zip(results, labels):
            # Try different reward types
            if 'alg_true_rewards' in res and len(res['alg_true_rewards']) > 0:
                rewards = np.cumsum(res['alg_true_rewards'])
                ax.plot(range(len(rewards)), rewards, linewidth=2.0, label=f"{label} (True)")
            elif 'alg_observed_rewards' in res and len(res['alg_observed_rewards']) > 0:
                rewards = np.cumsum(res['alg_observed_rewards'])
                ax.plot(range(len(rewards)), rewards, linewidth=2.0, label=f"{label} (Observed)")
            elif 'alg_expected_rewards' in res and len(res['alg_expected_rewards']) > 0:
                rewards = np.cumsum(res['alg_expected_rewards'])
                ax.plot(range(len(rewards)), rewards, linewidth=2.0, label=f"{label} (Expected)")
        
        ax.set_title("Cumulative Algorithm Rewards")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Cumulative Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_round_rewards_comparison(results, labels, title, T=None):
    """
    Plot per-round (instantaneous) rewards for different algorithms.
    
    Parameters:
    -----------
    results : list
        List of algorithm result dictionaries
    labels : list
        List of algorithm labels
    title : str
        Title for the plot
    T : int, optional
        Number of time steps
    """
    if len(results) == 0:
        return
    
    has_scaled = any('round_scaled_rewards' in res for res in results)
    has_true_depol = any('round_true_depol_rewards' in res for res in results)
    has_cost = any('round_costs' in res for res in results)
    
    num_plots = sum([has_scaled, has_true_depol, has_cost])
    
    if num_plots == 0:
        print("No per-round reward data available")
        return
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 4))
    
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    if has_scaled:
        ax = axes[plot_idx]
        for res, label in zip(results, labels):
            if 'round_scaled_rewards' in res:
                # Smooth the rewards with a moving average
                rewards = res['round_scaled_rewards']
                window = min(50, len(rewards) // 10)
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    ax.plot(range(len(smoothed)), smoothed, linewidth=1.5, label=label, alpha=0.7)
                else:
                    ax.plot(range(len(rewards)), rewards, linewidth=1.5, label=label, alpha=0.7)
        
        ax.set_title("Per-Round Scaled Rewards (Smoothed)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plot_idx += 1
    
    if has_true_depol:
        ax = axes[plot_idx]
        for res, label in zip(results, labels):
            if 'round_true_depol_rewards' in res:
                rewards = res['round_true_depol_rewards']
                window = min(50, len(rewards) // 10)
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    ax.plot(range(len(smoothed)), smoothed, linewidth=1.5, label=label, alpha=0.7)
                else:
                    ax.plot(range(len(rewards)), rewards, linewidth=1.5, label=label, alpha=0.7)
        
        ax.set_title("Per-Round True Depolarization Rewards (Smoothed)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plot_idx += 1
    
    if has_cost:
        ax = axes[plot_idx]
        for res, label in zip(results, labels):
            if 'round_costs' in res:
                costs = res['round_costs']
                window = min(50, len(costs) // 10)
                if window > 1:
                    smoothed = np.convolve(costs, np.ones(window)/window, mode='valid')
                    ax.plot(range(len(smoothed)), smoothed, linewidth=1.5, label=label, alpha=0.7)
                else:
                    ax.plot(range(len(costs)), costs, linewidth=1.5, label=label, alpha=0.7)
        
        ax.set_title("Per-Round Costs (Smoothed)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Cost")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def small(pol3, T, label="Polarizing FJ"):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(range(len(pol3)), pol3, label=label, linewidth=2.5)

    ax.set_title('Polarization over time')
    ax.set_xlim(0, T)
    ax.set_ylabel(r"$\pi(z) = \|z\| / N$")
    ax.set_xlabel("Time step")
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.show()
