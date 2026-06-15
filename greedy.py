import numpy as np
import random

from network import pol_L1, FJ_update, update_prejudices
from content_influence import (attention, one_step_user_reward, update_opinion,slate_cost, precompute_generic_cache_mask,
                                simple_cost_user_pool, greedy_cost_slate, gen_posts, greedy_slate,
                                  user_pool, generic_posts, cost_user_pool, slate_cost, cost_friendly_slate, item_cost,
                                  item_q, item_z, item_source, greedy_cost_constrained_slate, greedy_soft_cost_slate, cost_pool)

"""
Here is the Greedy baseline.

def greedy_multi_objective_user_selection(G, k, k_f, k_g, v_vec, e_vec, M_g, M_f,T=2000, k_users=5, K=1, d_target=10, 
                                          gamma=0.5, cache_fraction=0.5, seed=1, lambda_cost=0.0, weights=None, min_susceptibility=0.0, 
                                          max_susceptibility=1.0,cost_budget=2.0,drift=False,):
    
    N = G.number_of_nodes()
    all_users = list(G.nodes)
    rng = np.random.default_rng(seed)
    
    if weights is None:
        #weights = (1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4)  # Default: equal weight
        weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)

    OBS_ROUNDS = {0, 20, 50, 70, 90, 100}

    history = [{n: G.nodes[n]['z'] for n in G.nodes}]
    pol_values = [pol_L1(G)]
    sim_points = []

    z_gen, q_gen = generic_posts(M_g, T)

    cumulative_regret = 0.0
    regret_list = []
    avg_regret_list = []
    inst_regret_list = []

    cumulative_reward = 0.0
    reward_list = []
    cum_reward_list = []
    
    # Per-round metrics for Pareto analysis (compatible with exp3s / restart_linUCB)
    round_true_depol_rewards = []
    round_costs = []
    cum_costs = []
    cum_net_rewards = []
    round_net_rewards = []
    selected_history = []
    cache_mask = precompute_generic_cache_mask(
        T=T,
        M_g=M_g,
        cache_fraction=cache_fraction,
        rng=rng,
    )

    def normalized_slate_cost(slate):
        if slate is None or len(slate) == 0:
            return 0.0
        return float(slate_cost(slate) / len(slate))

    for t in range(T):
        record = (t in OBS_ROUNDS)
        z_now = np.array([G.nodes[n]['z'] for n in all_users], dtype=float)
        z_post_t, q_post_t = gen_posts(v_vec, e_vec, t, z_now)

        # -------- Compute user context features --------
        contexts = {}
        user_properties = {}  # Store properties for scoring
        
        for u in all_users:
            z_u = G.nodes[u]['z']
            deg = float(G.in_degree[u])
            lambda_u = float(G.nodes[u].get('lambda', 0.5))  # Susceptibility
            mean_pol = np.mean([G.nodes[v]['z'] for v in G.predecessors(u)]) if G.in_degree[u] > 0 else 0.0

            # Context features (same as LinUCB for consistency)
            x_u = np.array([
                z_u,
                abs(z_u),
                deg,
                mean_pol
            ], dtype=float)

            contexts[u] = x_u
            
            # Store properties for greedy scoring
            user_properties[u] = {
                'z': z_u,
                'extremity': abs(z_u),
                #'susceptibility': lambda_u,
                'influence': deg,
                'neighbor_mean': mean_pol,
            }

        # -------- Build slates for every user (so we can reuse them) --------

        """"""
        action_info = {}
        
        # -------- Greedy user selection scores --------
        scores = []
        for u in all_users:
            z_before = G.nodes[u]['z']

            z_user = G.nodes[u]['z']
            pool = cost_pool(G, u, t, M_f, M_g, z_post_t, q_post_t, z_gen, q_gen, cache_mask, d_target=d_target, rng=rng, use_predecessors=True)
            if len(pool) == 0:
                slate = []
            else:
                slate = greedy_cost_constrained_slate(pool, z_user, k, k_f, k_g, gamma=gamma, alpha=0.7, epsilon=0.3, cost_budget=cost_budget, fallback_relax_cost=False)

            action_info[u] = {
                'pool': pool,
                'slate': slate,
            }
            
            props = user_properties[u]
            
            # Normalize features to [0, 1]
            extremity = min(props['extremity'], 1.0)
            influence = min(props['influence'] / max(N - 1, 1), 1.0)  # Normalize by max degree
            
            w_ext,  w_inf, w_cost = weights
            
            score = w_ext * extremity  + w_inf * influence - w_cost * slate_cost(slate)
                
            scores.append(score)

        # Select top-k users by score
        scores_arr = np.array(scores)
        idx = np.argsort(scores_arr)[-k_users:]
        users = [all_users[i] for i in idx]

        # -------- Dynamic oracle regret --------
        oracle_rewards = []
        for u in all_users:
            r_u = one_step_user_reward(G, t, u, M_f, M_g, k, k_f, k_g, e_vec, v_vec, z_gen, q_gen, gamma=gamma)
            oracle_rewards.append(r_u)

        oracle_rewards = np.array(oracle_rewards, dtype=float)
        oracle_topk = np.sort(oracle_rewards)[-k_users:]
        oracle_round_reward = float(np.sum(oracle_topk))
        # ----------------------------------------

        rewards_this_round = []

        # Apply selected users' slates and observe rewards/costs
        round_true_depol = 0.0
        round_cost = 0.0
        round_net = 0.0

        for user in users:
            z_before = G.nodes[user]['z']
            slate = action_info[user]['slate']

            if len(slate) == 0:
                continue

            z_cont = np.array([item_z(it) for it in slate], dtype=float)
            q_cont = np.array([item_q(it) for it in slate], dtype=float)

            new_z = update_opinion(G, user, t, z_cont, q_cont, gamma=gamma)
            G.nodes[user]['z'] = new_z

            reward = float(abs(z_before) - abs(new_z))
            rewards_this_round.append(reward)
            cost_i = float(slate_cost(slate))

            net_reward = reward - lambda_cost * cost_i
            round_true_depol += reward

            round_cost += cost_i
            round_net += net_reward

        alg_round_reward = float(np.sum(rewards_this_round))
        cumulative_reward += alg_round_reward
        reward_list.append(alg_round_reward)
        cum_reward_list.append(cumulative_reward)

        # Track per-round metrics
        round_true_depol_rewards.append(round_true_depol)
        round_costs.append(round_cost)
        cumulative_cost = cum_costs[-1] if len(cum_costs) > 0 else 0.0
        cumulative_cost += round_cost
        cum_costs.append(cumulative_cost)

        round_net_rewards.append(round_net)
        cumulative_reward_net = cum_net_rewards[-1] if len(cum_net_rewards) > 0 else 0.0
        cumulative_reward_net += round_net
        cum_net_rewards.append(cumulative_reward_net)

        selected_history.append(users)

        regret = oracle_round_reward - alg_round_reward

        cumulative_regret += regret
        inst_regret_list.append(regret)
        regret_list.append(cumulative_regret)
        avg_regret_list.append(cumulative_regret / (t + 1))

        new_pol = pol_L1(G)
        pol_values.append(new_pol)

        if t % K == 0:
            FJ_update(G)

        if drift:
            update_prejudices(G, tau=0.005)

        history.append({n: G.nodes[n]['z'] for n in G.nodes})

        if t % 200 == 0 and t > 0:
            depol = pol_values[0] - new_pol
            print(
                f"Round {t:5d} | Pol: {new_pol:.4f} | "
                f"Depolarization: {depol:+.4f} | "
                f"Cum Regret: {cumulative_regret:8.2f}"
            )
            print(f"R_t / sqrt(t) = {cumulative_regret / np.sqrt(t):.3f}")

    return {
        "history": history,
        "pol_values": pol_values,
        "sim_points": sim_points,
        "regret_list": regret_list,
        "avg_regret_list": avg_regret_list,
        "inst_regret_list": inst_regret_list,
        "reward_list": reward_list,
        "cum_reward_list": cum_reward_list,
        # per-round metrics for Pareto analysis
        "round_true_depol_rewards": round_true_depol_rewards,
        "round_costs": round_costs,
        "cum_costs": cum_costs,
        "round_net_rewards": round_net_rewards,
        "cum_net_rewards": cum_net_rewards,
        "selected_history": selected_history,
    }
"""


def greedy_multi_objective_user_selection(G, k, k_f, k_g, v_vec, e_vec, M_g, M_f,T=2000, k_users=5, K=1, d_target=10, 
                                          gamma=0.5, cache_fraction=0.5, seed=1, lambda_cost=0.0, weights=None, min_susceptibility=0.0, 
                                          max_susceptibility=1.0,cost_budget=2.0,drift=False,cost_ratio=None,eta=0.4):
    rng = np.random.default_rng(seed)
    if weights is None:
        w_ext, w_inf = (0.5, 0.5)
    else:
        assert len(weights) == 2, "weights must be (w_ext, w_inf)."
        w_ext, w_inf = weights
    all_users = list(G.nodes)

    N = len(all_users)
    # Precompute generic content and cache mask
    z_gen, q_gen = generic_posts(M_g, T)
    cache_mask = precompute_generic_cache_mask(
        T=T, M_g=M_g, cache_fraction=cache_fraction, rng=rng
    )
    # Logs
    history = [{n: float(G.nodes[n]['z']) for n in G.nodes}]
    pol_values = [pol_L1(G)]
    reward_list, cum_reward_list = [], []
    round_true_depol_rewards = []
    round_costs, cum_costs = [], []
    selected_history = []
    cumulative_reward = 0.0
    cumulative_cost = 0.0
    for t in range(T):
        
        if (t + 1) % K == 0:
            FJ_update(G)

        # Round content
        z_now = np.array([float(G.nodes[n]['z']) for n in all_users], dtype=float)
        z_post_t, q_post_t = gen_posts(v_vec, e_vec, t, z_now)

        
        # 1) Context-only scores (no slate lookahead)
        scores = []
        for u in all_users:
            z_u = float(G.nodes[u]['z'])
            deg_u = float(G.out_degree[u])  # influence proxy out degree indicates influence
            l_u = float(G.nodes[u].get('lambda', 0.5))  # susceptibility (not used in this version)
            extremity = min(abs(z_u), 1.0)
            influence = min(deg_u / max(N - 1, 1), 1.0)
            #score = (deg_u /(N - 1)) #+ (1/90000000)*abs(z_u) + (1/3)*l_u #w_ext * extremity + w_inf * influence
            score = (4/5)*extremity + (1/5)*influence
            scores.append(score)

        scores = np.asarray(scores)
        chosen_idx = np.argsort(scores)[-k_users:]
        users = [all_users[i] for i in chosen_idx]
        selected_history.append(users)

        # 2) For selected users ONLY: build budget-feasible slates and apply
        rewards_this_round = []
        round_true_depol = 0.0
        round_cost = 0.0

        for user in users:
            z_before = float(G.nodes[user]['z'])
            pool = cost_pool(
                G, user, t, M_f, M_g, z_post_t, q_post_t,
                z_gen, q_gen, cache_mask,
                d_target=d_target, rng=rng, use_predecessors=True
            )
            if len(pool) == 0:
                continue
            slate = greedy_cost_constrained_slate(pool,z_before,k,k_f,k_g,gamma=gamma,cost_ratio=cost_ratio,)
            #slate = random_cost_constrained_slate(pool, z_before, k, k_f, k_g, alpha=0.4, cost_ratio=cost_ratio)
            #slate = greedy_soft_cost_slate(pool, z_before, k, k_f, k_g, gamma=gamma, alpha=0.4, epsilon=0.3, eta = cost_ratio)

            if len(pool) > 0 and len(slate) == 0:
                    print("\nEMPTY SLATE DEBUG")

                    continue

            z_cont = np.array([item_z(it) for it in slate], dtype=float)
            q_cont = np.array([item_q(it) for it in slate], dtype=float)
            z_after = update_opinion(G, user, t, z_cont, q_cont, gamma=gamma)
            G.nodes[user]['z'] = float(z_after)
            reward = float(abs(z_before) - abs(z_after))
            rewards_this_round.append(reward)
            c_i = float(slate_cost(slate))
            round_cost += c_i
            round_true_depol += reward

        # 3) Logging
        alg_round_reward = float(np.sum(rewards_this_round))
        cumulative_reward += alg_round_reward
        reward_list.append(alg_round_reward)
        cum_reward_list.append(cumulative_reward)
        cumulative_cost += round_cost
        round_costs.append(round_cost)
        cum_costs.append(cumulative_cost)
        round_true_depol_rewards.append(round_true_depol)

        if t % 200 == 0 and t > 0:
            new_pol = pol_L1(G)
            depol = pol_values[0] - new_pol
            print(
                f"Round {t:5d} | Pol: {new_pol:.4f} | "
                f"Depolarization: {depol:+.4f} | "
                f"Cumulative Reward: {cumulative_reward:.2f} | "
                f"Cumulative Cost: {cumulative_cost:.2f}"
            )
        # 4) Polarization and FJ timing
        new_pol = pol_L1(G)
        pol_values.append(new_pol)
        # Apply FJ after actions for fairness across methods
        
        if drift :
            update_prejudices(G, tau=0.001)
        history.append({n: float(G.nodes[n]['z']) for n in G.nodes})

    return {
        "history": history,
        "pol_values": pol_values,
        "reward_list": reward_list,
        "cum_reward_list": cum_reward_list,
        "round_true_depol_rewards": round_true_depol_rewards,
        "round_costs": round_costs,
        "cum_costs": cum_costs,
        "selected_history": selected_history,
    }

def greedy_multiobjective(*args, **kwargs):
    """Backward-compatible alias for greedy multi-objective user selection."""
    return greedy_multi_objective_user_selection(*args, **kwargs)