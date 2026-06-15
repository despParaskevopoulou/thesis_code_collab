import numpy as np
import random

from network import pol_L1, FJ_update, update_prejudices
from content_influence import (attention, one_step_user_reward, update_opinion,slate_cost, precompute_generic_cache_mask,
                                simple_cost_user_pool, greedy_cost_slate, gen_posts, greedy_slate,
                                  user_pool, generic_posts, cost_user_pool, slate_cost, cost_friendly_slate, item_cost,
                                  item_q, item_z, item_source, greedy_cost_constrained_slate,cost_pool)

"""
Here is the Naive Greedy baseline.
"""

def greedy_naive(G, k, k_f, k_g, v_vec, e_vec, M_g, M_f,T=2000, k_users=5, K=1, d_target=10, 
                                          gamma=0.5, cache_fraction=0.5, seed=1, lambda_cost=0.0, weights=None, min_susceptibility=0.0, 
                                          max_susceptibility=1.0,cost_budget=2.0,drift=False,cost_ratio=None):
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
        # Round content
        z_now = np.array([float(G.nodes[n]['z']) for n in all_users], dtype=float)
        z_post_t, q_post_t = gen_posts(v_vec, e_vec, t, z_now)

        if (t + 1) % K == 0:
            FJ_update(G)
        # 1) Context-only scores (no slate lookahead)
        scores = []
        for u in all_users:
            z_u = float(G.nodes[u]['z'])
            deg_u = float(G.out_degree[u])  # influence proxy out degree indicates influence
            l_u = float(G.nodes[u].get('lambda', 0.5))  # susceptibility (not used in this version)
            extremity = min(abs(z_u), 1.0)
            influence = min(deg_u / max(N - 1, 1), 1.0)
            #score = (deg_u /(N - 1)) #+ (1/90000000)*abs(z_u) + (1/3)*l_u #w_ext * extremity + w_inf * influence
            score = influence
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
            slate = greedy_cost_constrained_slate(pool,z_before,k,k_f,k_g,gamma=gamma,alpha=0.7,epsilon=0.3,cost_ratio=cost_ratio,fallback_relax_cost=False,)

            if len(pool) > 0 and len(slate) == 0:
                    print("\nEMPTY SLATE DEBUG")
                    print(f"t={t}, user={user}")
                    print(f"z_user={z_before:.4f}")
                    print(f"pool size={len(pool)}")
                    print(f"k={k}, k_f={k_f}, k_g={k_g}")
                    print(f"alpha={0.7}, epsilon={0.3}")
                    print(f"cost_ratio={cost_ratio}")
                    print("first pool items:")
                    for it in pool[:10]:
                        print(
                            "z=", item_z(it),
                            "q=", item_q(it),
                            "cost=", item_cost(it),
                            "source=", item_source(it)
                        )

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
        
        if drift:
            update_prejudices(G, tau=0.005)
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
