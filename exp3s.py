import numpy as np
import random

from network import pol_L1, FJ_update, update_prejudices
from content_influence import (attention, update_opinion,slate_cost, precompute_generic_cache_mask,
                                simple_cost_user_pool, greedy_cost_slate, gen_posts, greedy_slate,
                                  user_pool, generic_posts, cost_user_pool, slate_cost, cost_friendly_slate, item_cost,
                                  item_q, item_z, item_source, greedy_cost_constrained_slate,greedy_soft_cost_slate, hard_slate, cost_pool)

"""
In this file we implement the EXP3.S algorithm for user selection,
combined with the same slate construction and reward structure as in LinUCB. 
The main function is `exp3s_topk`, which takes the same parameters as `linucb_s` 
but with additional parameters specific to EXP3.S, such as `exp3_gamma` and `alpha_share`.

"""

def exp3s_topk(G,k,k_f,k_g,v_vec,e_vec,M_g,M_f,T=5000,k_users=5,K=1, d_target = 10,gamma=0.5,eta=0.05,
               exp3_gamma=0.1,alpha_share=0.01,seed=2,lambda_cost=0.0,cost_budget=2.0,drift=False,cache_fraction=0.5,eta_=0.4,cost_ratio=None,**kwargs):
    
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    all_users = list(G.nodes)
    N = len(all_users)

    # EXP3.S weights over users
    w = np.ones(N, dtype=float)

    history = [{n: G.nodes[n]["z"] for n in G.nodes}]
    pol_values = []

    round_scaled_rewards = []
    round_true_depol_rewards = []
    round_costs = []
    cum_scaled_rewards = []
    cum_true_depol_rewards = []
    cum_costs = []

    selected_history = []
    weight_history = []

    z_gen, q_gen = generic_posts(M_g, T)

    # New: pre-generate cache states for generic posts.
    cache_mask = precompute_generic_cache_mask(
        T=T,
        M_g=M_g,
        cache_fraction=cache_fraction,
        rng=rng,
    )

    cumulative_scaled_reward = 0.0
    cumulative_true_depol = 0.0
    cumulative_cost = 0.0

    for t in range(T):
        old_pol = pol_L1(G)

        if t % K == 0:
            FJ_update(G)
            
        gamma_t = max(0.02, exp3_gamma / np.sqrt(t + 1))
        z_now = np.array([G.nodes[n]["z"] for n in all_users], dtype=float)
        z_post_t, q_post_t = gen_posts(v_vec, e_vec, t, z_now)

        # --------------------------------------------------
        # Build slate for every user, exactly like LinUCB
        # --------------------------------------------------
        action_info = {}

        for user in all_users:
            z_user = float(G.nodes[user]["z"])
            prev_z_user = float(history[-1].get(user, z_user))

            #pool = simple_cost_user_pool(G, user, t, M_f, M_g, z_post_t, q_post_t, z_gen, q_gen, cache_mask, rng)
            pool = cost_pool(G,user,t,M_f,M_g,z_post_t,q_post_t,z_gen,q_gen,cache_mask,d_target=d_target,rng=rng,use_predecessors=True)

            if len(pool) == 0:
                action_info[user] = {
                    "pool": pool,
                    "slate": [],
                }
                continue

            #slate = greedy_cost_slate(pool,z_user,k,lambda_cost,k_f,k_g,alpha=0.4,epsilon=0.3)
            slate = greedy_cost_constrained_slate(pool, z_user, k, k_f, k_g, cost_ratio=cost_ratio)
            #slate = random_cost_constrained_slate(pool, z_user, k, k_f, k_g, alpha=0.4, cost_ratio=cost_ratio)
            #slate = hard_slate(pool, z_user, k, k_f, k_g, cost_budget=cost_budget)
            #slate = greedy_soft_cost_slate(pool, z_user, k, k_f, k_g, gamma=gamma, alpha=0.4, epsilon=0.3, eta = cost_ratio)

            action_info[user] = {
                "pool": pool,
                "slate": slate,
            }
        # --------------------------------------------------
        # EXP3.S probability distribution over users
        # --------------------------------------------------
        probs = (1.0 - gamma_t) * (w / np.sum(w)) + gamma_t / N
        probs = probs / np.sum(probs)

        # Select k_users without replacement
        selected_idx = rng.choice(
            N,
            size=k_users,
            replace=False,
            p=probs
        )

        selected_users = [all_users[i] for i in selected_idx]
        selected_history.append(selected_users)

        # --------------------------------------------------
        # Apply selected users' slates and observe rewards
        # --------------------------------------------------
        round_scaled_reward = 0.0
        round_true_depol = 0.0
        round_cost = 0.0
        round_net_reward = 0.0

        for i in selected_idx:
            user = all_users[i]
            slate = action_info[user]["slate"]

            z_before_round = {
                u: float(G.nodes[u]["z"])
                for u in all_users
            }

            if len(slate) == 0:
                # neutral reward for EXP3 update
                reward_i = 0.0
                print(f"User {user} has empty slate at round {t}. Assigning zero reward.")
                scaled_reward = 0.5
                raw_reward = 0.0
                cost_i = 0.0
            else:
                z_before = float(G.nodes[user]["z"])

                #z_cont = np.array([it[0] for it in slate], dtype=float)
                #q_cont = np.array([it[1] for it in slate], dtype=float)
                z_cont = np.array([item_z(it) for it in slate], dtype=float)
                q_cont = np.array([item_q(it) for it in slate], dtype=float)

                z_after = update_opinion(
                    G,
                    user,
                    t,
                    z_cont,
                    q_cont,
                    gamma=gamma
                )

                G.nodes[user]["z"] = z_after

                z_after_round = {
                    u: float(G.nodes[u]["z"])
                    for u in all_users
                }

                # True depolarization reward
                reward_i = float(abs(z_before) - abs(z_after))

                def neighborhood_depol_reward(G, user, z_before_round, z_after_round, include_self=True):
                    """
                    Graph-aware reward for selected user.
                    Measures depolarization in user's local neighborhood.
                    """

                    neigh = set(G.predecessors(user)) | set(G.successors(user))

                    if include_self:
                        neigh.add(user)

                    if len(neigh) == 0:
                        return abs(z_before_round[user]) - abs(z_after_round[user])

                    before = np.mean([abs(z_before_round[j]) for j in neigh])
                    after = np.mean([abs(z_after_round[j]) for j in neigh])

                    return float(before - after)

                #reward_i = neighborhood_depol_reward(G,user,z_before_round,z_after_round,include_self=True)
                reward_i = abs(z_before) - abs(z_after)

                # New: normalized network cost of the slate
                cost_i = slate_cost(slate)
                raw_reward = reward_i - lambda_cost * cost_i

                # EXP3.S requires reward in [0, 1]
                scaled_reward = float(np.clip((raw_reward + 1.0) / 2.0, 0.0, 1.0))

                """
                if t % 100 == 0:
                    print(
                        f"selected user={user}, "
                        f"z_before={z_before:.3f}, "
                        f"z_after={z_after:.3f}, "
                        f"reward={reward_i:.4f}, "
                        f"cost={cost_i:.2f}, "
                        f"raw={raw_reward:.4f}"
                    )
                """
                

            round_scaled_reward += scaled_reward
            round_true_depol += reward_i
            round_net_reward += raw_reward
            round_cost += cost_i

            # Importance-weighted reward estimate
            r_hat = scaled_reward / max(probs[i], 1e-12)

            # EXP3 update
            w[i] *= np.exp(eta * r_hat / k_users)
        """
        if t % 100 == 0:
            top_w_idx = np.argsort(w)[-10:][::-1]
            print(f"\nRound {t}")
            print("Top EXP3.S users by weight:")

            for idx in top_w_idx:
                u = all_users[idx]
                print(
                    f"user={u}, "
                    f"weight={w[idx]:.3f}, "
                    f"prob={probs[idx]:.3f}, "
                    f"z={G.nodes[u]['z']:.3f}, "
                    f"degree={G.degree[u]}"
                )
        """
        

        # --------------------------------------------------
        # EXP3.S fixed-share step
        # --------------------------------------------------
        total_w = np.sum(w)
        w = (1.0 - alpha_share) * w + alpha_share * total_w / N

        # Numerical stabilization
        w = np.maximum(w, 1e-12)
        w = w / np.mean(w)

        # --------------------------------------------------
        # Track rewards
        # --------------------------------------------------
        cumulative_scaled_reward += round_scaled_reward
        cumulative_true_depol += round_true_depol
        cumulative_cost += round_cost

        round_scaled_rewards.append(round_scaled_reward)
        round_true_depol_rewards.append(round_true_depol)
        round_costs.append(round_cost)

        cum_scaled_rewards.append(cumulative_scaled_reward)
        cum_true_depol_rewards.append(cumulative_true_depol)
        cum_costs.append(cumulative_cost)

        weight_history.append(w.copy())

        # --------------------------------------------------
        # Network dynamics
        # --------------------------------------------------
        new_pol = pol_L1(G)
        
        pol_values.append(new_pol)


        if drift:
            update_prejudices(G, tau=0.001)

        history.append({n: G.nodes[n]["z"] for n in G.nodes})

        if t % 200 == 0 and t > 0:
            print(
                f"Round {t:5d} | Pol: {new_pol:.4f} | "
                f"Cum true depol: {cumulative_true_depol:.3f} | "
                f"Cum scaled reward: {cumulative_scaled_reward:.3f}"
            )

    return {
        "G": G,
        "history": history,
        "pol_values": pol_values,

        "round_scaled_rewards": round_scaled_rewards,
        "round_true_depol_rewards": round_true_depol_rewards,
        "round_costs": round_costs,

        "cum_scaled_rewards": cum_scaled_rewards,
        "cum_true_depol_rewards": cum_true_depol_rewards,
        "cum_costs": cum_costs,

        "selected_history": selected_history,
        "weight_history": weight_history,

        "final_weights": w,
        "eta": eta,
        "exp3_gamma": exp3_gamma,
        "alpha_share": alpha_share,
        "k_users": k_users,
    }



