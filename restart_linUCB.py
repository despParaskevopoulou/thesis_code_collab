import numpy as np
import random

from network import pol_L1, FJ_update, update_prejudices
from content_influence import (attention, update_opinion,slate_cost, precompute_generic_cache_mask,
                                simple_cost_user_pool, greedy_cost_slate, gen_posts, greedy_slate,
                                  user_pool, generic_posts, cost_user_pool, slate_cost, cost_friendly_slate, item_cost,
                                  item_q, item_z, item_source, greedy_cost_constrained_slate, cost_pool, random_cost_constrained_slate)


"""
In this file we implement restart linUCB
"""

def restart_topk_linucb(G, k, k_f, k_g, v_vec, e_vec, M_g, M_f,T=5000, k_users=5, K=1, d_target = 10, gamma=0.5, c=1.0, d=None,restart_period=500, 
                        lambda_reg=1.0,sigma_reward=0.03,theta_path=None, drift=False,delta=0.05, seed=2, lambda_cost=0.0, 
                        cache_fraction=0.5,cost_budget=2.0,base_theta=None,cost_ratio=None):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    all_users = list(G.nodes)
    N = len(all_users)

    if base_theta is None:
        base_theta = np.array([
            0.1,    # bias
            0.1,   # abs(z_user)
            0.1,  # mismatch
        #    0.1,    # out_deg_norm
            0.1,    # mean_abs_neigh_z
            0.1,    # dep_proxy
            #0.1,    # quality
        ], dtype=float)
    else:
        base_theta = np.asarray(base_theta, dtype=float)
    base_theta = base_theta / np.linalg.norm(base_theta)

    if d is None:
        d = len(base_theta)

    if theta_path is None:
        # Default: stationary path, equivalent to vanilla LinUCB except for restarts.
        theta_path = np.tile(base_theta, (T, 1))
    else:
        theta_path = np.asarray(theta_path, dtype=float)
        assert theta_path.shape == (T, d), (
            f"theta_path must have shape {(T, d)}, got {theta_path.shape}"
        )

    S = float(np.max(np.linalg.norm(theta_path, axis=1)))

    q_clip = 3*0.02

    L = float(np.sqrt(
        1.0**2 +     # bias
        1.0**2 +     # |z_user|
        2.0**2 +     # mismatch
        1.0**2      # dep_proxy
        #q_clip**2    # quality feature
        #1.0**2       # normalized slate cost
    ))

    R_noise = sigma_reward

    A = lambda_reg * np.eye(d)
    b = np.zeros(d)
    local_round = 0

    history = [{n: G.nodes[n]["z"] for n in G.nodes}]
    pol_values = []

    regret_list = []
    inst_regret_list = []
    avg_regret_list = []
    scaled_regret_list = []

    oracle_rewards_list = []
    alg_expected_rewards_list = []
    alg_observed_rewards_list = []
    alg_true_rewards_list = []

    theta_hat_err_list = []
    theta_hat_cos_list = []
    theta_hat_dep_list = []

    round_costs = []
    cum_costs = []
    cumulative_cost = 0.0

    restart_points = []

    theory_beta_list = []

    cumulative_dynamic_regret = 0.0

    z_gen, q_gen = generic_posts(M_g, T)

    # New: pre-generate cache states for generic posts.
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

    def compute_phi(user, z_user, slate, prev_z_user=None):
        if prev_z_user is None:
            prev_z_user = z_user

        if slate is None or len(slate) == 0:
            return np.zeros(d, dtype=float), z_user, 0.0, 0.0

        #z_cont = np.array([it[0] for it in slate], dtype=float)
        #q_cont = np.array([it[1] for it in slate], dtype=float)
        z_cont = np.array([item_z(it) for it in slate], dtype=float)
        q_cont = np.array([item_q(it) for it in slate], dtype=float)
        mean_z = float(np.mean(z_cont))
        mean_q = float(np.mean(q_cont))

        _, z_bar = attention(z_user, z_cont, q_cont, beta=1.0, c=0.5)
        z_next_pred = (1.0 - gamma) * z_user + gamma * z_bar

        dep_proxy = abs(z_user) - abs(z_next_pred)

        q_feat = float(np.clip(np.log(max(mean_q, 1e-12)), -q_clip, q_clip))

        cost_feat = normalized_slate_cost(slate)

        if G.is_directed():
            neigh = list(set(list(G.predecessors(user)) + list(G.successors(user))))
        else:
            neigh = list(G.neighbors(user))

        if len(neigh) > 0:
            neigh_z = np.array(
                [float(G.nodes[j]["z"]) for j in neigh],
                dtype=float
            )
            mean_abs_neigh_z = float(np.mean(np.abs(neigh_z)))
        else:
            mean_abs_neigh_z = 0.0

        N = G.number_of_nodes()
        out_deg_norm = float(G.out_degree(user)) / max(1, N - 1)

        if G.is_directed():
            neigh = list(G.predecessors(user)) + list(G.successors(user))
        else:
            neigh = list(G.neighbors(user))

        neigh = list(set(neigh))

        if len(neigh) > 0:
            neigh_opinions = np.array(
                [float(G.nodes[j].get("z", 0.0)) for j in neigh],
                dtype=np.float32
            )

            mean_neigh_z = float(np.mean(neigh_opinions))
            local_disagreement = float(abs(z_user - mean_neigh_z))
            local_abs_polarization = float(np.mean(np.abs(neigh_opinions)))
        else:
            mean_neigh_z = 0.0
            local_disagreement = 0.0
            local_abs_polarization = 0.0

        phi = np.array([
        #    1.0,
            z_user,
           # abs(z_user),
            out_deg_norm,
            abs(mean_z - z_user),
            local_abs_polarization,
            #mean_abs_neigh_z,
            dep_proxy,
            #q_feat,
           # cost_feat,
        ], dtype=float)

        return phi, z_next_pred, dep_proxy, cost_feat
    
    def beta_t_fn(m):
        m = max(1, m)
        total_obs = max(1, k_users * m)

        logdet_bound = d * np.log(1.0 + (total_obs * L**2) / (lambda_reg * d))

        return (R_noise * np.sqrt(logdet_bound + 2.0 * np.log(1.0 / delta))+ np.sqrt(lambda_reg) * S)
    
    for t in range(T):

        if t%restart_period == 0 and t > 0:
            restart_points.append(t)
            A = lambda_reg * np.eye(d)
            b = np.zeros(d)
            local_round = 0
            print(f"Restarting LinUCB at time {t}...")

        if t % K == 0:
            FJ_update(G)

        local_round += 1

        theta_star_t = theta_path[t]
        beta_t = beta_t_fn(local_round)
        theory_beta_list.append(beta_t)

        z_now = np.array([G.nodes[n]["z"] for n in all_users], dtype=float)
        z_post_t, q_post_t = gen_posts(v_vec, e_vec, t, z_now)

        action_info = {}

        for user in all_users:
            z_user = float(G.nodes[user]["z"])
            #pool = simple_cost_user_pool(G, user, t, M_f, M_g, z_post_t, q_post_t, z_gen, q_gen, cache_mask, rng)
            pool = cost_pool(G,user,t,M_f,M_g,z_post_t,q_post_t,z_gen,q_gen,cache_mask,d_target,rng,use_predecessors=True)
            
            if len(pool) == 0:
                slate = []
            else:
                #slate = greedy_cost_slate(pool,z_user,k,lambda_cost,k_f,k_g,alpha=0.4,epsilon=0.3)
                slate = greedy_cost_constrained_slate(pool, z_user, k, k_f, k_g, cost_ratio=cost_ratio)
                #slate = random_cost_constrained_slate(pool, z_user, k, k_f, k_g, alpha=0.4, cost_ratio=cost_ratio)

            phi, z_next_pred, dep_proxy, cost_feat = compute_phi(user, z_user, slate)
            #phi, z_next_pred, dep_proxy = compute_phi(user, z_user, slate)

            mu = float(phi @ base_theta)

            action_info[user] = {
                "slate": slate,
                "phi": phi,
                "z_next_pred": z_next_pred,
                "dep_proxy": dep_proxy,
                "cost_feat": cost_feat,
                "mu": mu,
            }

        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)

        theta_hat = A_inv @ b

        scores = []

        for user in all_users:
            phi = action_info[user]["phi"]
            exploit = float(phi @ theta_hat)
            unc = float(np.sqrt(max(phi @ A_inv @ phi, 0.0)))
            score = exploit + beta_t * unc
            scores.append(score)

        selected_idx = np.argsort(scores)[-k_users:]
        selected_users = [all_users[i] for i in selected_idx]

        mus = np.array([action_info[user]["mu"] for user in all_users], dtype=float)

        oracle_round_reward = float(np.sum(np.sort(mus)[-k_users:]))
        alg_round_expected = float(
            np.sum([action_info[user]["mu"] for user in selected_users])
        )

        inst_regret = oracle_round_reward - alg_round_expected
        cumulative_dynamic_regret += inst_regret

        oracle_rewards_list.append(oracle_round_reward)
        alg_expected_rewards_list.append(alg_round_expected)

        regret_list.append(cumulative_dynamic_regret)
        inst_regret_list.append(inst_regret)
        avg_regret_list.append(cumulative_dynamic_regret / (t + 1))
        scaled_regret_list.append(cumulative_dynamic_regret / np.sqrt(t + 1))

        alg_round_observed = 0.0
        alg_round_true_depol = 0.0
        round_cost = 0.0

        for user in selected_users:
            info = action_info[user]
            slate = info["slate"]
            phi = info["phi"]
            mu = info["mu"]

            if len(slate) == 0:
                continue

            z_before = float(G.nodes[user]["z"])

            #z_cont = np.array([it[0] for it in slate], dtype=float)
            #q_cont = np.array([it[1] for it in slate], dtype=float)
            z_cont = np.array([item_z(it) for it in slate], dtype=float)
            q_cont = np.array([item_q(it) for it in slate], dtype=float)

            # True environment transition
            new_z = update_opinion(
                G,
                user,
                t,
                z_cont,
                q_cont,
                gamma=gamma,
            )

            G.nodes[user]["z"] = new_z

            # Observed nonstationary surrogate reward
            #reward_obs = float(mu + rng.normal(0.0, sigma_reward))
            reward_obs = float(abs(z_before) - abs(new_z)) #- lambda_cost * normalized_slate_cost(slate) + rng.normal(0.0, sigma_reward)

            depol_gain = float(abs(z_before) - abs(new_z))
            cost_i = slate_cost(slate)
            round_cost += cost_i

            alg_round_observed += reward_obs
            alg_round_true_depol += depol_gain

            A += np.outer(phi, phi)
            b += reward_obs * phi

        alg_observed_rewards_list.append(alg_round_observed)

        cumulative_cost += round_cost
        round_costs.append(round_cost)
        cum_costs.append(cumulative_cost)

        theta_hat_end = np.linalg.pinv(A) @ b

        theta_err = float(np.linalg.norm(theta_hat_end - theta_star_t))
        theta_cos = float(
            np.dot(theta_hat_end, theta_star_t)
            / (
                np.linalg.norm(theta_hat_end) * np.linalg.norm(theta_star_t)
                + 1e-12
            )
        )

        theta_hat_err_list.append(theta_err)
        theta_hat_cos_list.append(theta_cos)
        theta_hat_dep_list.append(float(theta_hat_end[3]))

        # --------------------------------------------------------
        # Network dynamics
        # --------------------------------------------------------
        new_pol = pol_L1(G)
        pol_values.append(new_pol)

        if drift:
            update_prejudices(G, tau=0.0005)

        history.append({n: G.nodes[n]["z"] for n in G.nodes})

        if t % 200 == 0:
            print(
                f"Round {t:5d} | "
                f"Epoch round: {local_round:4d} | "
                f"Pol: {new_pol:.4f} | "
                f"Dyn regret: {cumulative_dynamic_regret:.3f} | "
                f"Avg regret: {avg_regret_list[-1]:.5f} | "
                f"Theta err: {theta_err:.4f}"
            )

    path_length = float(
        np.sum(np.linalg.norm(theta_path[1:] - theta_path[:-1], axis=1))
    )
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS: Restarted Top-k LinUCB")
    print(f"{'='*80}")
    print(f"Path length P_T: {path_length:.4f}")
    print(f"Depolarization: {pol_values[0]:.4f} -> {pol_values[-1]:.4f}")
    print(f"Cumulative dynamic surrogate regret: {cumulative_dynamic_regret:.4f}")
    print(f"Average dynamic regret: {cumulative_dynamic_regret / T:.6f}")
    print(f"R(T)/sqrt(T): {cumulative_dynamic_regret / np.sqrt(T):.4f}")
    print(f"Final theta error: {theta_hat_err_list[-1]:.4f}")
    print(f"{'='*80}\n")

    return {
        "history": history,
        "pol_values": pol_values,

        # dynamic regret
        "regret_list": regret_list,
        "inst_regret_list": inst_regret_list,
        "avg_regret_list": avg_regret_list,
        "scaled_regret_list": scaled_regret_list,

        # rewards
        "oracle_rewards": oracle_rewards_list,
        "alg_expected_rewards": alg_expected_rewards_list,
        "alg_observed_rewards": alg_observed_rewards_list,
        "alg_true_rewards": alg_true_rewards_list,

        # theta
        "theta_path": theta_path,
        "theta_hat_err_list": theta_hat_err_list,
        "theta_hat_cos_list": theta_hat_cos_list,
        "theta_hat_dep_list": theta_hat_dep_list,

        # costs
        "round_costs": round_costs,
        "cum_costs": cum_costs,

        # nonstationarity
        "path_length": path_length,
        "restart_period": restart_period,
        "restart_points": restart_points,

    }