"""
In this file we are going to present the implementation of vanilla linUCB.
The algorithm is used to choose the top-k users to recommend content to.
"""

import random
import numpy as np
from network import pol_L1, FJ_update, update_prejudices
from content_influence import attention, update_opinion, gen_posts, greedy_slate, user_pool, generic_posts

def linucb_s(G, k, k_f, k_g, v_vec, e_vec, M_g, M_f,T=5000, k_users=5, K=1, gamma=0.5, c=1.0, d=None,window_size=None, lambda_reg=1.0, drift=False,delta=0.7, seed=2,):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    N = G.number_of_nodes()
    all_users = list(G.nodes)

    if window_size is None:
        window_size = max(50, int(np.sqrt(T)))

    # ============================================================
    # SURROGATE REWARD MODEL: r_{t,u} = phi_{t,u}^T theta_* + noise
    # ============================================================
    true_theta = np.array([
        1.0,    # bias
        -0.1,   # abs(z_user)
        -0.15,  # mismatch = |mean_z - z_user|
        0.9,    # dep_proxy
        0.1,    # quality
    ], dtype=float)
    true_theta /= np.linalg.norm(true_theta)

    # If d is not provided, infer it from theta
    if d is None:
        d = len(true_theta)

    assert d == len(true_theta), f"d={d} but true_theta has length {len(true_theta)}"

    # ============================================================
    # THEORY CONSTANTS
    # ============================================================
    sigma_reward = 0.03              # std of Gaussian reward noise
    R_noise = sigma_reward           # sub-Gaussian proxy for Gaussian noise
    S = float(np.linalg.norm(true_theta))   # = 1 after normalization

    # Feature clipping bound for quality term
    q_clip = 3 * 0.02   # same scale as your original code

    # Conservative bound on ||phi||_2
    # phi = [1, |z_user|, |mean_z-z_user|, dep_proxy, clipped_log_mean_q]
    # bounds:
    # 1 <= 1
    # |z_user| <= 1
    # |mean_z-z_user| <= 2
    # dep_proxy = |z|-|z_next_pred|  in [-1,1] so |dep_proxy| <= 1
    # q_feat <= q_clip
    L = float(np.sqrt(1.0**2 + 1.0**2 + 2.0**2 + 1.0**2 + q_clip**2))

    # A universal plotting constant for the theory curve
    C_theory = 1.0

    print(f"\n{'='*80}")
    print("LinUCB")
    print(f"{'='*80}")
    print(f"Feature dimension d: {d}")
    print(f"True reward model: r_t = phi_t^T theta_* + noise")
    print(f"Noise std: {sigma_reward}")
    print(f"S = ||theta_*||_2 = {S:.4f}")
    print(f"L = sup ||phi||_2 <= {L:.4f}")
    print(f"lambda = {lambda_reg}, delta = {delta}")
    print(f"{'='*80}\n")

    # ============================================================
    # BANDIT STATE
    # ============================================================
    A = np.eye(d) * lambda_reg
    b = np.zeros(d, dtype=float)

    # ============================================================
    # TRACKING
    # ============================================================
    history = [{n: G.nodes[n]['z'] for n in G.nodes}]
    pol_values = []

    regret_list = []
    inst_regret_list = []
    avg_regret_list = []
    scaled_regret_list = []      # R(T) / sqrt(T)

    oracle_rewards_list = []     # oracle expected surrogate reward
    alg_expected_rewards_list = []   # selected-users expected surrogate reward
    alg_observed_rewards_list = []   # selected-users observed surrogate reward
    alg_true_rewards_list = []       # selected-users true depolarization reward
    surr_round_reward_list = []
    real_round_reward_list = []
    surr_cum_reward_list = []
    real_cum_reward_list = []

    theory_beta_list = []
    theory_bound_list = []

    theta_hat_err_list = []
    theta_hat_cos_list = []
    theta_hat_dep_list = []

    cumulative_regret = 0.0
    cumulative_surr_reward = 0.0
    cumulative_real_reward = 0.0

    z_gen, q_gen = generic_posts(M_g, T)

    # ============================================================
    # FEATURE MAP
    # ============================================================
    def compute_phi(user, z_user, slate, prev_z_user=None):
        if prev_z_user is None:
            prev_z_user = z_user

        if slate is None or len(slate) == 0:
            return np.zeros(d, dtype=float), z_user, 0.0

        z_cont = np.array([it[0] for it in slate], dtype=float)
        q_cont = np.array([it[1] for it in slate], dtype=float)

        mean_z = float(np.mean(z_cont))
        mean_q = float(np.mean(q_cont))

        # attention returns (weights, z_bar)
        _, z_bar = attention(z_user, z_cont, q_cont, beta=1.0, c=0.5)
        z_next_pred = (1 - gamma) * z_user + gamma * z_bar

        dep_proxy = abs(z_user) - abs(z_next_pred)

        q_feat = float(np.clip(np.log(max(mean_q, 1e-12)), -q_clip, q_clip))

        phi = np.array([
            1.0,
            abs(z_user),
            abs(mean_z - z_user),
            dep_proxy,
            q_feat,
        ], dtype=float)

        return phi, z_next_pred, dep_proxy

    # ============================================================
    # THEORY HELPERS
    # ============================================================
    def beta_t_fn(t):
        m = k_users * t
        return (
            R_noise * np.sqrt(
                d * np.log(1.0 + (m * L**2) / (lambda_reg * d))
                + 2.0 * np.log(1.0 / delta)
            )
            + np.sqrt(lambda_reg) * S
        )

    def regret_bound_fn(t):
        m = k_users * t
        bt = beta_t_fn(t)
        logdet_bound = d * np.log(1.0 + (m * L**2) / (lambda_reg * d))
        return 0.01* 2 * bt * np.sqrt(2.0 * k_users * t * logdet_bound)

    # ============================================================
    # MAIN LOOP
    # ============================================================
    for t in range(T):
        t1 = t + 1  # 1-indexed time for theorem formulas
        beta_t = beta_t_fn(t1)

        z_now = np.array([G.nodes[n]['z'] for n in all_users], dtype=float)
        z_post_t, q_post_t = gen_posts(v_vec, e_vec, t, z_now)

        # --------------------------------
        # Build contexts and true expected rewards mu = phi^T theta_*
        # --------------------------------
        action_info = {}

        for user in all_users:
            z_user = float(G.nodes[user]['z'])
            prev_z_user = float(history[-1].get(user, z_user))

            pool = user_pool(G, M_f, M_g, t, user, z_post_t, q_post_t, z_gen, q_gen)

            if len(pool) == 0:
                phi, z_next_pred, dep_proxy = compute_phi(user, z_user, [], prev_z_user=prev_z_user)
                mu = float(phi @ true_theta)
                action_info[user] = {
                    "pool": pool,
                    "slate": [],
                    "phi": phi,
                    "mu": mu,                  # true expected surrogate reward
                    "dep_proxy": dep_proxy,
                }
                continue

            slate = greedy_slate(pool, z_user, k, k_f, k_g, alpha=0.4, epsilon=0.3)
            phi, z_next_pred, dep_proxy = compute_phi(user, z_user, slate, prev_z_user=prev_z_user)
            mu = float(phi @ true_theta)

            action_info[user] = {
                "pool": pool,
                "slate": slate,
                "phi": phi,
                "mu": mu,                  # true expected surrogate reward
                "dep_proxy": dep_proxy,
            }

        # --------------------------------
        # LinUCB score: exploit + bonus
        # --------------------------------
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

        idx = np.argsort(scores)[-k_users:]
        selected_users = [all_users[i] for i in idx]

        # --------------------------------
        # Oracle expected reward and algorithm expected reward
        # PSEUDO-REGRET MUST USE EXPECTED SURROGATE REWARDS ONLY
        # --------------------------------
        oracle_user_rewards = [action_info[user]["mu"] for user in all_users]
        oracle_topk = np.sort(oracle_user_rewards)[-k_users:]
        oracle_round_reward = float(np.sum(oracle_topk))
        oracle_rewards_list.append(oracle_round_reward)


        if t%200 == 0:
            mus = np.array([action_info[u]["mu"] for u in all_users], dtype=float)
            oracle_topk_2 = np.sort(mus)[-k_users:]
            selected_mus = np.array([action_info[u]["mu"] for u in selected_users], dtype=float)
            print(
                f"Round {t}: "
                f"mu_range=({mus.min():.4f},{mus.max():.4f}), "
                f"oracle_sum={oracle_topk_2.sum():.4f}, "
                f"alg_sum={selected_mus.sum():.4f}, "
                f"inst_regret={oracle_topk_2.sum() - selected_mus.sum():.6f}"
            )

        alg_round_expected = float(np.sum([action_info[user]["mu"] for user in selected_users]))
        alg_expected_rewards_list.append(alg_round_expected)

        regret = oracle_round_reward - alg_round_expected
        cumulative_regret += regret

        regret_list.append(cumulative_regret)
        inst_regret_list.append(regret)
        avg_regret_list.append(cumulative_regret / t1)
        scaled_regret_list.append(cumulative_regret / np.sqrt(t1))

        # theoretical bound curve
        theory_beta_list.append(beta_t)
        theory_bound_list.append(regret_bound_fn(t1))

        # --------------------------------
        # Apply selected actions
        # Update environment with true dynamics
        # Update learner with OBSERVED SURROGATE reward
        # --------------------------------
        alg_round_observed = 0.0
        alg_round_true = 0.0

        for user in selected_users:
            info = action_info[user]
            slate = info["slate"]
            phi = info["phi"]
            mu = info["mu"]

            if len(slate) == 0:
                continue

            z_before = float(G.nodes[user]['z'])

            z_cont = np.array([it[0] for it in slate], dtype=float)
            q_cont = np.array([it[1] for it in slate], dtype=float)

            # true environment transition
            new_z = update_opinion(G, user, t, z_cont, q_cont, gamma=gamma)
            G.nodes[user]['z'] = new_z

            # observed surrogate reward for learning
            reward_obs = float(mu + rng.normal(0.0, sigma_reward))
            #reward_obs = float(abs(z_before) - abs(new_z))
            alg_round_observed += reward_obs
            alg_observed_rewards_list.append(reward_obs)

            # true reward for evaluation only
            reward_true = float(abs(z_before) - abs(new_z))
            alg_round_true += reward_true
            alg_true_rewards_list.append(reward_true)

            # LinUCB update
            A += np.outer(phi, phi)
            b += reward_obs * phi

        # diagnostics after update
        theta_hat_round_end = np.linalg.pinv(A) @ b
        theta_err = float(np.linalg.norm(theta_hat_round_end - true_theta))
        theta_cos = float(
            np.dot(theta_hat_round_end, true_theta)
            / (np.linalg.norm(theta_hat_round_end) * np.linalg.norm(true_theta) + 1e-12)
        )

        theta_hat_err_list.append(theta_err)
        theta_hat_cos_list.append(theta_cos)
        theta_hat_dep_list.append(float(theta_hat_round_end[3]))  # dep_proxy coordinate

        alg_observed_rewards_list.append(alg_round_observed)
        alg_true_rewards_list.append(alg_round_true)
        surr_round_reward_list.append(alg_round_observed)
        real_round_reward_list.append(alg_round_true)
        cumulative_surr_reward += alg_round_observed
        cumulative_real_reward += alg_round_true
        surr_cum_reward_list.append(cumulative_surr_reward)
        real_cum_reward_list.append(cumulative_real_reward)

        # --------------------------------
        # Network dynamics
        # --------------------------------
        new_pol = pol_L1(G)
        pol_values.append(new_pol)

        if t % K == 0:
            FJ_update(G)

        if drift:
            update_prejudices(G, tau=0.005)

        history.append({n: G.nodes[n]['z'] for n in G.nodes})

        if t % 200 == 0 and t > 0:
            print(
                f"Round {t:5d} | Pol: {new_pol:.4f} | "
                f"Cum Regret: {cumulative_regret:8.2f} | "
                f"Avg Regret: {avg_regret_list[-1]:.4f} | "
                f"R/sqrt(T): {scaled_regret_list[-1]:.3f} | "
                f"Theta err: {theta_hat_err_list[-1]:.4f}"
            )

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Depolarization: {pol_values[0]:.4f} -> {pol_values[-1]:.4f}")
    print(f"Cumulative pseudo-regret: {cumulative_regret:.2f}")
    print(f"Avg regret/round: {cumulative_regret / T:.4f}")
    print(f"R(T) / sqrt(T): {cumulative_regret / np.sqrt(T):.3f}")
    print(f"Final theta error: {theta_hat_err_list[-1]:.4f}")
    print(f"{'='*80}\n")

    return {
        "history": history,
        "pol_values": pol_values,

        # regret
        "regret_list": regret_list,                   # cumulative pseudo-regret
        "inst_regret_list": inst_regret_list,         # per-round pseudo-regret
        "avg_regret_list": avg_regret_list,           # R(T)/T
        "scaled_regret_list": scaled_regret_list,     # R(T)/sqrt(T)

        # rewards
        "oracle_rewards": oracle_rewards_list,            # oracle expected surrogate reward
        "alg_expected_rewards": alg_expected_rewards_list, # selected users expected surrogate reward
        "alg_observed_rewards": alg_observed_rewards_list, # noisy surrogate reward
        "alg_true_rewards": alg_true_rewards_list,         # true depolarization reward
        "surr_round_reward_list": surr_round_reward_list,
        "real_round_reward_list": real_round_reward_list,
        "surr_cum_reward_list": surr_cum_reward_list,
        "real_cum_reward_list": real_cum_reward_list,

        # theta tracking
        "theta_hat_err_list": theta_hat_err_list,
        "theta_hat_cos_list": theta_hat_cos_list,
        "theta_hat_dep_list": theta_hat_dep_list,

        # theory curve
        "theory_beta_list": theory_beta_list,
        "theory_bound_list": theory_bound_list,
    }
