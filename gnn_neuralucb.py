import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import random
from collections import deque
from network import create_nodes, create_edges, create_edges_for_hubs, pol_L1, FJ_update, update_prejudices
from content_influence import (attention, one_step_user_reward, update_opinion,slate_cost, precompute_generic_cache_mask,
                                simple_cost_user_pool, greedy_cost_slate, gen_posts, greedy_slate,
                                  user_pool, generic_posts, cost_user_pool, slate_cost, cost_friendly_slate, item_cost,
                                  item_q, item_z, item_source, greedy_cost_constrained_slate, hard_slate, cost_pool)

def build_node_features(G, opinion_key="z", prejudice_key="prejudice"):
    """
    Build node-feature matrix X_t for the current graph state.
    """

    N = len(G.nodes)
    X = []

    for i in range(N):
        # Current opinion z_i(t)
        z_i = float(G.nodes[i].get(opinion_key, 0.0))

        # Prejudice s_i. If missing, use current opinion.
        #s_i = float(G.nodes[N[i]].get(prejudice_key, z_i))

        # Community label, if available
        community_i = float(G.nodes[i].get("community", 0.0))

        # Degrees
        try:
            in_deg = float(G.in_degree(i))
            out_deg = float(G.out_degree(i))
        except:
            in_deg = float(G.degree(i))
            out_deg = float(G.degree(i))

        # Neighbors
        if G.is_directed():
            neigh = list(G.predecessors(i)) + list(G.successors(i))
        else:
            neigh = list(G.neighbors(i))

        neigh = list(set(neigh))

        if len(neigh) > 0:
            neigh_opinions = np.array(
                [float(G.nodes[j].get(opinion_key, 0.0)) for j in neigh],
                dtype=np.float32
            )

            mean_neigh_z = float(np.mean(neigh_opinions))
            local_disagreement = float(abs(z_i - mean_neigh_z))
            local_abs_polarization = float(np.mean(np.abs(neigh_opinions)))
        else:
            mean_neigh_z = 0.0
            local_disagreement = 0.0
            local_abs_polarization = 0.0

        if abs(z_i) > 1e-8:
            regime_pressure = np.sign(z_i) * (mean_neigh_z - z_i)
        else:
            regime_pressure = 0.0

        x_i = [
            z_i, # 0
            abs(z_i), # 1
            #s_i,
            #abs(s_i),
           # community_i, # 2
           # in_deg, # 3
            out_deg, # 4
           # regime_pressure, # 5
           # mean_neigh_z, # 5
           # local_disagreement, # 6
            local_abs_polarization, # 7
            #z_i * mean_neigh_z, # 8
           # abs(z_i) - local_abs_polarization, # 9
        ]

        X.append(x_i)

    X = np.asarray(X, dtype=np.float32)

    # Normalize degree columns
    if X.shape[0] > 0:
        #max_in = max(1.0, float(np.max(X[:, 3])))
        max_out = max(1.0, float(np.max(X[:, 2])))

        #X[:, 3] = X[:, 3] / max_in
        X[:, 2] = X[:, 2] / max_out

    return X

def build_node_features_with_slate(G,t,M_f,M_g,z_post_t,q_post_t,z_gen,q_gen,cache_mask,k,k_f,k_g,gamma,cost_budget,d_target,rng,cost_ratio=None,):
    """
    Build node features enriched with slate-level information.

    For each user i:
        1. build their candidate pool
        2. construct their slate
        3. compute slate statistics
        4. append those statistics to the normal node features
    """

    X_base = build_node_features(G)
    N = len(G.nodes)

    slate_features = []
    slates_by_user = {}

    for user in range(N):
        z_i = float(G.nodes[user]["z"])

        pool = cost_pool(G,user,t,M_f,M_g,z_post_t,q_post_t,z_gen,q_gen,cache_mask,d_target=d_target,rng=rng,use_predecessors=True, )

        effective_cost_budget = None if cost_ratio is not None else cost_budget
        slate = greedy_cost_constrained_slate(pool,z_i,k,k_f,k_g,gamma=gamma,cost_ratio=cost_ratio)
        #slate = greedy_soft_cost_slate(pool, z_i, k, k_f, k_g, gamma=gamma, alpha=0.4, epsilon=0.3, eta=cost_ratio)

        slates_by_user[user] = slate

        if len(slate) == 0:
            mean_slate_z = 0.0
            mean_abs_slate_z = 0.0
            mean_slate_q = 0.0
            slate_c = 0.0
            sl_disaggree = 0.0
            slate_size = 0.0
            depol_proxy = 0.0
            expected_next_abs_z = abs(z_i)
            friend_frac = 0.0
            generic_frac = 0.0

        else:
            z_items = np.array([item_z(item) for item in slate], dtype=float)
            q_items = np.array([item_q(item) for item in slate], dtype=float)

            mean_slate_z = float(np.mean(z_items))
            mean_abs_slate_z = float(np.mean(np.abs(z_items)))
            mean_slate_q = float(np.mean(q_items))
            slate_c = float(slate_cost(slate))
            slate_size = float(len(slate)) / float(k)

            # Simple approximate next opinion.
            # This is not exactly update_opinion if attention is nonlinear,
            # but it is a useful proxy feature.
            z_next_proxy = (1.0 - gamma) * z_i + gamma * mean_slate_z
            expected_next_abs_z = float(abs(z_next_proxy))

            # Positive means the slate is expected to depolarize user i.
            depol_proxy = float(abs(z_i) - abs(z_next_proxy))

            sl_disaggree = float(abs(z_i - mean_slate_z))

            sources = [item_source(item) for item in slate]
            friend_frac = float(np.mean([s == "friend" for s in sources]))
            generic_frac = float(np.mean([s == "generic" for s in sources]))

        slate_features.append([sl_disaggree,depol_proxy])

    X_slate = np.asarray(slate_features, dtype=np.float32)

    # Normalize cost column approximately by slate size.
    # If costs are 0/1 per item, max cost is roughly k.
    #X_slate[:, 3] = X_slate[:, 3] / max(1.0, float(k))

    X = np.concatenate([X_base, X_slate], axis=1)

    return X, slates_by_user

def build_normalized_adjacency(G, add_self_loops=True, symmetrize=True, device="cpu"):
    """
    Build normalized adjacency matrix for a simple GCN.
    """

    N = G.number_of_nodes()
    A = np.zeros((N, N), dtype=np.float32)

    for u, v in G.edges():
        A[int(u), int(v)] = 1.0

    # For first implementation, I recommend symmetrizing.
    # This lets the GNN aggregate from both incoming and outgoing neighbors.
    if symmetrize:
        A = np.maximum(A, A.T)

    if add_self_loops:
        A = A + np.eye(N, dtype=np.float32)

    degree = np.sum(A, axis=1)

    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

    D_inv_sqrt = np.diag(degree_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    A_norm = torch.tensor(A_norm, dtype=torch.float32, device=device)

    return A_norm

class GNNRewardModel(nn.Module):
    """
    GNN reward model.

    Input:
        A_norm: normalized adjacency, shape (N, N)
        X: node features, shape (N, input_dim)

    Output:
        predicted reward for every user, shape (N,)
    """

    def __init__(self, input_dim, hidden_dim=32, embedding_dim=16, dropout=0.0):
        super().__init__()

        self.gcn1 = nn.Linear(input_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, embedding_dim)
        #self.gcn3 = nn.Linear(hidden_dim, embedding_dim)

        self.reward_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = dropout

    def forward(self, A_norm, X):
        H = A_norm @ X
        H = self.gcn1(H)
        H = F.relu(H)
        H = F.dropout(H, p=self.dropout, training=self.training)

        H = A_norm @ H
        H = self.gcn2(H)
        H = F.relu(H)
        H = F.dropout(H, p=self.dropout, training=self.training)

        rewards_hat = self.reward_head(H).squeeze(-1)

        return rewards_hat
    
class GNNNeuralUCB:
    """
    GNN version of NeuralUCB for top-k user selection.

    This replaces ensemble uncertainty with the NeuralUCB uncertainty term.

    For each user i:
        mu_i = f_theta(A, X)_i
        g_i  = gradient of mu_i with respect to the network parameters
        sigma_i = sqrt(g_i^T Z^{-1} g_i)
        score_i = mu_i + alpha * sigma_i

    For speed, this implementation uses a diagonal approximation of Z:
        Z_diag = lambda_reg + sum observed g_i^2

    This is much lighter than storing and inverting the full parameter matrix.
    """

    def __init__(self,input_dim,hidden_dim=32,embedding_dim=16,alpha=1.0,lr=1e-3,buffer_size=1000,batch_size=32,train_epochs=1,dropout=0.0,device="cpu",lambda_reg=1.0,grad_scale="sqrt_p",):
        self.device = device
        self.alpha = alpha
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.lambda_reg = float(lambda_reg)
        self.grad_scale = grad_scale

        self.model = GNNRewardModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.num_params = int(sum(p.numel() for p in self.params))

        # Diagonal approximation of the NeuralUCB design matrix Z.
        # Exact NeuralUCB would keep a full Z matrix, but that is expensive for a GNN.
        self.Z_diag = torch.ones(self.num_params, dtype=torch.float32, device=device) * self.lambda_reg

    def _to_tensor_inputs(self, A_norm, X):
        A_norm = A_norm.to(self.device)

        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        return A_norm, X

    def _flatten_grad(self, scalar_output):
        """
        Return grad_theta scalar_output as a single vector.
        """
        grads = torch.autograd.grad(
            scalar_output,
            self.params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        flat = []
        for p, g in zip(self.params, grads):
            if g is None:
                flat.append(torch.zeros_like(p).reshape(-1))
            else:
                flat.append(g.detach().reshape(-1))

        g_vec = torch.cat(flat)

        # NeuralUCB uses a width normalization. For this GNN, sqrt(num_params)
        # is a stable practical normalization.
        if self.grad_scale == "sqrt_p":
            g_vec = g_vec / np.sqrt(max(1, self.num_params))
        elif isinstance(self.grad_scale, (int, float)) and self.grad_scale > 0:
            g_vec = g_vec / float(self.grad_scale)

        return g_vec

    def _grad_for_user(self, pred_all, user):
        return self._flatten_grad(pred_all[int(user)])

    def predict_all(self, A_norm, X):
        A_norm, X = self._to_tensor_inputs(A_norm, X)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(A_norm, X)
        return pred

    def neuralucb_scores(self, A_norm, X):
        """
        Compute NeuralUCB scores for every user.

        Returns:
            scores, mu, sigma as torch tensors of shape (N,)
        """
        A_norm, X = self._to_tensor_inputs(A_norm, X)
        self.model.eval()

        pred_all = self.model(A_norm, X)
        N = pred_all.shape[0]

        sigmas = []
        for user in range(N):
            g_vec = self._grad_for_user(pred_all, user)
            # Diagonal NeuralUCB uncertainty: sqrt(g^T Z^{-1} g)
            sigma_user = torch.sqrt(torch.sum((g_vec * g_vec) / self.Z_diag).clamp_min(1e-12))
            sigmas.append(sigma_user)

        sigma = torch.stack(sigmas)
        mu = pred_all.detach()
        scores = mu + self.alpha * sigma.detach()

        return scores, mu, sigma

    def select_users(self, A_norm, X, k_users, t, epsilon=0.0):
        """
        Select top-k users using NeuralUCB.

        score_i = f_theta(A, X)_i + alpha * sqrt(g_i^T Z^{-1} g_i)
        """
        N = X.shape[0]

        scores, mu, sigma = self.neuralucb_scores(A_norm, X)
        epsilon_ = epsilon * (0.995 ** t)

        if np.random.rand() < epsilon_:
            selected_users = np.random.choice(N, size=k_users, replace=False).tolist()
        else:
            selected_users = torch.topk(scores, k=k_users).indices.detach().cpu().numpy().tolist()

        return (
            selected_users,
            scores.detach().cpu().numpy(),
            mu.detach().cpu().numpy(),
            sigma.detach().cpu().numpy(),
        )

    def observe(self, A_norm, X, selected_users, rewards, t):
        """
        Store observed rewards and update the NeuralUCB design matrix.

        Z_diag <- Z_diag + g_i^2 for each selected user.
        """
        A_cpu = A_norm.detach().cpu()

        if not torch.is_tensor(X):
            X_cpu = torch.tensor(X, dtype=torch.float32).detach().cpu()
        else:
            X_cpu = X.detach().cpu()

        # Update confidence matrix using gradients at the observed contexts.
        A_t, X_t = self._to_tensor_inputs(A_norm, X)
        self.model.eval()
        pred_all = self.model(A_t, X_t)

        for user, reward in zip(selected_users, rewards):
            g_vec = self._grad_for_user(pred_all, user)
            self.Z_diag += g_vec * g_vec

            self.buffer.append(
                {
                    "A_norm": A_cpu,
                    "X": X_cpu,
                    "user": int(user),
                    "reward": float(reward),
                    "time": int(t),
                }
            )

    def train(self):
        """
        Train the single GNN reward model on samples from the sliding buffer.
        """
        if len(self.buffer) < self.batch_size:
            return None

        buffer_list = list(self.buffer)
        self.model.train()
        model_losses = []

        for _ in range(self.train_epochs):
            batch = random.choices(buffer_list, k=self.batch_size)
            total_loss = 0.0

            for sample in batch:
                A_norm = sample["A_norm"].to(self.device)
                X = sample["X"].to(self.device)
                user = sample["user"]

                reward = torch.tensor(
                    sample["reward"],
                    dtype=torch.float32,
                    device=self.device,
                )

                pred_all = self.model(A_norm, X)
                pred_user = pred_all[user]

                loss = F.smooth_l1_loss(pred_user, reward)
                total_loss = total_loss + loss

            total_loss = total_loss / self.batch_size

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            model_losses.append(float(total_loss.detach().cpu()))

        return float(np.mean(model_losses))
    
def gnn_neuralucb_topk2(G,k,k_f,k_g,M_f,M_g,e_vec,v_vec,T=2000,k_users=5,drift=False,gamma=0.5,K=1,d_target=10,seed=2,alpha=0.5,lr=5e-4,buffer_size=3000,batch_size=64,train_epochs=5,device="cpu",
    cache_fraction=0.5,hidden_dim=64,embedding_dim=32,eta=0.4,n_models=5,warmup=300,cost_budget=2.0,epsilon=0.05,train_every=10,reward_scale=10.0,reward_clip=1.0,cost_ratio=None):
    """
    GNN NeuralUCB with slate-level node features.

    Main difference from the previous version:
        X_t is no longer only graph/user features.
        For every user, before selection, we build the candidate slate and append
        slate summary features to the user's node feature vector.

    This makes the GNN predict reward from:
        graph state + user state + available slate quality/cost/depolarization proxy.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rng = np.random.default_rng(seed)
    G = copy.deepcopy(G)

    T_eff = min(T, len(e_vec), len(v_vec))

    all_users = list(G.nodes)
    N = len(all_users)

    A_norm = build_normalized_adjacency(G, device=device)

    # Generate generic content and cache mask once, before building initial X.
    z_gen, q_gen = generic_posts(M_g, T_eff)

    cache_mask = precompute_generic_cache_mask(
        T=T_eff,
        M_g=M_g,
        cache_fraction=cache_fraction,
        rng=rng,
    )

    # --------------------------------------------------
    # Build initial slate-enriched features to get input_dim
    # --------------------------------------------------
    z_now0 = np.array(
        [float(G.nodes[n]["z"]) for n in all_users],
        dtype=float,
    )

    z_post_0, q_post_0 = gen_posts(v_vec, e_vec, 0, z_now0)

    X0,slates_by_user_0 = build_node_features_with_slate( G=G,t=0,M_f=M_f,M_g=M_g,z_post_t=z_post_0,q_post_t=q_post_0,z_gen=z_gen,q_gen=q_gen,cache_mask=cache_mask,k=k,k_f=k_f,k_g=k_g,gamma=gamma,cost_budget=cost_budget,d_target=d_target,rng=rng,cost_ratio=cost_ratio,)
    #X0 = build_node_features(G)
    input_dim = X0.shape[1]

    agent = GNNNeuralUCB(input_dim=input_dim,hidden_dim=hidden_dim,embedding_dim=embedding_dim,alpha=alpha,
        lr=lr,buffer_size=buffer_size,batch_size=batch_size,train_epochs=train_epochs,dropout=0.1,device=device,
        lambda_reg=1.0,grad_scale="sqrt_p",)

    history = [{n: G.nodes[n]["z"] for n in G.nodes}]
    pol_values = []
    reward_history = []
    selected_users_history = []

    cumulative_true_depol = 0.0
    cumulative_reward = 0.0
    cumulative_cost = 0.0
    cum_costs = []
    reward_scale = 10.0
    reward_clip = 1.0

    train_losses = []
    pol_prev = pol_L1(G)
    for t in range(T_eff):

        # --------------------------------------------------
        # 1. Current graph/content state
        # --------------------------------------------------
        if K is not None and K > 0 and (t + 1) % K == 0:
            FJ_update(G)

        z_now = np.array(
            [float(G.nodes[n]["z"]) for n in all_users],
            dtype=float,
        )

        z_post_t, q_post_t = gen_posts(v_vec, e_vec, t, z_now)

        # Slate-level features are computed before user selection.
        X,slates_by_user = build_node_features_with_slate(G=G,t=t, M_f=M_f,M_g=M_g,z_post_t=z_post_t,q_post_t=q_post_t,z_gen=z_gen,q_gen=q_gen,cache_mask=cache_mask,k=k,k_f=k_f,k_g=k_g,gamma=gamma,cost_budget=cost_budget,d_target=d_target,rng=rng,cost_ratio=cost_ratio,)
        #X = build_node_features(G)
   
        if t < warmup:
            selected_users = rng.choice(
                N,
                size=k_users,
                replace=False,
            ).tolist()

            scores = np.zeros(N)
            mu = np.zeros(N)
            sigma = np.zeros(N)

        else:
            selected_users, scores, mu, sigma = agent.select_users(
                A_norm=A_norm,
                X=X,
                k_users=k_users,
                t=t,
                epsilon=epsilon,
            )

        rewards_local = []
        true_depols = []
        costs = []

        for user in selected_users:

            z_before = float(G.nodes[user]["z"])

            #pool = cost_pool(G, user, t,M_f, M_g,z_post_t, q_post_t,z_gen, q_gen,cache_mask,d_target=d_target,rng=rng,use_predecessors=True,)

            #slate = greedy_cost_constrained_slate(pool,z_before,k,k_f,k_g,gamma=gamma,alpha=0.4,epsilon=0.3,cost_ratio=cost_ratio,)
            #slate = random_cost_constrained_slate(pool, z_before, k, k_f, k_g, alpha=0.4, cost_ratio=cost_ratio)
            slate = slates_by_user[user]
            if len(slate) == 0:
                depol_reward = 0.0
                cost = 0.0
                local_reward = 0.0

            else:
                z_cont = [item_z(item) for item in slate]
                q_cont = [item_q(item) for item in slate]

                z_after = update_opinion(G,user,t,z_cont,q_cont,gamma=gamma,)

                G.nodes[user]["z"] = z_after

                depol_reward = float(abs(z_before) - abs(z_after))
                cost = float(slate_cost(slate))

                local_reward = depol_reward

            rewards_local.append(float(local_reward))
            true_depols.append(float(depol_reward))
            costs.append(float(cost))

            cumulative_true_depol += float(depol_reward)
            cumulative_cost += float(cost)
            cum_costs.append(float(cumulative_cost))

        # --------------------------------------------------
        # 4. Store observations and train GNN ensemble
        # --------------------------------------------------
        
        if drift :
            update_prejudices(G, tau=0.001)

        new_pol = pol_L1(G)

        delta_pol = float(pol_prev - new_pol)  # positive if polarization decreased this step
        pol_prev = new_pol

        shaped_rewards = []

        for r in rewards_local:
            reward = reward_scale* (r+ 0.0*delta_pol)

            if reward_clip is not None:
                reward = float(np.clip(reward, -reward_clip, reward_clip))

            shaped_rewards.append(float(reward))

        round_shaped_sum = float(np.sum(shaped_rewards)) if len(shaped_rewards) > 0 else 0.0
        cumulative_reward += round_shaped_sum

        pol_values.append(float(new_pol))
        reward_history.append(round_shaped_sum)
        selected_users_history.append(selected_users)
        history.append({n: G.nodes[n]["z"] for n in G.nodes})

        if t % 200 == 0 and t > 0:
            last_loss = train_losses[-1] if len(train_losses) > 0 else np.nan
            print(
                f"Round {t:5d} | "
                f"Pol: {new_pol:.4f} | "
                f"Cum true depol: {cumulative_true_depol:.3f} | "
                f"Cum cost: {cumulative_cost:.3f} | "
                f"Cum reward: {cumulative_reward:.3f} | "
                f"Last train loss: {last_loss:.5f}"
            )
        
        agent.observe(
            A_norm=A_norm,
            X=X,
            selected_users=selected_users,
            rewards=shaped_rewards,
            t=t,
        )

        if train_every > 0 and t % train_every == 0:
            losses = agent.train()

            if losses is not None:
                train_losses.append(float(np.mean(losses)))

    return {
        "G": G,
        "history": np.array(history),
        "pol_values": np.array(pol_values),
        "reward": np.array(reward_history),
        "selected_history": selected_users_history,
        "agent": agent,
        "cumulative_true_depol": cumulative_true_depol,
        "cumulative_reward": cumulative_reward,
        "cumulative_cost": cumulative_cost,
        "cum_costs": np.array(cum_costs),
        "avg_cost_per_selected_user": cumulative_cost / max(1, T_eff * k_users),
        "train_losses": np.array(train_losses),
        "train_every": train_every,
    }
