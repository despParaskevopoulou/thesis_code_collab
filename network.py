import networkx as nx
import numpy as np

"""
In this file there is what we need to construct a network, and the community of the network,
including the creation of nodes and edges, and the FJ update function, and the polarization metric.
There is also a function where the FJ model is evolving for horizon T.
"""

def create_nodes(N,s_minus=-0.65,s_plus=0.65,case='stable',shift=0.1,std=0.1,p_mod = 0.00):
    G = nx.DiGraph()

    for i in range(N):
        community = 0 if (np.random.rand() < 0.5) else 1
        G.add_node(i, community=community)
        community = G.nodes[i]['community']

        is_moderate = (np.random.rand() < p_mod)
        if is_moderate:
            in_opinion = np.clip(np.random.normal(0.0, std), -1, 1) 
            opinion = in_opinion
        else:
            if community == 0:
                in_opinion = np.clip(np.random.normal(s_minus, std), -1, 1)
                opinion = np.clip(in_opinion + shift, -1, 1)
            else:
                in_opinion = np.clip(np.random.normal(s_plus, std), -1, 1)
                opinion = np.clip(in_opinion + shift, -1, 1)

        if case == 'stable':
            lambda_ = float(np.clip(np.random.normal(0.1, 0.02), 0.0, 1.0))
        elif case=='depol':
            lambda_ = float(np.clip(np.random.normal(0.75, 0.02), 0.0, 1.0))
        else:
            #lambda_ = float(np.clip(np.random.normal(0.88, 0.02), 0.0, 1.0))
            lambda_ = float(np.clip(np.random.normal(0.88, 0.02), 0.0, 1.0))
        

        G.nodes[i]['is_moderate'] = bool(is_moderate)
        G.nodes[i]['opinion'] = float(opinion)
        G.nodes[i]['s'] = float(in_opinion) # prejudice
        G.nodes[i]['z'] = float(opinion)
        G.nodes[i]['lambda'] = float(lambda_)

    return G

def create_edges(G,N, kappa=6,dtarget = 14, dmin=10):
    rng = np.random.default_rng(42)
    p0 = dtarget / (N - 1)
    d_target = dtarget
    s0 = {n: G.nodes[n]['z'] for n in G.nodes}

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            prob = p0 * np.exp(-kappa * abs(s0[i] - s0[j]))
            if rng.random() < prob:
                G.add_edge(j, i)


    for j in range(N):
        outdeg = G.out_degree(j)
        if outdeg >= dmin:
            continue

        succ = set(G.successors(j))
        candidates = [i for i in range(N) if i != j and i not in succ]
        candidates.sort(key=lambda i: abs(s0[i] - s0[j]))

        needed = dmin - outdeg
        for i in candidates[:needed]:
            G.add_edge(j, i)


    A = np.zeros((N, N), dtype=int)
    for (u, v) in G.edges():  # u -> v
        A[v, u] = 1

    W = np.zeros((N, N), dtype=float)

    for i in range(N):
        Ni = list(G.predecessors(i))  # j such that i <- j

        if len(Ni) == 0:
            W[i, i] = 1.0
            continue

        diffs = np.array([abs(s0[i] - s0[j]) for j in Ni], dtype=float)
        w_raw = np.exp(-kappa * diffs)
        w = w_raw / w_raw.sum()

        for j, wij in zip(Ni, w):
            W[i, j] = float(wij)


    for (u, v) in G.edges():
        G[u][v]["weight"] = float(W[v, u])

    return G, s0, A, W, d_target

def create_edges_for_hubs(G,N, kappa=6,dtarget = 14, dmin=10,hub_d=10,s_key="s", comm_key="community"):
    rng = np.random.default_rng(42)
    p0 = dtarget / (N - 1)
    d_target = dtarget
    s0 = {n: G.nodes[n]['z'] for n in G.nodes}

    comm0 = [n for n in G.nodes if G.nodes[n].get(comm_key, 0) == 0]
    comm1 = [n for n in G.nodes if G.nodes[n].get(comm_key, 0) == 1]

    comm0_sorted = sorted(comm0, key=lambda n: G.nodes[n][s_key], reverse=True)
    comm1_sorted = sorted(comm1, key=lambda n: G.nodes[n][s_key], reverse=True)

    k0 = max(1, int(np.ceil(0.05 * len(comm0_sorted))))
    k1 = max(1, int(np.ceil(0.05 * len(comm1_sorted))))

    h0 = comm0_sorted[:k0]
    h1 = comm1_sorted[:k1]
    hubs = h1 + h0

    for h in h0:
        G.nodes[h][s_key] = -np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)
        G.nodes[h]['lambda'] = float(np.clip(np.random.normal(0.88, 0.02), 0.0, 1.0))

    for h in h1:
        G.nodes[h][s_key] = np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)
        G.nodes[h]['lambda'] = float(np.clip(np.random.normal(0.88, 0.02), 0.0, 1.0))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            prob = p0 * np.exp(-kappa * abs(s0[i] - s0[j]))
            if prob > 1.0:
                prob = 1.0
            if rng.random() < prob:
                G.add_edge(j, i)


    for j in range(N):
        outdeg = G.out_degree(j)
        if outdeg >= dmin:
            continue

        succ = set(G.successors(j))
        candidates = [i for i in range(N) if i != j and i not in succ]
        candidates.sort(key=lambda i: abs(s0[i] - s0[j]))

        needed = dmin - outdeg
        for i in candidates[:needed]:
            G.add_edge(j, i)

    
    for h in hubs:
        current_in = G.in_degree(h)
        if current_in >= hub_d:
            continue

        succ = set(G.successors(j))
        candidates = [i for i in range(N) if i != j and i not in succ]
        candidates.sort(key=lambda i: abs(s0[i] - s0[j]))

        needed = hub_d - outdeg
        for i in candidates[:needed]:
            G.add_edge(j, i)


    A = np.zeros((N, N), dtype=int)
    for (u, v) in G.edges():  # u -> v
        A[v, u] = 1

    W = np.zeros((N, N), dtype=float)

    for i in range(N):
        Ni = list(G.predecessors(i))  # j such that i <- j

        if len(Ni) == 0:
            W[i, i] = 1.0
            continue

        diffs = np.array([abs(s0[i] - s0[j]) for j in Ni], dtype=float)
        w_raw = np.exp(-kappa * diffs)

        if i in hubs:
            w_raw = w_raw * 3

        w = w_raw / w_raw.sum()

        for j, wij in zip(Ni, w):
            W[i, j] = float(wij)


    for (u, v) in G.edges():
        G[u][v]["weight"] = float(W[v, u])

    return G, s0, A, W, d_target

def FJ_update(G):
    new_z = {}
    old_z = {n: G.nodes[n]['z'] for n in G.nodes}

    for i in G.nodes:
        p_i = G.nodes[i].get('s', 0.0)
        l_i = G.nodes[i].get('lambda', 0.1)

        term = 0.0
        total_w = 0.0

        for j in G.predecessors(i):
            w_ji = G[j][i].get('weight', 1.0)
            term += w_ji * old_z[j]
            total_w += w_ji

        if total_w > 0:
            social = term / total_w
        else:
            social = old_z[i]  # W_ii = 1 when no neighbors

        new_z[i] = (1.0 - l_i) * p_i + l_i * social

        

    nx.set_node_attributes(G, new_z, "z")
        
    return new_z

def pol_L1(G):
    z_vals = np.array([G.nodes[n]['z'] for n in G.nodes])
    val = np.mean(np.abs(z_vals))
    return val

def update_prejudices(G, tau):
    if tau <= 0.0:
        return

    for i in G.nodes:
        p = G.nodes[i]['s'] # prejudice
        z = G.nodes[i]['z']
        G.nodes[i]['s'] = (1 - tau) * p + tau * z

    for i in G.nodes:
        p = G.nodes[i]['s'] # prejudice
        z = G.nodes[i]['z']
        G.nodes[i]['s'] = (1 - tau) * p + tau * z

def FJ_model_overT(G,T=100,extra_steps=50,stable_required=10,tol=1e-4):
    history = [{n: G.nodes[n]['z'] for n in G.nodes}]
    pol_values = []
    pol = pol_L1(G)
    pol_values.append(pol)
    converged = False
    stable = 0
    prev_pol = None
    count = 0

    for t in range(T):
        new_z = {}
        old_z = {n: G.nodes[n]['z'] for n in G.nodes}

        for i in G.nodes:
            p_i = G.nodes[i].get('s', 0.0)
            l_i = G.nodes[i].get('lambda', 0.1)

            term = 0.0
            total_w = 0.0

            for j in G.predecessors(i):
                w_ji = G[j][i].get('weight', 1.0)
                term += w_ji * old_z[j]
                total_w += w_ji

            if total_w > 0:
                social = term / total_w
            else:
                social = old_z[i]  # W_ii = 1 when no neighbors

            new_z[i] = (1.0 - l_i) * p_i + l_i * social

        nx.set_node_attributes(G, new_z, "z")
        history.append(new_z.copy())

        z_vals = np.array([G.nodes[n]['z'] for n in G.nodes])
        pol = pol_L1(G)
        pol_values.append(pol)
        history.append({n: G.nodes[n]['z'] for n in G.nodes})

    return history,pol_values,z_vals

