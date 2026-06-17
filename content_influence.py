import copy
import numpy as np

"""
This file includes whatever is related withe the content influence model, 
including the generative process of posts, the attention mechanism, and the slate generation algorithm.
"""

def similarity(s,z):
    return 1.0 - abs(s-z)/2.0

def avg_sim(z,z_cont):
    sims = [similarity(z,float(z_content)) for z_content in z_cont]
    avg = float(np.mean(sims))
    return avg

def attention(z_i,z_cont,q_cont,beta=1.0,c=1.0):
    z_cont = np.asarray(z_cont,dtype=float)
    q_cont = np.asarray(q_cont,dtype=float)

    var = beta*similarity(z_i,z_cont) + c*np.log(q_cont)

    var = var - np.max(var)
    a = np.exp(var)
    a = a / np.sum(a)

    z_bar = float(np.sum(a * z_cont))

    return a,z_bar

def update_opinion(G, node, t, z_cont, q_cont, gamma=0.5,K=1):
    s_i = G.nodes[node]['s']
    z_i = float(G.nodes[node]['z'])

    _, z_bar = attention(z_i,z_cont,q_cont,beta=5,c=3)

    new_z = np.clip(((1-gamma)*z_i+gamma*z_bar),-1.0,1.0)
    
    return new_z

def gen_streams(T,N,sigma_post=0.02,sigma_q=0.02):
    rng = np.random.default_rng(42)
    e_vec = rng.normal(loc=0.0, scale=sigma_post, size=(T, N))  # e_i(t)
    v_vec  = rng.normal(loc=0.0, scale=sigma_q,    size=(T, N))  # ν_i(t)
    return e_vec,v_vec

def gen_posts(v_vec,e_vec,t,z):
    z_post = np.clip(z+e_vec[t],-1.0,1)
    q_post = np.exp(v_vec[t])
    return z_post, q_post

def generic_posts(M_g,T,mu_minus=-0.65,mu_plus=0.65,std=0.02,std_q=0.05):
    rng = np.random.default_rng(42)
    M1 = M_g//2
    M2 = M_g-M1

    z_neg = rng.normal(mu_minus, std, size=(T, M1))
    z_pos = rng.normal(mu_plus, std, size=(T, M2))

    z_gen = np.concatenate([z_neg, z_pos], axis=1)
    z_gen = np.clip(z_gen, -1.0, 1.0)

    for t in range(T):
        rng.shuffle(z_gen[t])

    v_ = rng.normal(0.0, std_q, size=(T, M_g))

    q_gen = np.exp(v_)

    return z_gen, q_gen

def user_pool(G,M_f,M_g,t,user,z_post,q_post,z_gen,q_gen):
    rng = np.random.default_rng()
    neighbors = list(G.predecessors(user))
    rng.shuffle(neighbors)

    friends = neighbors[:M_f]
    friend_posts = [(z_post[j], q_post[j], "friend", 0.0) for j in friends]

    gen_posts = [(z_gen[t,k], q_gen[t,k], "generic", 0.0) for k in range(M_g)]
    pool = friend_posts + gen_posts
    return pool

"""
def maxsim_slate(z_user, pool, k_f, k_g):
    if len(pool) == 0:
        return []

    # Support both labeled items (source, id, z, q) and plain (z, q) tuples.
    first_item = pool[0]
    is_labeled = (
        isinstance(first_item, (tuple, list))
        and len(first_item) >= 4
        and first_item[0] in ("friend", "generic")
    )

    if is_labeled:
        friends = [it for it in pool if it[0] == "friend"]
        generics = [it for it in pool if it[0] == "generic"]

        friends.sort(key=lambda it: similarity(z_user, float(it[2])), reverse=True)
        generics.sort(key=lambda it: similarity(z_user, float(it[2])), reverse=True)
        return friends[:k_f] + generics[:k_g]

    # Unlabeled pool: take the top-(k_f + k_g) items by similarity.
    budget = max(0, int(k_f) + int(k_g))
    ranked = sorted(pool, key=lambda it: similarity(z_user, float(it[0])), reverse=True)
    return ranked[:budget]
"""
""""
def maxsim_slate(z_user, pool, k_f, k_g):
    if len(pool) == 0:
        return []

    budget = max(0, int(k_f) + int(k_g))

    ranked = sorted(
        pool,
        key=lambda it: similarity(z_user, item_z(it)),
        reverse=True
    )

    return ranked[:budget]
"""

def maxsim_slate(z_user, pool, k_f=None, k_g=None, num_contents=None):
    if len(pool) == 0:
        return []

    # -----------------------------
    # Free mode: no friend/generic split
    # -----------------------------
    if k_f is None or k_g is None:
        if num_contents is None:
            raise ValueError("num_contents must be provided when k_f or k_g is None.")

        ranked = sorted(
            pool,
            key=lambda it: similarity(z_user, item_z(it)),
            reverse=True
        )

        return ranked[:num_contents]

    # -----------------------------
    # Split mode: force k_f friends and k_g generic
    # -----------------------------
    friend_pool = [it for it in pool if item_source(it) == "friend"]
    generic_pool = [it for it in pool if item_source(it) == "generic"]

    ranked_friends = sorted(
        friend_pool,
        key=lambda it: similarity(z_user, item_z(it)),
        reverse=True
    )

    ranked_generic = sorted(
        generic_pool,
        key=lambda it: similarity(z_user, item_z(it)),
        reverse=True
    )

    return ranked_friends[:k_f] + ranked_generic[:k_g]
    
def one_step_user_reward(G, t, user, M_f, M_g, k, k_f, k_g, e_vec, v_vec, z_gen, q_gen, gamma=0.5):
    G_tmp = copy.deepcopy(G)
    N = G_tmp.number_of_nodes()

    z_now = np.array([G_tmp.nodes[n]['z'] for n in range(N)], dtype=float)
    z_post_t, q_post_t = gen_posts(v_vec, e_vec, t, z_now)

    z_before = float(G_tmp.nodes[user]['z'])

    pool = user_pool(G_tmp, M_f, M_g, t, user, z_post_t, q_post_t, z_gen, q_gen)
    slate = greedy_slate(pool, z_before, k, k_f, k_g, alpha=0.4, epsilon=0.3)

    if len(slate) == 0:
        return 0.0

    z_cont = np.array([it[0] for it in slate], dtype=float)
    q_cont = np.array([it[1] for it in slate], dtype=float)

    if gamma > 0.0:
        new_z = update_opinion(G_tmp, user, t, z_cont, q_cont, gamma=gamma)
    else:
        new_z = z_before

    reward = (abs(z_before) - abs(new_z)) #/ (abs(z_before) + 1e-6)
    return float(reward)

def greedy_helper(slate,z):
    if len(slate) == 0:
        return 0.0
    z_items = [it[0] for it in slate]
    return avg_sim(z, z_items)

def greedy_slate(pool,z,num_contents,k_f,k_g,gamma=0.5,alpha=0.5,epsilon=0.3):
    slate = maxsim_slate(z, pool, k_f, k_g)
    sim_ref = greedy_helper(slate, z)

    sim_floor = alpha * sim_ref

    remaining = pool.copy()
    slate = []
    sims = []

    for step in range(num_contents):
        best_idx = None
        best_score = np.inf

        for idx, item in enumerate(remaining):
            z_item, q_item, source, cost = item

            possible_slate = slate + [(z_item, q_item, source, cost)]
            sim_val = float(similarity(z, z_item))
            if not np.isfinite(sim_val):
                continue
            poss_sims = sims + [sim_val]

            if len(poss_sims) == 0:
                continue
            sim_slate_avg = float(np.mean(poss_sims))

            if sim_slate_avg < sim_floor:
                continue

            z_cont = np.array([it[0] for it in possible_slate], dtype=float)
            q_cont = np.array([it[1] for it in possible_slate], dtype=float)

            _, z_bar = attention(z, z_cont, q_cont, beta=0.8, c=0.5)

            s_next = (1 - gamma) * z + gamma * z_bar

            # objective: depolarize + small quality bonus
            score = abs(s_next) - epsilon * q_item

            #print(f"candidate z={z_item:.3f}, predicted |s_next|={abs(score):.3f}")

            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        best_item = remaining.pop(best_idx)
        slate.append(best_item)

        sims.append(float(similarity(z, best_item[0])))
        

    return slate

"""
At this part we are going to add the slate optimization constrained to the cost. 
First, we need a cost generating function of user_pool.
Each 
"""

def cost_user_pool(G,M_f,M_g,t,user,z_post,q_post,z_gen,q_gen,cached_friend_prob=0.2,cached_generic_prob=0.4,cost_cached=0.0,cost_uncached=1.0):
    rng = np.random.default_rng()
    neighbors = list(G.predecessors(user))
    rng.shuffle(neighbors)

    friends = neighbors[:M_f]
    pool = []

    for j in friends:
        content_id = ("friend", int(j), int(t))

        z = float(z_post[j])
        q = float(q_post[j])
        is_cached = rng.random() < cached_friend_prob
        cost = cost_cached if is_cached else cost_uncached

        pool.append((content_id, z, q, cost))

    # generic posts
    for k in range(M_g):
        content_id = ("generic", int(k), int(t))

        z = float(z_gen[t, k])
        q = float(q_gen[t, k])
        is_cached = rng.random() < cached_generic_prob
        cost = cost_cached if is_cached else cost_uncached

        pool.append((content_id, z, q, cost))

    return pool

def item_z(item):
    return float(item[0])

def item_q(item):
    return float(item[1])

def item_source(item):
    return item[2]

def item_cost(item):
    return float(item[3])

def cost_helper(slate, z):
    if len(slate) == 0:
        return 0.0

    z_items = [item_z(it) for it in slate]
    return avg_sim(z, z_items)

def cost_friendly_slate(pool,z,num_contents,k_f,k_g,gamma=0.5,alpha=0.5,epsilon=0.3):
    """Optimize a slate while considering cost constraints."""
    if len(pool) == 0:
        return []

    slate_ref = maxsim_slate(z, pool, k_f, k_g)
    sim_ref = cost_helper(slate_ref, z)

    sim_floor = alpha * sim_ref

    remaining = pool.copy()
    slate = []
    sims = []

    for step in range(num_contents):
        best_idx = None
        best_score = np.inf

        for idx, item in enumerate(remaining):
            z_item = item_z(item)
            q_item = item_q(item)

            possible_slate = slate + [item]

            sim_val = float(similarity(z, z_item))
            if not np.isfinite(sim_val):
                continue

            poss_sims = sims + [sim_val]

            if len(poss_sims) == 0:
                continue

            sim_slate_avg = float(np.mean(poss_sims))

            if sim_slate_avg < sim_floor:
                continue

            z_cont = np.array([item_z(it) for it in possible_slate], dtype=float)
            q_cont = np.array([item_q(it) for it in possible_slate], dtype=float)

            _, z_bar = attention(z, z_cont, q_cont, beta=0.8, c=0.5)

            s_next = np.clip((1.0 - gamma) * z + gamma * z_bar, -1.0, 1.0)

            # Same objective as before:
            # minimize predicted extremeness, with a small quality bonus.
            score = abs(s_next) - epsilon * q_item +  item_cost(item)

            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        best_item = remaining.pop(best_idx)
        slate.append(best_item)

        sims.append(float(similarity(z, item_z(best_item))))

    return slate

def greedy_cost_slate(pool, z, num_contents,lambda_slate, k_f, k_g, gamma=0.5, alpha=0.5, epsilon=0.3):
    """
    Builds the slate using only depolarization/relevance logic.
    Cost is NOT used inside this function.

    Each item in pool has format:
        (z_item, q_item, source, cost)
    """

    slate_ref = maxsim_slate(z, pool, k_f, k_g)
    sim_ref = greedy_helper(slate_ref, z)

    sim_floor = alpha * sim_ref

    remaining = pool.copy()
    slate = []
    sims = []

    for step in range(num_contents):
        best_idx = None
        best_score = np.inf

        for idx, item in enumerate(remaining):
            z_item = item_z(item)
            q_item = item_q(item)

            possible_slate = slate + [item]

            sim_val = float(similarity(z, z_item))

            if not np.isfinite(sim_val):
                continue

            poss_sims = sims + [sim_val]
            sim_slate_avg = float(np.mean(poss_sims))

            if sim_slate_avg < sim_floor:
                continue

            z_cont = np.array([item_z(it) for it in possible_slate], dtype=float)
            q_cont = np.array([item_q(it) for it in possible_slate], dtype=float)

            _, z_bar = attention(z, z_cont, q_cont, beta=0.8, c=0.5)

            s_next = (1.0 - gamma) * z + gamma * z_bar

            # Objective: depolarize + small quality bonus.
            score = abs(s_next) - epsilon * q_item + lambda_slate * item_cost(item)

            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        best_item = remaining.pop(best_idx)
        slate.append(best_item)
        sims.append(float(similarity(z, item_z(best_item))))

    return slate

def cost_pool(G,user,t,M_f,M_g,z_post_t,q_post_t,z_gen,q_gen,generic_cache_mask,d_target=10,rng=None,use_predecessors=True,famous_degree_type="out"):
    if rng is None:
        rng = np.random.default_rng()

    pool = []

    # --------------------------------------------------
    # Friend posts
    # --------------------------------------------------
    if use_predecessors:
        friends = list(G.predecessors(user))
    else:
        friends = list(G.successors(user))

    if M_f is not None and len(friends) > M_f:
        friends = list(rng.choice(friends, size=M_f, replace=False))

    for j in friends:
        z_j = float(z_post_t[j])
        q_j = float(q_post_t[j])

        if G.is_directed():
            if famous_degree_type == "out":
                deg_j = G.out_degree(j)
            elif famous_degree_type == "in":
                deg_j = G.in_degree(j)
            elif famous_degree_type == "total":
                deg_j = G.degree(j)
            else:
                raise ValueError(
                    "famous_degree_type must be 'out', 'in', or 'total'."
                )
        else:
            deg_j = G.degree(j)

        is_famous = deg_j > d_target

        cost = 0.0 if is_famous else 1.0

        pool.append((z_j, q_j, "friend", cost))

    # --------------------------------------------------
    # Generic posts
    # --------------------------------------------------
    z_gen_t, q_gen_t = get_generic_round(z_gen, q_gen, t)

    for m in range(M_g):
        z_m = float(z_gen_t[m])
        q_m = float(q_gen_t[m])

        is_cached = int(generic_cache_mask[t, m]) == 1

        cost = 0.0 if is_cached else 1.0

        pool.append((z_m, q_m, "generic", cost))

    return pool

# the budget cosntraint version
def greedy_cost_constrained_slate(pool,z,num_contents,k_f,k_g,gamma=0.5,
    alpha=0.5,epsilon=0.3,cost_budget=None,cost_ratio=None,fallback_relax_cost=False,eta = 0.4):

    if len(pool) == 0:
        return []

    # Reference slate for similarity threshold
    slate_ref = maxsim_slate(z, pool, k_f, k_g)
    sim_ref = greedy_helper(slate_ref, z)

    sim_floor = alpha * sim_ref

    # Cost budget
    if cost_budget is None:
        if cost_ratio is not None:
            ref_cost = slate_cost(slate_ref)
            cost_budget = cost_ratio * ref_cost
        else:
#        # No cost constraint
            cost_budget = np.inf

    remaining = pool.copy()
    slate = []
    sims = []
    current_cost = 0.0

    for step in range(num_contents):
        best_idx = None
        best_score = np.inf

        for idx, item in enumerate(remaining):
            z_item = item_z(item)
            q_item = item_q(item)
            c_item = item_cost(item)

            possible_slate = slate + [item]

            sim_val = float(similarity(z, z_item))

            if not np.isfinite(sim_val):
                continue

            poss_sims = sims + [sim_val]
            sim_slate_avg = float(np.mean(poss_sims))

            # Hard similarity constraint
            if sim_slate_avg < sim_floor:
                continue

            # Hard cost constraint
            possible_cost = current_cost + float(c_item)

            if possible_cost > cost_budget:
                continue

            z_cont = np.array([item_z(it) for it in possible_slate], dtype=float)
            q_cont = np.array([item_q(it) for it in possible_slate], dtype=float)

            _, z_bar = attention(z, z_cont, q_cont, beta=0.8, c=0.5)

            s_next = (1.0 - gamma) * z + gamma * z_bar

            # Objective: depolarization + quality.
            # Cost is NOT in the score.
            score = abs(s_next) - epsilon * q_item
            #score = - epsilon * q_item + eta * possible_cost

            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        best_item = remaining.pop(best_idx)
        slate.append(best_item)
        sims.append(float(similarity(z, item_z(best_item))))
        current_cost += float(item_cost(best_item))

    return slate

def random_cost_constrained_slate(pool, z, num_contents, k_f, k_g,
                                  alpha=0.5, cost_ratio=None,
                                  cost_budget=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if len(pool) == 0:
        return []

    slate_ref = maxsim_slate(z, pool, k_f, k_g)
    sim_ref = greedy_helper(slate_ref, z)
    sim_floor = alpha * sim_ref

    if cost_budget is None:
        if cost_ratio is not None:
            cost_budget = cost_ratio * slate_cost(slate_ref)
        else:
            cost_budget = np.inf

    candidates = list(pool)
    rng.shuffle(candidates)

    slate = []
    sims = []
    current_cost = 0.0

    for item in candidates:
        if len(slate) >= num_contents:
            break

        sim_val = float(similarity(z, item_z(item)))
        if not np.isfinite(sim_val):
            continue

        sim_slate_avg = float(np.mean(sims + [sim_val]))
        if sim_slate_avg < sim_floor:
            continue

        c_item = float(item_cost(item))
        if current_cost + c_item > cost_budget:
            continue

        slate.append(item)
        sims.append(sim_val)
        current_cost += c_item

    return slate

def hard_slate(pool, z_user, num_contents, k_f, k_g, cost_budget=None, cost_ratio=None):
    """
    Build a max-similarity slate under a hard cost constraint only.

    Unlike greedy_cost_constrained_slate, this function does not optimize
    depolarization and does not enforce a similarity-floor constraint.
    It simply keeps the max-similarity logic and filters selections through
    the cost budget.
    """

    if len(pool) == 0:
        return []

    slate_ref = maxsim_slate(z_user, pool, k_f, k_g)

    if cost_budget is None:
        if cost_ratio is not None:
            ref_cost = slate_cost(slate_ref)
            cost_budget = cost_ratio * ref_cost
        else:
            cost_budget = np.inf

    if not np.isfinite(cost_budget):
        return slate_ref[:num_contents]

    friend_pool = sorted(
        [it for it in pool if item_source(it) == "friend"],
        key=lambda it: similarity(z_user, item_z(it)),
        reverse=True,
    )
    generic_pool = sorted(
        [it for it in pool if item_source(it) == "generic"],
        key=lambda it: similarity(z_user, item_z(it)),
        reverse=True,
    )

    slate = []
    current_cost = 0.0

    for item in friend_pool:
        if len([it for it in slate if item_source(it) == "friend"]) >= k_f:
            break
        next_cost = current_cost + item_cost(item)
        if next_cost <= cost_budget:
            slate.append(item)
            current_cost = next_cost

    for item in generic_pool:
        if len([it for it in slate if item_source(it) == "generic"]) >= k_g:
            break
        next_cost = current_cost + item_cost(item)
        if next_cost <= cost_budget:
            slate.append(item)
            current_cost = next_cost

    if len(slate) < num_contents:
        remaining = [it for it in slate_ref if it not in slate]
        for item in remaining:
            if len(slate) >= num_contents:
                break
            next_cost = current_cost + item_cost(item)
            if next_cost <= cost_budget:
                slate.append(item)
                current_cost = next_cost

    return slate[:num_contents]

def precompute_generic_cache_mask(T, M_g, cache_fraction=0.5, rng=None):
    """
    Returns a T x M_g binary array.

    cache_mask[t, m] = 1 means generic item m at round t is cached.
    cache_mask[t, m] = 0 means it is not cached.
    """

    if rng is None:
        rng = np.random.default_rng()

    cache_fraction = float(cache_fraction)

    if cache_fraction < 0.0 or cache_fraction > 1.0:
        raise ValueError("cache_fraction must be in [0, 1].")

    C_g = int(round(cache_fraction * M_g))

    cache_mask = np.zeros((T, M_g), dtype=int)

    for t in range(T):
        if C_g > 0:
            cached_idx = rng.choice(M_g, size=C_g, replace=False)
            cache_mask[t, cached_idx] = 1

    return cache_mask

def get_generic_round(z_gen, q_gen, t):
    """
    Handles both possible shapes:
        z_gen.shape == (T, M_g)
    or
        z_gen.shape == (M_g, T)
    """

    z_gen = np.asarray(z_gen)
    q_gen = np.asarray(q_gen)

    if z_gen.shape[0] > t:
        z_t = z_gen[t]
        q_t = q_gen[t]
    else:
        z_t = z_gen[:, t]
        q_t = q_gen[:, t]

    return np.asarray(z_t, dtype=float), np.asarray(q_t, dtype=float)

def simple_cost_user_pool(G,user,t,M_f,M_g,z_post_t,q_post_t,z_gen,q_gen,cache_mask,rng=None,use_predecessors=True,):
    """
    Returns pool items with format:
        (z, q, source, cost)

    Friend posts:
        cost = 1.0

    Generic posts:
        cost = 0.0 if cached
        cost = 1.0 if uncached
    """

    if rng is None:
        rng = np.random.default_rng()

    pool = []

    # --------------------------
    # Friend posts
    # --------------------------
    if use_predecessors:
        friends = list(G.predecessors(user))
    else:
        friends = list(G.successors(user))

    if M_f is not None and len(friends) > M_f:
        friends = list(rng.choice(friends, size=M_f, replace=False))

    for j in friends:
        z_j = float(z_post_t[j])
        q_j = float(q_post_t[j])

        source = "friend"
        cost = 1.0

        pool.append((z_j, q_j, source, cost))

    # --------------------------
    # Generic posts
    # --------------------------
    z_gen_t, q_gen_t = get_generic_round(z_gen, q_gen, t)

    for m in range(M_g):
        z_m = float(z_gen_t[m])
        q_m = float(q_gen_t[m])

        cached = int(cache_mask[t, m])
        cost = 0.0 if cached == 1 else 1.0

        source = "generic"

        pool.append((z_m, q_m, source, cost))

    return pool

def slate_cost(slate):
    """
    Normalized slate cost.

    Since each item has cost 0 or 1, this returns the fraction
    of expensive/uncached items in the slate.
    """

    if len(slate) == 0:
        return 0.0

    return float(np.mean([item_cost(item) for item in slate]))
