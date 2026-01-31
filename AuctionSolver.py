from copy import deepcopy
import sys
from ESOPInstance import ESOPInstance
from GreedySolver import greedy_schedule, greedy_schedule_P_u


def plan_reward(plan, user_id):
    """
        Calcule le reward total d'un plan utilisateur.
    """
    total = 0
    user_plan = plan.get(user_id, {})
    for sat_plan in user_plan.values():
        if isinstance(sat_plan, list):
            total += sum(obs.reward for obs, _ in sat_plan)
    return total

def get_schedule(plan, task_id):
    """
        Extrait le schedule pour une tâche spécifique
    """
    for user_plan in plan.values():
        for sat_plan in user_plan.values():
            if isinstance(sat_plan, list):
                for obs_t in sat_plan:
                    if len(obs_t) == 2 and obs_t[0].task_id == task_id:
                        return obs_t
    return None

def create_instance_with_fixed_observations(instance, fixed_obs):
    """
        Crée une nouvelle instance où certaines observations sont fixées (notamment pour plan u0)
    """
    new_inst = deepcopy(instance)
    fixed_tasks = set()
    for sat_plans in fixed_obs.values():
        for obs, _ in sat_plans:
            fixed_tasks.add(obs.task_id)

    new_inst.tasks = [t for t in new_inst.tasks if t.tid not in fixed_tasks]
    new_inst.nb_tasks = len(new_inst.tasks)
    new_inst.observations = [o for o in new_inst.observations if o.task_id not in fixed_tasks]
    return new_inst

def bid(u_id, instance, request):
    """
        Calcule l'enchère d'un utilisateur u_id pour une requête donnée.
    """
    u = next((user for user in instance.users if user.uid == u_id), None)
    if u is None:
        return 0.0, None
    
    # UNIQUEMENT tâches/obs de u
    tasks_u = [t for t in instance.tasks if t.owner == u_id]
    obs_u = [o for o in instance.observations if o.owner == u_id]
    inst_u = ESOPInstance(nb_satellites=instance.nb_satellites, nb_users=1, nb_tasks=len(tasks_u),
                        horizon=instance.horizon, satellites=instance.satellites, users=[u],
                        tasks=tasks_u, observations=obs_u)
    Mu = greedy_schedule_P_u(inst_u, u_id)
    old_reward = plan_reward({"temp": Mu}, u_id)
    
    # tâches/obs de u + r
    tasks_u_r = tasks_u + [request]
    obs_u_r = obs_u + [o for o in instance.observations if o.task_id == request.tid]
    inst_u_r = ESOPInstance(nb_satellites=instance.nb_satellites, nb_users=1, nb_tasks=len(tasks_u_r),
                            horizon=instance.horizon, satellites=instance.satellites, users=[u],
                            tasks=tasks_u_r, observations=obs_u_r)
    new_plan = greedy_schedule_P_u(inst_u_r, u_id)
    new_reward = plan_reward({"temp": new_plan}, u_id)
    
    bid_value = new_reward - old_reward # Gain marginal LOCAL
    schedule_r = get_schedule({"temp": new_plan}, request.tid)
    return bid_value, schedule_r

def integrate_observation(current_plan, new_obs_schedule, instance):
    """
        Opérateur (+) (cercle) : refait greedy sur instance mise à jour
    """
    obs, t_start = new_obs_schedule
    u_id = obs.owner
    u = next((user for user in instance.users if user.uid == u_id), None)
    if u is None:
        return current_plan

    tasks_u = [t for t in instance.tasks if t.owner == u_id]
    obs_u = [o for o in instance.observations if o.owner == u_id and o.task_id != obs.task_id]

    inst_u = ESOPInstance(nb_satellites=instance.nb_satellites, nb_users=1, nb_tasks=len(tasks_u),
                        horizon=instance.horizon, satellites=instance.satellites, users=[u],
                        tasks=tasks_u, observations=obs_u)

    return greedy_schedule_P_u(inst_u, u_id)

############ PSI
def psi_solve(instance):
    """
        Algorithme PSI de l'article
        retourne (plans, nb_messages, comm_load)
    """
    nb_messages = 0
    comm_load = 0

    Mu0 = {}
    exclusive_users = [u for u in instance.users if u.uid != "u0"]

    # Résolution locale initiale pour chaque user
    initial_plans = {u.uid: greedy_schedule_P_u(instance, u.uid) for u in exclusive_users}

    # Requêtes du central (items)
    u0_tasks = [t for t in instance.tasks if t.owner == "u0"]

    # Annonce globale des items à tous les exclusifs
    # 1 message par user contenant la liste complète des items (hyp. choisie)
    for u in exclusive_users:
        nb_messages += 1
        comm_load += sys.getsizeof((u.uid, u0_tasks))

    allocations = []

    # Bidding en parallèle sur chaque requête
    for r in u0_tasks:
        bids_r = {}
        for u in exclusive_users:
            b_val, sigma = bid(u.uid, instance, r)
            bids_r[u.uid] = (b_val, sigma)
            # 1 message bid (valeur + éventuellement schedule)
            nb_messages += 1
            comm_load += sys.getsizeof((r.tid, u.uid, b_val))

        if not bids_r:
            continue

        winner_id = max(bids_r, key=lambda uid: bids_r[uid][0])
        winner_bid, sigma_w = bids_r[winner_id]

        if sigma_w is not None and winner_bid > 0:
            allocations.append((winner_bid, r.tid, winner_id, sigma_w))

    # determination globale du winner
    allocations.sort(key=lambda x: (-x[0], x[1]))
    for _, task_id, winner_id, sigma_w in allocations:
        obs, t = sigma_w
        sid = obs.satellite
        Mu0.setdefault(sid, []).append(sigma_w)
        # notif. au gagnant
        nb_messages += 1
        comm_load += sys.getsizeof((task_id, winner_id, sigma_w))

    # Plan de u0 avec les obs fixées
    inst_u0_fixed = create_instance_with_fixed_observations(instance, Mu0)
    final_u0_plan = greedy_schedule(inst_u0_fixed).get("u0", {})

    return {**initial_plans, "u0": final_u0_plan}, nb_messages, comm_load

##### SSI
def ssi_solve(instance, sort_key=lambda r: r.t_end):
    """
        Algorithme SSI.
    """
    nb_messages = 0
    comm_load = 0

    Mu0 = {}
    exclusive_users = [u for u in instance.users if u.uid != "u0"]

    # Plans locaux initiaux
    user_plans = {u.uid: greedy_schedule_P_u(instance, u.uid) for u in exclusive_users}

    # Requêtes de u0 triées
    u0_tasks = sorted([t for t in instance.tasks if t.owner == "u0"], key=sort_key)

    # Boucle séquentielle sur les requêtes
    for r in u0_tasks:
        for u in exclusive_users: # annonce de la requête r à chaque user
            nb_messages += 1
            comm_load += sys.getsizeof((u.uid, r.tid))

        bids_r = {}
        for u in exclusive_users:
            b_val, sigma = bid(u.uid, instance, r)
            bids_r[u.uid] = (b_val, sigma)
            nb_messages += 1
            comm_load += sys.getsizeof((r.tid, u.uid, b_val))

        if not bids_r:
            continue

        winner_id = max(bids_r, key=lambda uid: bids_r[uid][0])
        winner_bid, sigma_w = bids_r[winner_id]

        if sigma_w is None or winner_bid <= 0:
            continue

        obs, t = sigma_w
        sid = obs.satellite
        Mu0.setdefault(sid, []).append(sigma_w)

        # màj du plan gagnant
        user_plans[winner_id] = integrate_observation(user_plans[winner_id], sigma_w, instance)

        # notif winner
        nb_messages += 1
        comm_load += sys.getsizeof((r.tid, winner_id, sigma_w))

    # et plan final de u0
    inst_u0_fixed = create_instance_with_fixed_observations(instance, Mu0)
    final_u0_plan = greedy_schedule(inst_u0_fixed).get("u0", {})

    return {**user_plans, "u0": final_u0_plan}, nb_messages, comm_load

######## REGRET AUCTION (extension SSI)
def regret_bid(u_id, instance, request, all_user_plans, history_bids, alpha = 0.1):
    """
        Enchère par regret : bid = gain_marginal + alpha * regret_passé
    """
    classic_bid, schedule = bid(u_id, instance, request)
    past_regret = history_bids.get(u_id, 0.0)
    regret_bonus = alpha * past_regret
    final_bid = classic_bid + regret_bonus
    return final_bid, schedule

def regret_auction_solve(instance, sort_key=lambda r: r.t_end, alpha = 0.1, n_rounds = 3):
    """
        Regret Auction : extension multi-rounds d'SSI
    """
    nb_messages = 0 # toujours sous hyp. choisies car manque d'infos dans l'article.
    comm_load = 0

    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    history_bids = {u.uid: 0.0 for u in exclusive_users} # historique des regrets

    # Plans initiaux
    user_plans = {u.uid: greedy_schedule_P_u(instance, u.uid) for u in exclusive_users}
    best_plans = deepcopy(user_plans)
    best_score = 0.0

    u0_tasks = sorted([t for t in instance.tasks if t.owner == "u0"], key=sort_key)

    for round_num in range(n_rounds):
        Mu0 = {}

        for r in u0_tasks:
            # Annonce r + info regret aux users
            for u in exclusive_users:
                nb_messages += 1
                comm_load += sys.getsizeof((u.uid, r.tid, history_bids[u.uid]))

            bids_r = {}
            for u in exclusive_users:
                bid_val, schedule = regret_bid(u.uid, instance, r, user_plans, history_bids, alpha)
                bids_r[u.uid] = (bid_val, schedule)
                nb_messages += 1
                comm_load += sys.getsizeof((r.tid, u.uid, bid_val))

            if not bids_r:
                continue

            winner_id = max(bids_r, key=lambda uid: bids_r[uid][0])
            winner_bid, sigma_w = bids_r[winner_id]

            marginal_bid = winner_bid - history_bids.get(winner_id, 0.0)
            if sigma_w is None or marginal_bid <= 0:
                continue

            obs, t = sigma_w
            sid = obs.satellite
            Mu0.setdefault(sid, []).append(sigma_w)

            user_plans[winner_id] = integrate_observation(user_plans[winner_id], sigma_w, instance)

            # notification gagnant
            nb_messages += 1
            comm_load += sys.getsizeof((r.tid, winner_id, sigma_w))

            for loser_id, (loser_bid, _) in bids_r.items(): # màj regret perdants
                if loser_id != winner_id:
                    history_bids[loser_id] += loser_bid * 0.1

        inst_u0_fixed = create_instance_with_fixed_observations(instance, Mu0)
        final_u0_plan = greedy_schedule(inst_u0_fixed).get("u0", {})
        round_plans = {**user_plans, "u0": final_u0_plan}
        round_score = sum(sum(sum(obs.reward for obs, _ in (obslist or [])) for obslist in sat_plans.values()) for sat_plans in round_plans.values())

        if round_score > best_score:
            best_score = round_score
            best_plans = deepcopy(round_plans)

    return best_plans, nb_messages, comm_load
