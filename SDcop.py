import time
from typing import Dict, List, Tuple
from DCOP import parse_assignment_from_output, run_pydcop_solve
from ESOPInstance import ESOPInstance, Observation, Task, User
from GreedySolver import greedy_schedule, greedy_schedule_P_u

def build_plan_for_user_with_extra(
    instance,
    user_id,
    extra_obs):
    """
    Résout le sous-problème P_u ou P_{u ∪ {o}} par un greedy,
    en considérant seulement :
      - les tâches de u,
      - + éventuellement l'observation extra_obs (de u0).
    Retourne un plan {sat_id: [(obs, t_start), ...]} pour u.
    """

    # Filtrer les observations pertinentes
    if extra_obs is None:
        filtered_obs = [
            o for o in instance.observations
            if o.owner == user_id
        ]
    else:
        filtered_obs = [
            o for o in instance.observations
            if o.owner == user_id
        ] + [extra_obs]

    # On crée une instance restreinte P_u (ou P_{u∪{o}})
    # avec les mêmes satellites, mais uniquement ces observations.
    inst_restricted = ESOPInstance(
        nb_satellites=instance.nb_satellites,
        nb_users=1,  # 1 exclusif ici, mais ce champ est surtout descriptif
        nb_tasks=instance.nb_tasks,
        horizon=instance.horizon,
        satellites=instance.satellites,
        users=[u for u in instance.users if u.uid == user_id],
        tasks=instance.tasks,
        observations=filtered_obs
    )

    # Un greedy sur cette instance restreinte
    # renvoie user_plans[uid][sid]; on ne garde que user_id.
    all_plans = greedy_schedule(inst_restricted)
    return all_plans.get(user_id, {})
    

def total_reward(plan_for_user: Dict[str, List[Tuple[Observation, int]]]) -> int:
    return sum(obs.reward for sat_plan in plan_for_user.values() for (obs, _) in sat_plan)


def compute_pi(instance: ESOPInstance, user_id: str, obs: Observation):
    """
    π(o, M_u) = coût marginal pour user_id si il accepte obs de u0,
    évalué par replanification complète P_u et P_{u ∪ {o}} (via greedy).
    Retourne un coût (>= 0, car DPOP minimise) ou None si aucun gain.
    """

    # Plan avant (P_u)
    plan_before = build_plan_for_user_with_extra(instance, user_id, None)
    reward_before = total_reward(plan_before)

    # Plan après (P_{u ∪ {o}})
    plan_after = build_plan_for_user_with_extra(instance, user_id, obs)
    reward_after = total_reward(plan_after)

    # Si obs n'améliore pas le plan, l'utilisateur ne devrait pas la prendre
    if reward_after <= reward_before:
        return None

    gain = reward_after - reward_before
    # DCOP minimise -> coût = reward_before - reward_after = -gain
    # on retourne un entier >= 0 en inversant le signe
    cost = -gain
    return cost

import yaml

def generate_sdcop_yaml_for_request(
    instance: ESOPInstance,
    request: Task,
    yaml_path: str
) -> bool:
    """
    Génère un DCOP YAML pour une requête centrale 'request' (r),
    conforme au modèle s-dcop :
      - variables x_{u,o} pour u exclusif, o opportunité de r appartenant à u0,
      - coût local = π(o, M_u),
      - contrainte: au plus un x_{u,o} = 1.
    Retourne False si aucun candidat n'existe.
    """

    # Utilisateurs exclusifs (U_ex)
    exclusive_users: List[User] = [u for u in instance.users if u.uid != "u0"]

    # Opportunités de cette requête pour le central (o ∈ θ_r, owner = u0)
    obs_r: List[Observation] = [
        o for o in request.opportunities
        if o.owner == "u0"
    ]

    domains = {
        "binary": {"values": [0, 1]}
    }
    variables = {}
    constraints = {}
    agents = set()
    var_names = []

    # Pour chaque utilisateur exclusif u et observation o ∈ θ_r
    for u in exclusive_users:
        u_id = u.uid

        for o in obs_r:
            # Vérifier qu'il existe une fenêtre exclusive de u compatible avec o [file:2]
            can_take = any(
                w.satellite == o.satellite and
                not (w.t_end <= o.t_start or w.t_start >= o.t_end)
                for w in u.exclusive_windows
            )
            if not can_take:
                continue

            # Calcul de π(o, M_u) via le greedy local
            pi_val = compute_pi(instance, u_id, o)
            if pi_val is None:
                # pas intéressant pour u
                continue

            agents.add(u_id)
            var_name = f"x_{u_id}_{o.oid}"
            var_names.append(var_name)

            variables[var_name] = {
                "domain": "binary",
                "agent": u_id
            }

            # Fonction de coût locale : c(x) = pi_val * x
            # En PyDcop, on peut écrire une fonction Python simple inline
            constraints[f"c_pi_{u_id}_{o.oid}"] = {
                "type": "intention",
                "function": f"lambda {var_name}: {pi_val} * {var_name}"
            }

    if not var_names:
        # Aucun candidat, aucune variable -> pas de DCOP pour cette requête
        return False

    # Contrainte "au plus un" sur la somme des x_{u,o}
    # coût 0 si somme <= 1, sinon très grand (pénalité)
    sum_expr = " + ".join(var_names) if len(var_names) > 1 else var_names[0]
    constraints[f"c_atmost1_{request.tid}"] = {
        "type": "intention",
        "function": f"lambda {', '.join(var_names)}: 0 if ({sum_expr}) <= 1 else 1e9"
    }

    dcop_dict = {
        "name": f"sdcop_{request.tid}",
        "objective": "min",
        "domains": domains,
        "agents": list(agents),
        "variables": variables,
        "constraints": constraints
    }

    with open(yaml_path, "w") as f:
        yaml.dump(dcop_dict, f, sort_keys=False)

    return True

import os
from typing import Tuple, Optional


def sdcop_with_pydcop(
    instance: ESOPInstance,
    algo: str = "dpop",
    base_yaml_name: str = "sdcop_req"
):
    """
    Implémentation fidèle (et séquentielle) de l'algorithme s-dcop de l'article,
    avec PyDcop / DPOP comme solveur DCOP.

    Retourne :
      - current_plans : {u_id -> plan utilisateur}
      - assignments_global : [(request_id, user_id, observation)]
      - avg_time_per_request : temps moyen DPOP par requête centrale.
    """

    # 1) Utilisateurs exclusifs
    exclusive_users: List[User] = [u for u in instance.users if u.uid != "u0"]

    # 2) Plans initiaux M_u (P_u résolu par greedy)
    current_plans: Dict[str, Dict[str, List[Tuple[Observation, int]]]] = {}
    for u in exclusive_users:
        current_plans[u.uid] = greedy_schedule_P_u(instance, u.uid)

    # 3) Requêtes du central (u0) triées par deadline
    central_requests: List[Task] = [r for r in instance.tasks if r.owner == "u0"]
    central_requests.sort(key=lambda r: r.t_end)

    assignments_global: List[Tuple[str, str, Observation]] = []
    times: List[float] = []

    for r in central_requests:
        yaml_path = f"{base_yaml_name}_{r.tid}.yaml"

        ok = generate_sdcop_yaml_for_request(instance, r, yaml_path)
        if not ok:
            # aucun candidat pour cette requête
            continue

        # 4) Résolution DCOP avec DPOP
        t0 = time.time()
        output = run_pydcop_solve(yaml_path, algo=algo)
        t1 = time.time()
        times.append(t1 - t0)

        if output is None:
            continue

        assignment = parse_assignment_from_output(output)
        if not assignment:
            continue

        # 5) Extraire (u, o) gagnant : x_{u,o} = 1
        winner_u: Optional[str] = None
        chosen_obs: Optional[Observation] = None

        for var_name, val in assignment.items():
            if val != 1:
                continue
            try:
                _, u_id, o_id = var_name.split("_", 2)
            except ValueError:
                continue

            try:
                obs = next(o for o in instance.observations if o.oid == o_id)
            except StopIteration:
                continue

            winner_u = u_id
            chosen_obs = obs
            break  # au plus un gagnant

        if winner_u is not None and chosen_obs is not None:
            # 6) Mettre à jour le plan du gagnant en replanifiant P_u
            # (les données de l'instance restent les mêmes ; le greedy utilisera toutes les obs)
            current_plans[winner_u] = greedy_schedule_P_u(instance, winner_u)
            assignments_global.append((r.tid, winner_u, chosen_obs))

        # optionnel : supprimer le fichier yaml
        if os.path.exists(yaml_path):
            os.remove(yaml_path)

    avg_time = sum(times) / len(times) if times else 0.0
    return current_plans, assignments_global, avg_time
