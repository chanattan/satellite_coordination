import os
import time
from typing import Dict, List, Tuple, Optional

import yaml

from DCOP import run_pydcop_solve, parse_assignment_from_output
from ESOPInstance import ESOPInstance, Observation, Task, User
from GreedySolver import greedy_schedule_P_u as greedy_schedule_for_user


# ----------------------------------------------------------------------
# Helpers : plan local et coût π(o, M_u)
# ----------------------------------------------------------------------

def build_restricted_plan_for_user(
    instance: ESOPInstance,
    user_id: str,
    extra_obs: Optional[Observation],
    accepted_u0_obs: List[Observation],
) -> Dict[str, List[Tuple[Observation, int]]]:
    """
    Construit un plan glouton pour user_id en ne considérant que :
      - ses propres observations,
      - + accepted_u0_obs (obs de u0 déjà acceptées pour ce user),
      - + extra_obs si non None.

    On utilise pour cela greedy_schedule_for_user sur une sous-instance ESOP
    contenant uniquement ces observations pour user_id.

    Retourne un plan {sat_id: [(obs, t_start), ...]} pour user_id.
    """
    # User u
    u = next(user for user in instance.users if user.uid == user_id)

    # Observations de u + obs de u0 déjà acceptées (pour ce user)
    base_obs = [
        o for o in instance.observations
        if (o.owner == user_id) or (o in accepted_u0_obs)
    ]
    if extra_obs is not None:
        base_obs = base_obs + [extra_obs]

    # Sous-instance restreinte pour P_u (ou P_u ∪ {o})
    inst_u = ESOPInstance(
        nb_satellites=instance.nb_satellites,
        nb_users=1,
        nb_tasks=instance.nb_tasks,
        horizon=instance.horizon,
        satellites=instance.satellites,
        users=[u],
        tasks=instance.tasks,
        observations=base_obs,
    )

    # Greedy local (même logique que solve(P_u) de l’article)
    all_plans_u = greedy_schedule_for_user(inst_u, user_id)
    return all_plans_u


def compute_reward_from_plan(plan_for_user: Dict[str, List[Tuple[Observation, int]]]) -> int:
    return sum(obs.reward for sat_plan in plan_for_user.values() for (obs, _) in sat_plan)


def compute_pi(
    instance: ESOPInstance,
    user_id: str,
    obs: Observation,
    accepted_u0_obs_for_u: List[Observation],
) -> Optional[int]:
    """
    π(o, M_u) = coût marginal pour user_id s'il accepte obs de u0,
    en replanifiant P_u et P_u ∪ {o} via le greedy local.

    - M_u est représenté implicitement par accepted_u0_obs_for_u
      (obs de u0 déjà acceptées) + obs propres de u.
    - On retourne un coût >= 0 (DPOP minimise).

    Retourne None si obs n’apporte pas de gain.
    """
    # Plan avant (P_u) : propres obs + accepted_u0_obs_for_u
    plan_before = build_restricted_plan_for_user(
        instance, user_id, extra_obs=None, accepted_u0_obs=accepted_u0_obs_for_u
    )
    reward_before = compute_reward_from_plan(plan_before)

    # Plan après (P_u ∪ {o})
    plan_after = build_restricted_plan_for_user(
        instance, user_id, extra_obs=obs, accepted_u0_obs=accepted_u0_obs_for_u
    )
    reward_after = compute_reward_from_plan(plan_after)

    if reward_after <= reward_before:
        # pas de gain -> u ne devrait pas prendre cette obs
        return None

    gain = reward_after - reward_before
    # DCOP minimise : coût = -gain (valeur <= 0). Pour rester cohérent avec pyDCOP
    # on renvoie un coût (numérique) utilisé directement dans la fonction.
    # On peut garder le signe négatif, pyDCOP acceptera des coûts négatifs.
    return -gain


# ----------------------------------------------------------------------
# Génération du DCOP pour une requête (s-dcop)
# ----------------------------------------------------------------------

def generate_sdcop_yaml_for_request(
    instance: ESOPInstance,
    request: Task,
    current_accepted: Dict[str, List[Observation]],
    yaml_path: str,
) -> bool:
    """
    Génère un DCOP YAML pour une requête centrale 'request' (r), conforme au modèle s-dcop :

    - agents = utilisateurs exclusifs,
    - variables x_{u,o} pour u exclusif, o opportunité de r appartenant à u0
      (et compatible avec au moins une exclusive de u),
    - coût local = π(o, M_u) * x_{u,o},
    - contrainte "au plus un" sur la somme des x_{u,o}.

    current_accepted : {u_id -> liste des obs de u0 déjà acceptées pour ce user}.

    Retourne False si aucun candidat n'existe (pas de DCOP à résoudre).
    """
    exclusive_users: List[User] = [u for u in instance.users if u.uid != "u0"]
    # Opportunités de cette requête pour le central (owner = u0)
    obs_r: List[Observation] = [
        o for o in request.opportunities if o.owner == "u0"
    ]

    variables = {}
    constraints = {}
    agents: List[str] = []
    var_names: List[str] = []

    for u in exclusive_users:
        u_id = u.uid
        agents.append(u_id)
        accepted_for_u = current_accepted.get(u_id, [])

        for o in obs_r:
            # Vérifier qu'au moins une exclusive de u intersecte la fenêtre de o
            can_take = any(
                w.satellite == o.satellite and
                not (w.t_end <= o.t_start or w.t_start >= o.t_end)
                for w in u.exclusive_windows
            )
            if not can_take:
                continue

            # Calcul de π(o, M_u) via le greedy local
            pi_val = compute_pi(instance, u_id, o, accepted_for_u)
            if pi_val is None:
                continue

            v_name = f"x_{u_id}_{o.oid}"
            variables[v_name] = {"domain": "binary", "agent": u_id}
            var_names.append(v_name)

            c_name = f"c_pi_{u_id}_{o.oid}"
            # Coût local : pi_val * x_{u_id,o}
            constraints[c_name] = {
                "type": "intention",
                "function": f"{pi_val} * {v_name}",
            }

    if not var_names:
        # Aucun candidat : cette requête ne génère pas de DCOP
        return False

    # Contrainte "au plus un" sur la somme des x_{u,o}
    total_expr = " + ".join(var_names) if len(var_names) > 1 else var_names[0]
    constraints[f"c_atmost1_{request.tid}"] = {
        "type": "intention",
        "function": f"0 if {total_expr} <= 1 else 1e9",
    }

    dcop_dict = {
        "name": f"sdcop_{request.tid}",
        "objective": "min",
        "domains": {"binary": {"values": [0, 1]}},
        "agents": agents,
        "variables": variables,
        "constraints": constraints,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(dcop_dict, f, sort_keys=False)

    return True


# ----------------------------------------------------------------------
# Implémentation séquentielle s-dcop (Algorithm 5)
# ----------------------------------------------------------------------

def sdcop_with_pydcop(
    instance: ESOPInstance,
    algo: str = "dpop",
    base_yaml_name: str = "sdcop_req",
):
    """
    Implémentation séquentielle de l'algorithme s-dcop (section 5.2).

    Étapes :
      1) Pour chaque utilisateur exclusif u, on résout P_u avec le greedy local
         -> dans cette version, on ne stocke pas explicitement M_u, on encode
            son effet via current_accepted[u] + les obs propres de u.
      2) Pour chaque requête r de u0, triée par deadline :
         - on génère un DCOP (variables x_{u,o}, coûts π(o,M_u), contrainte at-most-one),
         - on résout ce DCOP avec pyDCOP (algo DPOP par défaut),
         - on met à jour current_accepted du user gagnant.
      3) À la fin, on reconstruit les plans finaux des exclusifs avec un greedy restreint.

    Retourne :
      - final_plans : {u_id -> plan utilisateur (sat_id -> [(obs, t_start), ...])}
      - assignments_global : [(request_id, user_id, observation)]
      - avg_time_per_request : temps moyen de résolution DCOP par requête centrale.
    """
    exclusive_users: List[User] = [u for u in instance.users if u.uid != "u0"]

    # current_accepted[u] : liste des obs de u0 acceptées pour ce user
    current_accepted: Dict[str, List[Observation]] = {u.uid: [] for u in exclusive_users}

    # Requêtes du central triées par deadline
    central_requests: List[Task] = [r for r in instance.tasks if r.owner == "u0"]
    central_requests.sort(key=lambda r: r.t_end)

    assignments_global: List[Tuple[str, str, Observation]] = []
    times: List[float] = []

    for r in central_requests:
        yaml_path = f"{base_yaml_name}_{r.tid}.yaml"

        ok = generate_sdcop_yaml_for_request(instance, r, current_accepted, yaml_path)
        if not ok:
            # aucun candidat pour cette requête
            continue

        t0 = time.time()
        output = run_pydcop_solve(yaml_path, algo=algo)
        t1 = time.time()
        times.append(t1 - t0)

        if output is None:
            continue

        assignment = parse_assignment_from_output(output)
        if not assignment:
            continue

        winner_u: Optional[str] = None
        chosen_obs: Optional[Observation] = None

        # Cherche x_{u,o} = 1
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
            current_accepted[winner_u].append(chosen_obs)
            assignments_global.append((r.tid, winner_u, chosen_obs))

        # suppression du YAML temporaire
        if os.path.exists(yaml_path):
            os.remove(yaml_path)

    # Plans finaux pour les exclusifs à partir de accepted + obs propres
    final_plans: Dict[str, Dict[str, List[Tuple[Observation, int]]]] = {}
    for u in exclusive_users:
        u_id = u.uid
        final_plans[u_id] = build_restricted_plan_for_user(
            instance, u_id, extra_obs=None, accepted_u0_obs=current_accepted[u_id]
        )

    avg_time = sum(times) / len(times) if times else 0.0
    return final_plans, assignments_global, avg_time
