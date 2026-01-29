import os
import time
from typing import Dict, List, Tuple, Optional
import yaml

from DCOP import run_pydcop_solve, parse_assignment_from_output
from ESOPInstance import ESOPInstance, Observation, Task, User
from GreedySolver import greedy_schedule_P_u as greedy_schedule_for_user

# =============================================================================
# Helpers : plan local et coût π(o, M_u)
# =============================================================================

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
    
    Retourne un plan {sat_id: [(obs, t_start), ...]} pour user_id.
    """
    u = next(user for user in instance.users if user.uid == user_id)
    
    base_obs = [
        o for o in instance.observations
        if (o.owner == user_id) or (o in accepted_u0_obs)
    ]
    
    if extra_obs is not None:
        base_obs = base_obs + [extra_obs]
    
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
    
    Retourne None si obs n'apporte pas de gain.
    """
    plan_before = build_restricted_plan_for_user(
        instance, user_id, extra_obs=None, accepted_u0_obs=accepted_u0_obs_for_u
    )
    reward_before = compute_reward_from_plan(plan_before)
    
    plan_after = build_restricted_plan_for_user(
        instance, user_id, extra_obs=obs, accepted_u0_obs=accepted_u0_obs_for_u
    )
    reward_after = compute_reward_from_plan(plan_after)
    
    if reward_after <= reward_before:
        return None
    
    gain = reward_after - reward_before
    return -gain  # DCOP minimise


# =============================================================================
# Génération du DCOP pour une requête (s-dcop) - VERSION COMPLÈTE
# =============================================================================

def generate_sdcop_yaml_for_request(
    instance: ESOPInstance,
    request: Task,
    current_accepted: Dict[str, List[Observation]],
    current_plans: Dict[str, Dict[str, List[Tuple[Observation, int]]]],
    yaml_path: str,
) -> bool:
    """
    Génère un DCOP YAML pour une requête centrale 'request' (r), CONFORME À L'ARTICLE.
    
    Contraintes (équations 15-18 de l'article) :
    - (15) au plus une obs par requête : sum x_{u,o} <= 1
    - (16) capacité satellite : sum x_{u,o} <= κ*_s (capacité restante)
    - Transitions : simplifiées via pénalités (pas de variables temporelles continues)
    
    current_accepted : {u_id -> liste des obs de u0 déjà acceptées}
    current_plans : {u_id -> {sat_id -> [(obs, t_start)]}} plans actuels
    
    Retourne False si aucun candidat n'existe.
    """
    exclusive_users: List[User] = [u for u in instance.users if u.uid != "u0"]
    obs_r: List[Observation] = [o for o in request.opportunities if o.owner == "u0"]
    
    variables = {}
    constraints = {}
    agents: List[str] = []
    var_names: List[str] = []
    vars_by_sat: Dict[str, List[Tuple[str, Observation]]] = {}  # sat_id -> [(var_name, obs)]
    
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
            
            # Calcul de π(o, M_u)
            pi_val = compute_pi(instance, u_id, o, accepted_for_u)
            if pi_val is None:
                continue
            
            v_name = f"x_{u_id}_{o.oid}"
            variables[v_name] = {"domain": "binary", "agent": u_id}
            var_names.append(v_name)
            vars_by_sat.setdefault(o.satellite, []).append((v_name, o))
            
            # Coût local : π(o, M_u) * x_{u,o}
            c_name = f"c_pi_{u_id}_{o.oid}"
            constraints[c_name] = {
                "type": "intention",
                "function": f"{pi_val} * {v_name}",
            }
    
    if not var_names:
        return False
    
    # Contrainte (15) : au plus une obs par requête
    total_expr = " + ".join(var_names) if len(var_names) > 1 else var_names[0]
    constraints[f"c_atmost1_{request.tid}"] = {
        "type": "intention",
        "function": f"0 if {total_expr} <= 1 else 1e9",
    }
    
    # Contrainte (16) : capacité restante par satellite
    sats_by_id = {s.sid: s for s in instance.satellites}
    for sat_id, vars_on_sat in vars_by_sat.items():
        sat = sats_by_id[sat_id]
        
        # Capacité déjà utilisée sur ce satellite (tous users confondus)
        used_capacity = sum(
            len(sat_plan)
            for user_plan in current_plans.values()
            for sid, sat_plan in user_plan.items()
            if sid == sat_id
        )
        
        remaining_capacity = sat.capacity - used_capacity
        
        if remaining_capacity <= 0:
            # Satellite saturé, on pénalise toute allocation
            for v_name, _ in vars_on_sat:
                constraints[f"c_cap_full_{v_name}"] = {
                    "type": "intention",
                    "function": f"1e9 * {v_name}",
                }
        else:
            # Contrainte : sum des nouvelles obs sur ce sat <= capacité restante
            sat_vars = [v for v, _ in vars_on_sat]
            if len(sat_vars) > 1:
                sat_expr = " + ".join(sat_vars)
                constraints[f"c_cap_{sat_id}_{request.tid}"] = {
                    "type": "intention",
                    "function": f"0 if {sat_expr} <= {remaining_capacity} else 1e9",
                }
    
    # Contraintes de transition (simplifiées) : pénalité si obs potentiellement conflictuelles
    for v_name, obs_candidate in [(v, o) for sat_vars in vars_by_sat.values() for v, o in sat_vars]:
        sat_id = obs_candidate.satellite
        tau = sats_by_id[sat_id].transition_time
        
        # Obs déjà planifiées sur ce satellite
        existing_obs_on_sat = [
            (obs, t_start)
            for user_plan in current_plans.values()
            for sid, sat_plan in user_plan.items()
            if sid == sat_id
            for obs, t_start in sat_plan
        ]
        
        conflict_penalty = 0
        for existing_obs, existing_t in existing_obs_on_sat:
            existing_end = existing_t + existing_obs.duration
            
            candidate_earliest = obs_candidate.t_start
            candidate_latest = obs_candidate.t_end - obs_candidate.duration
            
            # Si la fenêtre de obs_candidate chevauche [existing_t - tau, existing_end + tau]
            if not (candidate_latest + obs_candidate.duration + tau <= existing_t or 
                    candidate_earliest >= existing_end + tau):
                conflict_penalty += 100  # pénalité modérée
        
        if conflict_penalty > 0:
            constraints[f"c_transition_{v_name}"] = {
                "type": "intention",
                "function": f"{conflict_penalty} * {v_name}",
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


# =============================================================================
# Implémentation séquentielle s-dcop (Algorithm 5) - VERSION CONFORME
# =============================================================================

# =============================================================================
# Implémentation séquentielle s-dcop (Algorithm 5) - VERSION CONFORME
# =============================================================================

def sdcop_with_pydcop(
    instance: ESOPInstance,
    algo: str = "dpop",
    base_yaml_name: str = "sdcop_req",
):
    """
    Implémentation de s-dcop CONFORME À L'ARTICLE (Algorithm 5, Section 5.2).
    
    Retourne :
    - final_plans : {u_id -> {sat_id -> [(obs, t_start)]}}
    - assignments_global : [(request_id, user_id, observation)]
    - avg_time_per_request : temps moyen de résolution DCOP
    """
    from GreedySolver import greedy_schedule  # Import du greedy centralisé
    
    exclusive_users: List[User] = [u for u in instance.users if u.uid != "u0"]
    
    # État global
    current_accepted: Dict[str, List[Observation]] = {u.uid: [] for u in exclusive_users}
    current_plans: Dict[str, Dict[str, List[Tuple[Observation, int]]]] = {}
    
    # Étape 1 : Résolution locale P_u pour chaque user exclusif
    for u in exclusive_users:
        current_plans[u.uid] = build_restricted_plan_for_user(
            instance, u.uid, extra_obs=None, accepted_u0_obs=[]
        )
    
    # Requêtes du central triées par deadline
    central_requests: List[Task] = [r for r in instance.tasks if r.owner == "u0"]
    central_requests.sort(key=lambda r: r.t_end)
    
    assignments_global: List[Tuple[str, str, Observation]] = []
    times: List[float] = []
    allocated_obs_ids: set = set()
    
    # Étape 2 : Pour chaque requête r de u0, résoudre le DCOP
    for r in central_requests:
        yaml_path = f"{base_yaml_name}_{r.tid}.yaml"
        
        ok = generate_sdcop_yaml_for_request(
            instance, r, current_accepted, current_plans, yaml_path
        )
        
        if not ok:
            continue
        
        t0 = time.time()
        output = run_pydcop_solve(yaml_path, algo=algo)
        t1 = time.time()
        times.append(t1 - t0)
        
        if output is None:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)
            continue
        
        assignment = parse_assignment_from_output(output)
        if not assignment:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)
            continue
        
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
            break
        
        if winner_u is not None and chosen_obs is not None:
            current_accepted[winner_u].append(chosen_obs)
            assignments_global.append((r.tid, winner_u, chosen_obs))
            allocated_obs_ids.add(chosen_obs.oid)
            
            # Mise à jour du plan de winner_u
            current_plans[winner_u] = build_restricted_plan_for_user(
                instance, winner_u, extra_obs=None, 
                accepted_u0_obs=current_accepted[winner_u]
            )
        
        if os.path.exists(yaml_path):
            os.remove(yaml_path)
    
    # ═════════════════════════════════════════════════════════════════════════
    # Étape 3 : Plans finaux - CONFORME À L'ARTICLE
    # ═════════════════════════════════════════════════════════════════════════
    
    # Les plans des exclusifs sont déjà calculés
    final_plans: Dict[str, Dict[str, List[Tuple[Observation, int]]]] = {}
    for u in exclusive_users:
        final_plans[u.uid] = current_plans[u.uid]
    
    # Pour u0 : créer une sous-instance SANS les obs allouées aux exclusifs
    # et résoudre avec greedy_schedule centralisé (qui respecte toutes les contraintes)
    
    u0_observations = [
        o for o in instance.observations 
        if o.owner == "u0" and o.oid not in allocated_obs_ids
    ]
    
    # Observations des exclusifs (déjà planifiées, à garder)
    exclusive_observations = [
        o for o in instance.observations
        if o.owner != "u0"
    ]
    
    # Sous-instance : u0 + exclusifs (avec leurs obs), mais u0 n'a que ses obs restantes
    inst_final = ESOPInstance(
        nb_satellites=instance.nb_satellites,
        nb_users=instance.nb_users,
        nb_tasks=instance.nb_tasks,
        horizon=instance.horizon,
        satellites=instance.satellites,
        users=instance.users,
        tasks=instance.tasks,
        observations=u0_observations + exclusive_observations,  # Obs restantes de u0 + toutes obs exclusifs
    )
    
    # Greedy centralisé sur toute l'instance (respecte les contraintes globales)
    all_plans = greedy_schedule(inst_final)
    
    # On ne garde que le plan de u0 (les exclusifs gardent leurs plans DCOP)
    final_plans["u0"] = all_plans.get("u0", {})
    
    avg_time = sum(times) / len(times) if times else 0.0
    return final_plans, assignments_global, avg_time
