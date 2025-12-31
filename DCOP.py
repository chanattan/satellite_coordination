import subprocess
import re
from InstanceGenerator import generate_DCOP_instance
from ESOPInstance import ESOPInstance
import time

def save_dcop_instance(dcop):
    """
    Sauvegarde l'instance DCOP au format YAML dans un fichier "esop_dcop.yaml".
    """
    with open("esop_dcop.yaml", "w") as f:
        f.write(dcop)

def run_pydcop_solve(yaml_path: str, algo: str = "dpop") -> str:
    """
    Lance 'pydcop solve --algo {algo} {yaml_path}' et renvoie stdout sous forme de string.
    """
    cmd = ["pydcop", "solve", "--algo", algo, yaml_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de pydcop: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Erreur: pydcop n'est pas installé ou n'est pas dans le PATH")
        print("Installez-le avec: pip install pydcop")
        return None

import json
import re
from typing import Dict


def parse_assignment_from_output(output: str) -> Dict[str, int]:
    """
    Parse l'assignement des variables depuis la sortie JSON de pydcop solve.
    
    La sortie de pyDCOP est capturée au format JSON pour réafficher proprement la section assignment.
    """
    if output is None or output.strip() == "":
        return {}
    
    assignment = {}
    
    try:
        data = json.loads(output)
        
        # extraction de la section assignment
        if "assignment" in data:
            assignment = data["assignment"]
            assignment = {k: int(v) for k, v in assignment.items()}
        
        return assignment
    
    except json.JSONDecodeError:
        print("Parsing JSON échoué -> parsing manuel")
        
        pattern = re.compile(r'^\s*"?([A-Za-z0-9_]+)"?\s*:\s*([0-9]+)')
        
        in_assign_section = False
        for line in output.splitlines():
            if "assignment" in line.lower():
                in_assign_section = True
                continue
            
            if not in_assign_section:
                continue
            if line.strip().startswith("}"):
                break
            
            m = pattern.match(line)
            if m:
                var_name, val = m.group(1), int(m.group(2))
                assignment[var_name] = val
        
        return assignment


def print_assignment_summary(instance, assignment: Dict[str, int]):
    """
    Affiche un résumé clair de l'assignement du DCOP.
    """
    print("="*70)
    print("RÉSUMÉ DES RÉSULTATS DU DCOP")
    print("="*70 + "\n")
    
    # map observation_id -> observation
    obs_by_id = {o.oid: o for o in instance.observations}
    
    # Grouper par utilisateur exclusif
    user_allocations = {}
    total_reward = 0
    
    for var_name, value in assignment.items():
        if value != 1:
            continue
        
        # Parser : x_{u_id}_{o_id}
        try:
            _, u_id, o_id = var_name.split("_", 2)
        except ValueError:
            continue
        
        if o_id not in obs_by_id:
            continue
        
        obs = obs_by_id[o_id]
        
        if u_id not in user_allocations:
            user_allocations[u_id] = []
        
        user_allocations[u_id].append(obs)
        total_reward += obs.reward
    
    if not user_allocations:
        print("  Aucune observation allouée aux utilisateurs exclusifs.\n")
    else:
        print(f"Observations allouées aux utilisateurs exclusifs:")
        print(f"{'-'*70}\n")
        
        for u_id in sorted(user_allocations.keys()):
            observations = user_allocations[u_id]
            user_reward = sum(o.reward for o in observations)
            
            print(f"  {u_id}:")
            print(f"    Nombre d'observations : {len(observations)}")
            print(f"    Récompense totale : {user_reward}")
            print(f"    Observations : ")
            
            for obs in observations:
                print(f"      - {obs.oid} (task={obs.task_id}, reward={obs.reward}, sat={obs.satellite})")
            print()
    
    print(f"TOTAL : {sum(len(obs_list) for obs_list in user_allocations.values())} observations allouées")
    print(f"RÉCOMPENSE TOTALE : {total_reward}\n")
    print("="*70 + "\n")
    
    return user_allocations, total_reward


def print_dcop_metrics(output: str):
    """
    Affiche les métriques de résolution du DCOP.
    """
    try:
        data = json.loads(output)
        
        print("="*70)
        print("MÉTRIQUES DE RÉSOLUTION DCOP")
        print("="*70 + "\n")
        
        print(f"  Statut: {data.get('status', 'UNKNOWN')}")
        print(f"  Coût final: {data.get('cost', 'N/A')}")
        print(f"  Violation: {data.get('violation', 'N/A')}")
        print(f"  Cycles: {data.get('cycle', 'N/A')}")
        print(f"  Temps de calcul: {data.get('time', 'N/A')} secondes")
        print(f"  Nombre de messages: {data.get('msg_count', 'N/A')}")
        print(f"  Taille des messages: {data.get('msg_size', 'N/A')} bytes\n")
        
        print("="*70 + "\n")
    except:
        pass

def assignment_to_user_plans(instance: ESOPInstance, assignment: dict):
    """
    Convertit un assignement DCOP en plannings utilisateurs.
    """
    user_plans = {}
    for var_name, value in assignment.items():
        if value != 1:
            continue
        # variable x_{u_id}_{o_id}
        try:
            _, u_id, o_id = var_name.split("_", 2)
        except ValueError:
            # variable qui ne respecte pas ce schéma
            continue

        # retrouver l'observation
        try:
            obs = next(o for o in instance.observations if o.oid == o_id)
        except StopIteration:
            continue

        user_plans.setdefault(u_id, {}).setdefault(obs.satellite, []).append((obs, None))

    return user_plans

def validate_dcop_functions(dcop_yaml: str) -> bool:
    """
    Tests pour valider que toutes les fonctions du DCOP retournent bien des valeurs.
    """
    print("Validation des fonctions de contraintes...")
    
    import re
    
    # extraction de toutes les fonctions
    func_pattern = re.compile(r'function: ["\'](.+?)["\']', re.DOTALL)
    functions = func_pattern.findall(dcop_yaml)
    
    errors = []
    for i, func_str in enumerate(functions):
        # check qu'il y a au moins un return
        if 'return' not in func_str:
            errors.append(f"Fonction {i+1}: Aucune instruction return trouvée")
        
        # check qu'il n'y a pas de chemins sans return
        lines = func_str.split('\\n')
        if_count = sum(1 for line in lines if 'if ' in line or 'elif ' in line)
        return_count = sum(1 for line in lines if 'return' in line)
        
        if if_count > 0 and return_count < if_count + 1:
            errors.append(f"Fonction {i+1}: Chemins possibles sans return")
    
    if errors:
        print("Erreurs :")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Toutes les fonctions sont valides")
    return True

def solve_dcop(inst, print_output=True):
    """
    Résout l'instance ESOP en la transformant en instance DCOP puis en utilisant l'algorithme DPOP avec PyDcop.
    """
    print("\n=== Résolution DCOP avec DPOP ===\n")
    
    yaml_path = "esop_dcop.yaml"
    
    # Générer le DCOP à partir de l'instance ESOP
    if print_output:
        print("> Génération du DCOP...")
    dcop_yaml = generate_DCOP_instance(inst)
    
    # Vérification pour valider le format yaml, dont les fonctions
    validate_dcop_functions(dcop_yaml)
    
    # Sauvegarde du DCOP en yaml pour exécution manuelle, ici on fera appel à la commande dans Python
    save_dcop_instance(dcop_yaml)
    if print_output:
        print(f"> DCOP sauvegardé dans {yaml_path}\n")
    
    lines = dcop_yaml.split('\n')
    nb_vars = sum(1 for line in lines if line.strip().startswith('x_'))
    nb_constraints = sum(1 for line in lines if line.strip().startswith('c_'))
    if print_output:
        print(f"> Informations du DCOP:")
    print(f"  - Nombre de variables: {nb_vars}")
    print(f"  - Nombre de contraintes: {nb_constraints}\n")

    # On résout avec PyDcop et on capture la sortie
    if print_output:
        print("Lancement de DPOP...")
    time_start = time.time()
    output = run_pydcop_solve(yaml_path, algo="dpop")
    time_end = time.time()
    print(f"Temps de résolution DPOP : {time_end - time_start:.4f} secondes\n")
    
    if output is None:
        print("\n!!! Échec de la résolution DCOP.")
        return
    
    if print_output:
        print("> Sortie de DPOP:")
        print("-" * 50)
        print(output)
        print("-" * 50)

    # On parse le résultat pour un affichage plus clair.
    if print_output:
        print("\n> Parsing de la solution...")
    assignment = parse_assignment_from_output(output)
    
    if not assignment:
        print("!!! Aucun résultat trouvé dans la sortie de DPOP.")
        return
    
    # Résultats
    if print_output:
        print_dcop_metrics(output)
    print_assignment_summary(inst, assignment)
    print("=== Fin de la résolution DCOP ===\n")

    # On recrée l'user plan
    assignment = assignment_to_user_plans(inst, assignment)
    return assignment

from typing import Dict, List, Tuple, Optional
from GreedySolver import greedy_schedule_for_user
from ESOPInstance import Observation, Task, ESOPInstance, User
import yaml

def build_restricted_plan_for_user(instance: ESOPInstance,
                                   user_id: str,
                                   extra_obs: Observation,
                                   accepted_u0_obs):
    """
    Construit un plan glouton pour user_id en ne considérant que :
      - ses propres observations,
      - + accepted_u0_obs (obs de u0 déjà acceptées),
      - + extra_obs si non None.
    Retourne un plan {sat_id: [(obs, t_start), ...]}.
    """

    user = next(u for u in instance.users if u.uid == user_id)
    sats = instance.satellites

    # Filtrer les obs pertinentes
    base_obs = [
        o for o in instance.observations
        if (o.owner == user_id) or (o in accepted_u0_obs)
    ]
    if extra_obs is not None:
        base_obs = base_obs + [extra_obs]

    plan = {s.sid: [] for s in sats}

    for sat in sats:
        sat_id = sat.sid
        # Observations de cette liste sur ce satellite
        candidate_obs = [o for o in base_obs if o.satellite == sat_id]

        # Glouton restreint : même logique que greedy_schedule_for_user_on_satellite
        candidate_obs.sort(key=lambda o: (-o.reward, o.t_start))
        local_plan = []

        def used_capacity():
            return len(local_plan)

        def try_insert(obs: Observation):
            if used_capacity() >= sat.capacity:
                return None
            if not local_plan:
                t0 = max(obs.t_start, sat.t_start)
                if t0 + obs.duration <= min(obs.t_end, sat.t_end):
                    return t0
                return None
            sorted_plan = sorted(local_plan, key=lambda p: p[1])
            tau = sat.transition_time

            # avant la première
            first_obs, first_t = sorted_plan[0]
            earliest_start = max(obs.t_start, sat.t_start)
            latest_end = min(obs.t_end, first_t - tau)
            if earliest_start + obs.duration <= latest_end:
                return earliest_start

            # entre deux
            for (o_prev, t_prev), (o_next, t_next) in zip(sorted_plan, sorted_plan[1:]):
                end_prev = t_prev + o_prev.duration
                start_next = t_next
                window_start = max(obs.t_start, end_prev + tau, sat.t_start)
                window_end = min(obs.t_end, start_next - tau, sat.t_end)
                if window_start + obs.duration <= window_end:
                    return window_start

            # après la dernière
            last_obs, last_t = sorted_plan[-1]
            end_last = last_t + last_obs.duration
            window_start = max(obs.t_start, end_last + tau, sat.t_start)
            window_end = min(obs.t_end, sat.t_end)
            if window_start + obs.duration <= window_end:
                return window_start
            return None

        for obs in candidate_obs:
            t_ins = try_insert(obs)
            if t_ins is not None:
                local_plan.append((obs, t_ins))

        local_plan.sort(key=lambda p: p[1])
        plan[sat_id] = local_plan

    return plan


def compute_reward_from_plan(plan_for_user: dict) -> int:
    return sum(obs.reward for sat_plan in plan_for_user.values() for (obs, _) in sat_plan)


def compute_pi(instance: ESOPInstance,
               user_id: str,
               obs: Observation,
               current_accepted_u0_obs):
    """
    π(o, M_u) = coût marginal pour user_id si il accepte obs de u0.
    M_u est implicite via current_accepted_u0_obs (les requêtes déjà acceptées).
    """

    # Plan avant : propres obs + accepted_u0_obs
    plan_before = build_restricted_plan_for_user(instance, user_id, None, current_accepted_u0_obs)
    reward_before = compute_reward_from_plan(plan_before)

    # Plan après : mêmes + obs
    plan_after = build_restricted_plan_for_user(instance, user_id, obs, current_accepted_u0_obs)
    reward_after = compute_reward_from_plan(plan_after)

    if reward_after <= reward_before:
        return None

    gain = reward_after - reward_before
    return -gain  # DPOP minimise

def generate_sdcop_yaml_for_request(instance, request, current_accepted, yaml_path):
    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    obs_r = [o for o in request.opportunities if o.owner == "u0"]

    variables = {}
    constraints = {}
    agents = []
    var_names = []

    for u in exclusive_users:
        u_id = u.uid
        agents.append(u_id)
        accepted_for_u = current_accepted.get(u_id, [])

        for o in obs_r:
            can_take = any(
                w.satellite == o.satellite and
                not (w.t_end <= o.t_start or w.t_start >= o.t_end)
                for w in u.exclusive_windows
            )
            if not can_take:
                continue

            pi_val = compute_pi(instance, u_id, o, accepted_for_u)
            if pi_val is None:
                continue

            v_name = f"x_{u_id}_{o.oid}"
            variables[v_name] = {"domain": "binary", "agent": u_id}
            var_names.append(v_name)

            c_name = f"c_pi_{u_id}_{o.oid}"
            constraints[c_name] = {
                "type": "intention",
                "function": f"{pi_val} * {v_name}"
            }

    if not var_names:
        return False

    total_expr = " + ".join(var_names) if len(var_names) > 1 else var_names[0]
    constraints[f"c_atmost1_{request.tid}"] = {
        "type": "intention",
        "function": f"0 if {total_expr} <= 1 else 1e9"
    }

    dcop_dict = {
        "name": f"sdcop_{request.tid}",
        "objective": "min",
        "domains": {"binary": {"values": [0, 1]}},
        "agents": agents,
        "variables": variables,
        "constraints": constraints
    }

    with open(yaml_path, "w") as f:
        yaml.dump(dcop_dict, f, sort_keys=False)

    return True


def sdcop_with_pydcop(instance: ESOPInstance,
                      algo: str = "dpop",
                      base_yaml_name: str = "sdcop_req"):

    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    current_accepted: dict[str, list[Observation]] = {u.uid: [] for u in exclusive_users}

    central_requests = [r for r in instance.tasks if r.owner == "u0"]
    central_requests.sort(key=lambda r: r.t_end)

    assignments_global = []

    for r in central_requests:
        yaml_path = f"{base_yaml_name}_{r.tid}.yaml"

        ok = generate_sdcop_yaml_for_request(instance, r, current_accepted, yaml_path)
        if not ok:
            continue

        output = run_pydcop_solve(yaml_path, algo=algo)
        if output is None:
            continue

        assignment = parse_assignment_from_output(output)
        if not assignment:
            continue

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

            current_accepted[u_id].append(obs)
            assignments_global.append((r.tid, u_id, obs))
            break

    # Construire les plans finaux des exclusifs à partir de accepted
    final_plans = {}
    for u in exclusive_users:
        u_id = u.uid
        final_plans[u_id] = build_restricted_plan_for_user(
            instance, u_id,
            extra_obs=None,
            accepted_u0_obs=current_accepted[u_id]
        )
    
    # Clean les fichiers yaml temporaires
    import os
    for r in central_requests:
        yaml_path = f"{base_yaml_name}_{r.tid}.yaml"
        if os.path.exists(yaml_path):
            os.remove(yaml_path)

    return final_plans, assignments_global


def recompute_plan_with_obs(instance, user_id):
    """
    Recalcule un plan complet pour user_id en relançant le planificateur glouton sur l'instance ESOP.
    """
    return greedy_schedule_for_user(instance, user_id)


def pi_for_observation(instance, user_id, current_plan):
    """
    Calcule pi(o, M_u) = coût marginal (négatif du gain de reward) pour user_id, en replanifiant complètement.

    Retourne None si ça n'apporte aucun gain.
    """

    reward_before = compute_reward_from_plan(current_plan)

    new_plan = recompute_plan_with_obs(instance, user_id)
    reward_after = compute_reward_from_plan(new_plan)

    if reward_after <= reward_before:
        # Aucun gain ou perte -> on ignore cette opportunité
        return None

    gain = reward_after - reward_before
    # DCOP minimise le coût -> coût = -gain
    return -gain

def solve_request_with_dcop_exact(instance, request, current_plans, exclusive_users):
    """
    Résout le DCOP pour une requête centrale 'request' (s-dcop simplifié).

    current_plans : {u_id -> {sat_id -> [(obs, t_start), ...]}}
    exclusive_users : liste des users exclusifs (uid != "u0")

    Retourne (winner_user_id, chosen_observation) ou (None, None).
    """

    # Opportunités de cette requête (theta_r)
    observations_r = [o for o in request.opportunities if o.owner == "u0"]

    best_cost = None
    best_choice: Tuple[Optional[str], Optional[Observation]] = (None, None)

    for obs in observations_r:
        for u in exclusive_users:
            u_id = u.uid

            # Vérifier qu'il existe au moins une fenêtre exclusive compatible
            can_take = any(
                w.satellite == obs.satellite and
                not (w.t_end <= obs.t_start or w.t_start >= obs.t_end)
                for w in u.exclusive_windows
            )
            if not can_take:
                continue

            # Plan courant de u
            plan_u = current_plans.get(u_id)
            if plan_u is None:
                plan_u = greedy_schedule_for_user(instance, u_id)
                current_plans[u_id] = plan_u

            cost = pi_for_observation(instance, u_id, plan_u)
            if cost is None:
                continue

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_choice = (u_id, obs)

    return best_choice


def s_dcop_solve(instance):
    """
    Implémentation simplifiée de l'algorithme s-dcop (article) en Python pur.

    - On calcule d'abord les plans locaux M_u pour tous les utilisateurs exclusifs.
    - Pour chaque requête du central (u0), on choisit (u, o) qui maximise
      le gain de reward via pi(o, M_u).
    - On met à jour M_u pour le gagnant en replanifiant.

    Retourne :
      - current_plans : {u_id -> plan utilisateur}
      - assignments : liste de tuples (request_id, user_id, observation)
    """

    # Utilisateurs exclusifs
    exclusive_users = [u for u in instance.users if u.uid != "u0"]

    # Plans initiaux M_u (problèmes P[u])
    current_plans = {}
    for u in exclusive_users:
        current_plans[u.uid] = greedy_schedule_for_user(instance, u.uid)

    # Requêtes du central (u0) triées par deadline
    central_requests = [r for r in instance.tasks if r.owner == "u0"]
    central_requests.sort(key=lambda r: r.t_end)

    assignments = []

    for r in central_requests:
        winner_u, chosen_obs = solve_request_with_dcop_exact(
            instance,
            r,
            current_plans,
            exclusive_users
        )

        if winner_u is not None and chosen_obs is not None:
            # Mettre à jour le plan du gagnant
            current_plans[winner_u] = greedy_schedule_for_user(instance, winner_u)
            assignments.append((r.tid, winner_u, chosen_obs))

    return current_plans, assignments



