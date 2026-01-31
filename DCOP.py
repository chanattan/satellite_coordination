import subprocess
import re
from InstanceGenerator import generate_DCOP_instance
import time
import json
from GreedySolver import greedy_schedule_P_u as greedy_schedule_for_user
from ESOPInstance import Observation, Task, ESOPInstance, User
import yaml

def save_dcop_instance(dcop):
    with open("esop_dcop.yaml", "w") as f:
        f.write(dcop)

def run_pydcop_solve(yaml_path, algo = "dpop", timeout = 60):
    """
        Lance le subprocess 'pydcop solve --algo {algo} {yaml_path}' avec timeout.
    """
    cmd = ["pydcop", "solve", "--algo", algo, yaml_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"Timeout PyDCOP > {timeout}s")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Erreur PyDCOP : {e}")
        return None
    except FileNotFoundError:
        print("Erreur file not found error.")
        return None

def extract_time_from_output(output):
    if output is None or output.strip() == "":
        return 0.0
    
    try:
        data = json.loads(output)
        return data.get('time', 0.0)
    except:
        return 0.0

def parse_assignment_from_output(output):
    """
        Parse l'assignement des variables depuis la sortie JSON de pydcop solve.
        La sortie de pyDCOP est capturée au format JSON pour réafficher proprement la section assignment.
    """
    if output is None or output.strip() == "":
        return {}
    
    assignment = {}
    
    try:
        data = json.loads(output)
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

def extract_metrics_from_output(output):
    if output is None or output.strip() == "":
        return 0, 0
    
    try:
        data = json.loads(output)
        nb_messages = data.get('msg_count', 0)
        comm_load = data.get('msg_size', 0)
        return nb_messages, comm_load
    except:
        return 0, 0


def print_assignment_summary(instance, assignment):
    print("---------------")
    print("Résultats DCOP")
    
    obs_by_id = {o.oid: o for o in instance.observations}
    
    # Grouper par utilisateur exclusif
    user_allocations = {}
    total_reward = 0
    
    for var_name, value in assignment.items():
        if value != 1:
            continue
        
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
        print("> Aucune observation allouée aux utilisateurs exclusifs.\n")
    else:
        print(f"Observations allouées aux utilisateurs exclusifs:")
        print("-------------------------")
        
        for u_id in sorted(user_allocations.keys()):
            observations = user_allocations[u_id]
            user_reward = sum(o.reward for o in observations)
            
            print(f"> {u_id}:")
            print(f"-> Nombre d'observations : {len(observations)}")
            print(f"-> Récompense totale : {user_reward}")
            print(f"-> Observations : ")
            
            for obs in observations:
                print(f"    -> {obs.oid} (task={obs.task_id}, reward={obs.reward}, sat={obs.satellite})")
            print()
    
    print(f"Total : {sum(len(obs_list) for obs_list in user_allocations.values())} ; observations allouées")
    print(f"Récompense totale : {total_reward}\n")
    
    return user_allocations, total_reward

def print_dcop_metrics(output):
    try:
        data = json.loads(output)
        
        print("Métriques de résolution DCOP")
        print(f"> Statut: {data.get('status', 'UNKNOWN')}")
        print(f"> Coût final: {data.get('cost', 'N/A')}")
        print(f"> Violation: {data.get('violation', 'N/A')}")
        print(f"> Cycles: {data.get('cycle', 'N/A')}")
        print(f"> Temps de calcul: {data.get('time', 'N/A')} secondes")
        print(f"> Nombre de messages: {data.get('msg_count', 'N/A')}")
        print(f"> Taille des messages: {data.get('msg_size', 'N/A')} bytes\n")
    except:
        pass

def assignment_to_user_plans(instance, assignment):
    """
        Conversion d'un assignement DCOP en plannings utilisateurs.
    """
    user_plans = {}
    for var_name, value in assignment.items():
        if value != 1:
            continue
        try:
            _, u_id, o_id = var_name.split("_", 2)
        except ValueError: # problème parsing ici sinon...
            continue

        try:
            obs = next(o for o in instance.observations if o.oid == o_id)
        except StopIteration:
            continue

        user_plans.setdefault(u_id, {}).setdefault(obs.satellite, []).append((obs, None))
    return user_plans


def validate_dcop_functions(dcop_yaml):
    """
        Validation du YAML associé au DCOP.
    """
    print("Validation des fonctions de contraintes...")
    
    func_pattern = re.compile(r'function: ["\'](.+?)["\']', re.DOTALL)
    functions = func_pattern.findall(dcop_yaml)
    
    errors = []
    for i, func_str in enumerate(functions):
        if 'return' not in func_str: # check qu'il y a au moins un return
            errors.append(f"Fonction {i+1}: Aucune instruction return trouvée")
        
        lines = func_str.split('\\n') # check qu'il n'y a pas de chemins sans return
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
    
    if print_output:
        print("> Génération du DCOP...")
    dcop_yaml = generate_DCOP_instance(inst)
    
    validate_dcop_functions(dcop_yaml)
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
        print("-------")
        print(output)

    if print_output:
        print("\n> Parsing de la solution...")
    assignment = parse_assignment_from_output(output)
    
    if not assignment:
        print("!!! Aucun résultat trouvé dans la sortie de DPOP.")
        return
    
    if print_output:
        print_dcop_metrics(output)
    print_assignment_summary(inst, assignment)
    print("=== Fin de la résolution DCOP ===\n")

    assignment = assignment_to_user_plans(inst, assignment)
    return assignment

def build_restricted_plan_for_user(instance, user_id, extra_obs, accepted_u0_obs):
    """
        Construit un plan glouton pour user_id en ne considérant que :
            - ses propres observations,
            - + accepted_u0_obs (obs de u0 déjà acceptées),
            - + extra_obs si non None.
        retourne le plan par satellite.
    """
    user = next(u for u in instance.users if u.uid == user_id)
    sats = instance.satellites

    base_obs = [o for o in instance.observations if (o.owner == user_id) or (o in accepted_u0_obs)]
    if extra_obs is not None:
        base_obs = base_obs + [extra_obs]

    plan = {s.sid: [] for s in sats}
    for sat in sats:
        sat_id = sat.sid
        # observations de cette liste sur ce satellite
        candidate_obs = [o for o in base_obs if o.satellite == sat_id]

        candidate_obs.sort(key=lambda o: (-o.reward, o.t_start)) # glouton reward puis earliest
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

            first_obs, first_t = sorted_plan[0] # avant la première
            earliest_start = max(obs.t_start, sat.t_start)
            latest_end = min(obs.t_end, first_t - tau)
            if earliest_start + obs.duration <= latest_end:
                return earliest_start

            for (o_prev, t_prev), (o_next, t_next) in zip(sorted_plan, sorted_plan[1:]): # entre deux
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

        # essayer d'insérer chaque observation candidate
        for obs in candidate_obs:
            t_ins = try_insert(obs)
            if t_ins is not None:
                local_plan.append((obs, t_ins))

        local_plan.sort(key=lambda p: p[1])
        plan[sat_id] = local_plan

    return plan

def compute_reward_from_plan(plan_for_user):
    return sum(obs.reward for sat_plan in plan_for_user.values() for (obs, _) in sat_plan)

def compute_pi(instance, user_id, obs, current_accepted_u0_obs):
    """
        pi(o, M_u) de l'article = coût marginal pour user_id s'il accepte obs de u0.
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

pi_cache = {}

def clear_pi_cache():
    global pi_cache
    pi_cache = {}

def __generate_sdcop_yaml_for_request(instance, request, current_accepted, yaml_path): #outdated
    """
        Génère le fichier YAML pour le s-dcop d'une requête du central.
    """
    global pi_cache
    
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

            # cache opti. key avec IDs d'observations
            accepted_ids = tuple(sorted([obs.oid for obs in accepted_for_u]))
            cache_key = (u_id, o.oid, accepted_ids)

            if cache_key in pi_cache:
                pi_val = pi_cache[cache_key]
            else:
                pi_val = compute_pi(instance, u_id, o, accepted_for_u)
                pi_cache[cache_key] = pi_val

            if pi_val is None:
                continue

            v_name = f"x_{u_id}_{o.oid}"
            variables[v_name] = {"domain": "binary"}
            var_names.append(v_name)

            c_name = f"c_pi_{u_id}_{o.oid}"
            constraints[c_name] = {
                "type": "intention",
                "function": f"{pi_val} * {v_name}"
            }

    if not var_names:
        return False

    # Contrainte au plus 1
    total_expr = " + ".join(var_names) if len(var_names) > 1 else var_names[0]
    constraints[f"c_atmost1_{request.tid}"] = {
        "type": "intention",
        "function": f"0 if {total_expr} <= 1 else 1e9"
    }
    
    # agents auxiliaires pour dcop
    nb_computations = len(variables) + len(constraints)
    agents_list = list(set(agents)) # remove duplicate
    
    while len(agents_list) < nb_computations:
        agents_list.append(f"aux_{len(agents_list)}")
    
    agents_with_capacity = {agent_id: {"capacity": 1000} for agent_id in agents_list}

    dcop_dict = {
        "name": f"sdcop_{request.tid}",
        "objective": "min",
        "domains": {"binary": {"values": [0, 1]}},
        "agents": agents_with_capacity,
        "variables": variables,
        "constraints": constraints
    }

    with open(yaml_path, "w") as f:
        yaml.dump(dcop_dict, f, sort_keys=False)

    return True


def recompute_plan_with_obs(instance, user_id):
    """
        Recalcule un plan complet pour user_id en relançant le planificateur glouton sur l'instance ESOP.
    """
    return greedy_schedule_for_user(instance, user_id)


def pi_for_observation(instance, user_id, current_plan):
    """
    Calcule pi(o, M_u) = coût marginal pour user_id avec replanif. complète.
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
        Résout le DCOP pour une requête centrale.
        Retourne (winner_user_id, chosen_observation) ou (None, None).
    """

    # Opportunités de cette requête
    observations_r = [o for o in request.opportunities if o.owner == "u0"]

    best_cost = None
    best_choice = (None, None)
    for obs in observations_r:
        for u in exclusive_users:
            u_id = u.uid

            # Vérifier qu'il existe au moins une fenêtre exclusive compatible
            can_take = any(w.satellite == obs.satellite and not (w.t_end <= obs.t_start or w.t_start >= obs.t_end) for w in u.exclusive_windows)
            if not can_take:
                continue

            plan_u = current_plans.get(u_id) # plan courant de u
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

def __s_dcop_solve(instance): # outdated
    """
        Implémentation de l'algorithme s-dcop (article).

        1. On calcule d'abord les plans locaux M_u pour tous les utilisateurs exclusifs.
        2. Pour chaque requête du central (u0), on choisit (u, o) qui maximise le gain de reward via pi(o, M_u).
        3. On met à jour M_u pour le gagnant en replanifiant.
    """
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
        winner_u, chosen_obs = solve_request_with_dcop_exact(instance, r, current_plans, exclusive_users)
        if winner_u is not None and chosen_obs is not None:
            # mise à jour du plan du gagnant
            current_plans[winner_u] = greedy_schedule_for_user(instance, winner_u)
            assignments.append((r.tid, winner_u, chosen_obs))

    return current_plans, assignments

def test_pydcop_output():
    print("test minimal : sortie PyDCOP")

    dcop_yaml = """
        name: test_minimal
        objective: min
        domains:
        binary:
            values: [0, 1]
        agents: [u1, u2]
        variables:
        x1:
            domain: binary
            agent: u1
        x2:
            domain: binary
            agent: u2
        constraints:
        c1:
            type: intention
            function: x1 + x2
        """
    
    yaml_path = "test_dcop_minimal.yaml"
    with open(yaml_path, "w") as f:
        f.write(dcop_yaml)
    
    print(f"> DCOP minimal sauvegardé dans {yaml_path}")
    print("> Lancement de DPOP...\n")
    
    import subprocess
    cmd = ["pydcop", "solve", "--algo", "dpop", yaml_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
        output = result.stdout
        #output = run_pydcop_solve(yaml_path, algo="dpop", timeout=20)
        
        if output:
            print("sortie pydcop :")
            print(output)
            
            try:
                import json
                data = json.loads(output)
                print("\nParsing JSON réussi !")
                print(f"- msg_count: {data.get('msg_count', 'NON TROUVÉ')}")
                print(f"- msg_size: {data.get('msg_size', 'NON TROUVÉ')}")
                print(f"- time: {data.get('time', 'NON TROUVÉ')}")
                print(f"- cost: {data.get('cost', 'NON TROUVÉ')}")
                print(f"- status: {data.get('status', 'NON TROUVÉ')}")
                
                print("\n> Clés JSON :")
                for key in data.keys():
                    print(f"  - {key}: {data[key]}")
                    
            except json.JSONDecodeError as e:
                print(f"Erreur parsing JSON : {e}")
        else:
            print("Pas de sortie de PyDCOP")
            print(output.stderr)
                
    except subprocess.TimeoutExpired:
        print("TIMEOUT") # instance trop complexe ?
    except Exception as e:
        print(f"Erreur : {e}")
    
    import os
    if os.path.exists(yaml_path):
        os.remove(yaml_path)

if __name__ == "__main__":
    test_pydcop_output()
