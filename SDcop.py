import os
import yaml
from ESOPInstance import ESOPInstance, Observation, Task
from GreedySolver import greedy_schedule, greedy_schedule_P_u

def build_restricted_plan_for_user(instance, user_id, extra_obs, accepted_u0_obs):
    """
    Construit un plan glouton pour user_id en respectant STRICTEMENT les fenêtres exclusives.
    """
    u = next(user for user in instance.users if user.uid == user_id)
    base_obs = [o for o in instance.observations if (o.owner == user_id) or (o in accepted_u0_obs)]
    
    if extra_obs is not None:
        base_obs = base_obs + [extra_obs]

    sats_by_id = {s.sid: s for s in instance.satellites}
    plans = {s.sid: [] for s in instance.satellites}
    
    # TRI PAR REWARD DÉCROISSANT (GREEDY)
    base_obs_sorted = sorted(base_obs, key=lambda o: (-o.reward, o.t_start))
    
    for obs in base_obs_sorted:
        sat = sats_by_id[obs.satellite]
        current_plan = plans[obs.satellite]
        
        if len(current_plan) >= sat.capacity:
            continue
        
        # check fenêtres exclusives
        exclusives_on_sat = [w for w in u.exclusive_windows if w.satellite == sat.sid]
        
        if exclusives_on_sat:
            # L'obs DOIT être dans AU MOINS UNE exclusive
            in_any_exclusive = any(obs.t_start >= w.t_start and obs.t_end <= w.t_end for w in exclusives_on_sat)
            
            if not in_any_exclusive:
                continue  # Skip cette obs (pas dans une exclusive)
            
            # Calculer la fenêtre (intersection obs ∩ exclusives valides), on prend la plus large fenêtre où l'obs est compatible
            valid_windows = [w for w in exclusives_on_sat if obs.t_start >= w.t_start and obs.t_end <= w.t_end]
            
            if not valid_windows:
                continue
            
            # Fenêtre = union des fenêtres valides (simplifié)
            eff_t_start = max(obs.t_start, min(w.t_start for w in valid_windows))
            eff_t_end = min(obs.t_end, max(w.t_end for w in valid_windows))
        else:
            # Pas d'exclusives sur ce satellite : fenêtre = obs intersection satellite
            eff_t_start = max(obs.t_start, sat.t_start)
            eff_t_end = min(obs.t_end, sat.t_end)
        
        if eff_t_end - eff_t_start < obs.duration: # pas assez de place
            continue
        
        tau = sat.transition_time
        if not current_plan:
            # Première obs sur ce satellite
            t_insert = eff_t_start
            if t_insert + obs.duration <= eff_t_end:
                current_plan.append((obs, t_insert))
                continue
        
        current_plan.sort(key=lambda p: p[1]) # trier par début temps
        
        # essai insérer avant la première obs
        first_obs, first_t = current_plan[0]
        if eff_t_start + obs.duration + tau <= first_t:
            current_plan.insert(0, (obs, eff_t_start))
            continue
        
        # essai d'insérer entre deux obs
        inserted = False
        for i in range(len(current_plan) - 1):
            obs_prev, t_prev = current_plan[i]
            obs_next, t_next = current_plan[i + 1]
            
            t_after_prev = t_prev + obs_prev.duration + tau
            t_before_next = t_next - tau
            t_candidate = max(t_after_prev, eff_t_start)
            if t_candidate + obs.duration <= min(t_before_next, eff_t_end):
                current_plan.insert(i + 1, (obs, t_candidate))
                inserted = True
                break
        
        if inserted:
            continue
        
        # essai d'insérer après la dernière obs
        last_obs, last_t = current_plan[-1]
        t_after_last = last_t + last_obs.duration + tau
        t_candidate = max(t_after_last, eff_t_start)
        
        if t_candidate + obs.duration <= eff_t_end:
            current_plan.append((obs, t_candidate))
    
    for sat_id in plans:
        plans[sat_id].sort(key=lambda p: p[1])
    
    return plans

def compute_reward_from_plan(plan_for_user):
    return sum(obs.reward for sat_plan in plan_for_user.values() for (obs, _) in sat_plan)

_plan_cache = {}

def compute_pi(instance, user_id, obs_candidate, allocated_obs):
    """
        Calcule π = reward_after - reward_before
        en créant une COPIE de obs_candidate avec owner=user_id
    """
    cache_key = (user_id, tuple(sorted([o.oid for o in allocated_obs])))
    
    if cache_key not in _plan_cache:
        plan_before = greedy_schedule_P_u(instance, user_id)
        _plan_cache[cache_key] = sum(o.reward for sat_plan in plan_before.values() 
                                   for o, _ in sat_plan)
    
    reward_before = _plan_cache[cache_key]
    
    # Créer une COPIE de obs_candidate avec owner=user_id
    obs_copy = Observation(oid=f"{obs_candidate.oid}_pi_{user_id}",
                        task_id=obs_candidate.task_id,
                        satellite=obs_candidate.satellite,
                        t_start=obs_candidate.t_start,
                        t_end=obs_candidate.t_end,
                        duration=obs_candidate.duration,
                        reward=obs_candidate.reward,
                        owner=user_id)
    
    # Création d'une sous-instance AVEC obs_copy
    user = next(u for u in instance.users if u.uid == user_id)
    
    new_obs = [o for o in instance.observations if o.owner == user_id]
    new_obs.append(obs_copy)
    new_tasks = [t for t in instance.tasks if t.owner == user_id]
    
    if obs_candidate.task_id not in [t.tid for t in new_tasks]: # Ajouter la task si elle n'existe pas
        orig_task = next((t for t in instance.tasks if t.tid == obs_candidate.task_id), None)
        if orig_task:
            # Créer copie
            task_copy = Task(tid=orig_task.tid,
                            owner=user_id,  # <- owner=user_id
                            t_start=orig_task.t_start,
                            t_end=orig_task.t_end,
                            duration=orig_task.duration,
                            reward=orig_task.reward,
                            opportunities=orig_task.opportunities)
            new_tasks.append(task_copy)
    
    inst_with_obs = ESOPInstance(nb_satellites=instance.nb_satellites, nb_users=1, nb_tasks=len(new_tasks), horizon=instance.horizon,
                                 satellites=instance.satellites, users=[user], tasks=new_tasks, observations=new_obs)
    
    plan_after = greedy_schedule(inst_with_obs).get(user_id, {})
    reward_after = sum(o.reward for sat_plan in plan_after.values() for o, _ in sat_plan)
    
    # Vérifier que obs_copy est bien dans le plan
    obs_inserted = any(obs_copy.oid == o.oid for sat_plan in plan_after.values() for o, _ in sat_plan)
    if not obs_inserted:
        return None
    
    pi_val = reward_after - reward_before
    assert pi_val > 0, "pi doit être strictement positif si l'observation est insérée" # le coût sera mis en négatif dans le yaml
    return pi_val

import yaml
from ESOPInstance import ESOPInstance, Task

def generate_sdcop_yaml_for_request(instance, request, user_allocated_obs, user_obs_times, output_path):
    """
        Génère un fichier YAML DCOP pour une requête centrale donnée.
    """
    agents = [u.uid for u in instance.users if u.uid != "u0"]
    if not agents:
        return False
    
    candidate_obs = [o for o in instance.observations 
                     if o.task_id == request.tid and o.owner == "u0"]
    
    if not candidate_obs:
        return False
    
    variables_section = {}
    constraints_section = {}
    vars_by_request = []
    vars_by_user_sat = {}
    
    exclusives_by_user = {u.uid: u.exclusive_windows for u in instance.users if u.uid != "u0"}
    
    for obs in candidate_obs: # variables
        for user_id in agents:
            windows = exclusives_by_user[user_id]
            
            in_exclusive = any(
                w.satellite == obs.satellite and
                not (w.t_end <= obs.t_start or w.t_start >= obs.t_end)
                for w in windows
            )
            
            if not in_exclusive:
                continue
            
            pi = compute_pi(instance, user_id, obs, user_allocated_obs.get(user_id, []))
            
            if pi is None:
                continue
            
            v_name = f"x_{user_id}_{obs.oid}"
            
            variables_section[v_name] = {
                "domain": "binary"
            }
            
            vars_by_request.append(v_name)
            key = (user_id, obs.satellite)
            vars_by_user_sat.setdefault(key, []).append(v_name)
            
            c_name = f"c_pi_{user_id}_{obs.oid}"
            constraints_section[c_name] = {
                "type": "intention",
                "function": f"{-pi} * {v_name}"
            }
    
    if not variables_section:
        return False
    
    nb_vars = len(variables_section)
    print(f"> DCOP {request.tid}: {nb_vars} variables")
    
    # contrainte au plus 1
    if len(vars_by_request) > 1:
        c_name = f"c_atmost1_{request.tid}"
        sum_expr = " + ".join(vars_by_request)
        
        constraints_section[c_name] = {"type": "intention", "function": f"0 if {sum_expr} <= 1 else 1e9"}
    
    # contrainte capacité satellite
    sat_capacity = {s.sid: s.capacity for s in instance.satellites}
    for (user_id, sat_id), vnames in vars_by_user_sat.items():
        c_name = f"c_cap_{sat_id}_{request.tid}"
        cap = sat_capacity[sat_id]
        
        sum_expr = " + ".join(vnames) if len(vnames) > 1 else vnames[0]
        
        constraints_section[c_name] = {"type": "intention", "function": f"0 if {sum_expr} <= {cap} else 1e9"}
    
    # !!! REAJOUTER agents auxiliaires AVEC capacité
    nb_computations = len(variables_section) + len(constraints_section)
    
    agents_list = list(agents)
    while len(agents_list) < nb_computations:
        agents_list.append(f"aux_{len(agents_list)}")
    
    agents_with_capacity = {}
    for agent_id in agents_list:
        agents_with_capacity[agent_id] = {"capacity": 1000} # grande capacité arbitraire pour agents auxiliaires
    
    dcop_dict = {
        "name": f"sdcop_{request.tid}",
        "objective": "min",
        "domains": {"binary": {"values": [0, 1]}},
        "agents": agents_with_capacity,
        "variables": variables_section,
        "constraints": constraints_section
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dcop_dict, f, sort_keys=False)
    
    return True

import tempfile
import os
import json
from DCOP import run_pydcop_solve, extract_metrics_from_output, parse_assignment_from_output
def sdcop_with_pydcop(instance: ESOPInstance, timeout_per_dcop=5000, algo="dpop"):
    """
    Résout l'instance ESOP avec l'approche SDCOP + PyDCOP.
    """
    central_requests = [r for r in instance.tasks if r.owner == "u0"]
    if not central_requests:
        return greedy_schedule(instance), [], 0.0, 0, 0
    
    user_allocated_obs = {u.uid: [] for u in instance.users if u.uid != "u0"}
    user_obs_times = {u.uid: {} for u in instance.users if u.uid != "u0"}
    
    all_assignments = []
    
    total_msgs = 0
    total_load = 0
    total_time = 0.0
    nb_dcops = 0
    nb_timeouts = 0
    nb_dcops_attempted = 0
    
    for request in central_requests:
        nb_dcops_attempted += 1
        yaml_path = None
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml_path = f.name
            ok = generate_sdcop_yaml_for_request(instance, request, user_allocated_obs, user_obs_times, yaml_path)
            
            if not ok:
                continue
            
            output = run_pydcop_solve(yaml_path, algo=algo, timeout=timeout_per_dcop)
            
            if output is None:
                nb_timeouts += 1
                continue
            
            msgs, load = extract_metrics_from_output(output)
            total_msgs += msgs
            total_load += load
            try:
                result_json = json.loads(output)
                total_time += result_json.get('time', 0.0)
            except:
                pass
            
            nb_dcops += 1
            
            assignment = parse_assignment_from_output(output)            
            for var_name, value in assignment.items(): # analyser l'assignement
                if value == 1 and var_name.startswith('x_'):
                    parts = var_name.split('_', 2)
                    if len(parts) == 3:
                        user_id = parts[1]
                        obs_id = parts[2]
                        
                        obs = next((o for o in instance.observations if o.oid == obs_id), None)
                        if obs:
                            all_assignments.append((request.tid, user_id, obs))
                            user_allocated_obs[user_id].append(obs)
        
        except Exception as e:
            print(f"Erreur {request.tid}: {e}")
        
        finally:
            if yaml_path and os.path.exists(yaml_path):
                try:
                    os.unlink(yaml_path)
                except:
                    pass
    
    print(f"> SDCOP: {nb_dcops}/{nb_dcops_attempted} DCOPs résolus, {nb_timeouts} timeouts, {len(all_assignments)} allocations")
    
    # Construction des plans finaux
    sdcop_plan = {}
    for user in instance.users:
        if user.uid == "u0":
            continue
        
        user_obs = [o for o in instance.observations if o.owner == user.uid]
        user_obs += user_allocated_obs[user.uid]
        
        user_tasks = [t for t in instance.tasks if t.owner == user.uid]
        
        for obs in user_allocated_obs[user.uid]:
            if obs.task_id not in [t.tid for t in user_tasks]:
                orig_task = next((t for t in instance.tasks if t.tid == obs.task_id), None)
                if orig_task:
                    user_tasks.append(orig_task)
        
        if user_obs:
            inst_user = ESOPInstance(nb_satellites=instance.nb_satellites,
                                    nb_users=1,
                                    nb_tasks=len(user_tasks),
                                    horizon=instance.horizon,
                                    satellites=instance.satellites,
                                    users=[user],
                                    tasks=user_tasks,
                                    observations=user_obs)
            plan = greedy_schedule(inst_user).get(user.uid, {})
            sdcop_plan[user.uid] = plan
        else:
            sdcop_plan[user.uid] = {}
    
    allocated_obs_ids = {obs.oid for _, _, obs in all_assignments}
    u0_obs = [o for o in instance.observations 
              if o.owner == "u0" and o.oid not in allocated_obs_ids]
    
    u0_tasks = [t for t in instance.tasks if t.owner == "u0"]
    u0_user = next(u for u in instance.users if u.uid == "u0")
    
    if u0_obs:
        inst_u0 = ESOPInstance(nb_satellites=instance.nb_satellites,
                                nb_users=1,
                                nb_tasks=len(u0_tasks),
                                horizon=instance.horizon,
                                satellites=instance.satellites,
                                users=[u0_user],
                                tasks=u0_tasks,
                                observations=u0_obs)
        sdcop_plan["u0"] = greedy_schedule(inst_u0).get("u0", {})
    else:
        sdcop_plan["u0"] = {}
    
    avg_time = total_time / nb_dcops if nb_dcops > 0 else 0.0
    return sdcop_plan, all_assignments, avg_time, total_msgs, total_load
