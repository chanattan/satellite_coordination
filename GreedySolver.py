from ESOPInstance import ESOPInstance

def greedy_schedule_P_u(instance, user_id):
    """
        Résout P_u avec l'algorithme glouton pour un utilisateur donné :

        P_u = <S, U, R_u, O_u>
        - S : mêmes satellites que l'instance globale
        - U : on peut garder tous les users ou seulement u (ici on garde juste u, car les autres n'interviennent pas dans le solve local)
        - R_u : tâches dont le owner est u
        - O_u : observations dont le owner est u
    """
    u = next(user for user in instance.users if user.uid == user_id)
    tasks_u = [t for t in instance.tasks if t.owner == user_id]
    obs_u = [o for o in instance.observations if o.owner == user_id]

    # Sous-instance P_u (mêmes satellites et horizon)
    inst_u = ESOPInstance(nb_satellites=instance.nb_satellites,
                        nb_users=1,  # nombre d'utilisateurs exclusifs dans cette sous-instance (ici juste u)
                        nb_tasks=len(tasks_u),
                        horizon=instance.horizon,
                        satellites=instance.satellites,
                        users=[u], # on ne garde que u ici, suffisant pour le solve local
                        tasks=tasks_u,
                        observations=obs_u,)
    all_plans_u = greedy_schedule(inst_u)

    # On ne récupère que le plan de u
    return all_plans_u.get(user_id, {})

def greedy_schedule(instance):
    """
        Algo 1 Greedy EOSCSP solver avec priorité absolue aux exclusifs en deux temps 1) exclusifs d'abord 2) u0 ensuite
    """
    user_plans = {} # uid -> sid -> liste (Observation, t_start)
    
    # Rs : plan actuel par satellite
    Rs = {sat.sid: [] for sat in instance.satellites}
    sat_by_id = {s.sid: s for s in instance.satellites}
    tasks_satisfied = set() # (au plus une obs par tâche)
    
    def first_slot(o):
        # trouve le premier créneau valide
        s = sat_by_id[o.satellite]
        sid = s.sid
        tau = s.transition_time
        
        if len(Rs[sid]) >= s.capacity:
            return None
        
        plan_s = Rs[sid]  # déjà trié par t_start
        
        # Satellite vide ?
        if not plan_s:
            t0 = max(s.t_start, o.t_start)
            if t0 + o.duration <= min(s.t_end, o.t_end):
                Rs[sid].append((o, t0))
                return t0
            return None
        
        # Avant première obs ?
        first_obs, first_t = plan_s[0]
        t_upper = first_t
        t0 = max(s.t_start, o.t_start)
        if t0 + o.duration + tau <= t_upper:
            Rs[sid].insert(0, (o, t0))
            return t0
        
        # Entre deux obs successives ?
        for i in range(len(plan_s) - 1):
            prev_obs, prev_t = plan_s[i]
            next_obs, next_t = plan_s[i + 1]
            
            # Après prev + transition
            after_prev = prev_t + prev_obs.duration + tau
            # Avant next - transition
            before_next = next_t - tau
            
            t0 = max(after_prev, o.t_start)
            if t0 + o.duration <= min(before_next, o.t_end, s.t_end):
                Rs[sid].insert(i + 1, (o, t0))
                return t0
        
        # Dernier cas après dernière obs
        last_obs, last_t = plan_s[-1]
        after_last = last_t + last_obs.duration + tau
        t0 = max(after_last, o.t_start)
        if t0 + o.duration <= min(s.t_end, o.t_end):
            Rs[sid].append((o, t0))
            return t0
        
        return None
    
    # UNIQUEMENT obs EXCLUSIFS (priorité absolue)
    exclusive_obs = [o for o in instance.observations if o.owner != "u0"]
    exclusive_obs.sort(key=lambda o: (-o.reward, o.t_start)) # tri par reward décroissant, t_start croissant
    
    for o in exclusive_obs:
        # Vérifier que l'obs est dans une exclusive de son owner
        u_owner = next(u for u in instance.users if u.uid == o.owner)
        in_exclusive = any(w.satellite == o.satellite and o.t_start >= w.t_start and o.t_end <= w.t_end for w in u_owner.exclusive_windows)
        if not in_exclusive:
            continue
            
        if o.task_id in tasks_satisfied:
            continue
            
        t = first_slot(o)
        if t is not None:
            tasks_satisfied.add(o.task_id)
            user_plans.setdefault(o.owner, {}).setdefault(o.satellite, []).append((o, t))
    
    # obs u0 APRÈS exclusifs
    u0_obs = [o for o in instance.observations if o.owner == "u0"]
    u0_obs.sort(key=lambda o: (-o.reward, o.t_start))
    
    for o in u0_obs:
        if o.task_id in tasks_satisfied:
            continue
            
        t = first_slot(o)
        if t is not None:
            tasks_satisfied.add(o.task_id)
            user_plans.setdefault("u0", {}).setdefault(o.satellite, []).append((o, t))
    
    # Tri final par temps
    for uid in user_plans:
        for sid in user_plans[uid]:
            user_plans[uid][sid].sort(key=lambda p: p[1])
    
    #print(f"> Greedy: {len(tasks_satisfied)}/{len(instance.tasks)} tâches satisfaites")
    return user_plans