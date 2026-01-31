from ESOPInstance import *
import random
import yaml

def print_user_plans(user_plans):
    if not user_plans:
        print("Aucun planning généré.")
        return
    
    for u_id, plan in user_plans.items():
        print(f"Planning pour l'utilisateur {u_id}:")
        for sat_id, observations in plan.items():
            print(f"- Sur le satellite {sat_id}:")
            for obs, _ in observations:
                print(f" -> Observation {obs.oid} (reward: {obs.reward})")
        
        # Calculer le score total pour cet utilisateur
        total_reward = sum(obs.reward for sat_obs in plan.values() for obs, _ in sat_obs)
        print(f"> Score total: {total_reward}\n")

def generate_DCOP_instance(instance):
    """
        Génère une instance DCOP à partir d'une instance ESOP donnée.
    """
    agents = [u.uid for u in instance.users if u.uid != "u0"] # tous les utilisateurs exclusifs

    # Variables x_{u,o} pour les observations du central
    central_observations = [o for o in instance.observations if o.owner == "u0"]
    exclusives_by_user = {u.uid: u.exclusive_windows for u in instance.users if u.uid != "u0"}
    obs_by_id = {o.oid: o for o in instance.observations}

    variables_section = {}
    vars_by_obs = {}
    vars_by_user_sat = {}
    for o in central_observations:
        for u_id, windows in exclusives_by_user.items():
            has_excl = any(w.satellite == o.satellite and not (w.t_end <= o.t_start or w.t_start >= o.t_end) for w in windows)
            if not has_excl: continue

            v_name = f"x_{u_id}_{o.oid}"
            variables_section[v_name] = {"domain": "binary", "agent": u_id}

            vars_by_obs.setdefault(o.oid, []).append(v_name)
            key = (u_id, o.satellite)
            vars_by_user_sat.setdefault(key, []).append(v_name)

    constraints_section = {}
    # au plus une par observation
    for o in central_observations:
        if o.oid not in vars_by_obs:
            continue
        vnames = vars_by_obs[o.oid]
        c_name = f"c_atmost1_{o.oid}"

        # syntaxe expression directe
        total_expr = " + ".join(vnames) if len(vnames) > 1 else vnames[0]
        expression = f"0 if {total_expr} <= 1 else 1e9"

        constraints_section[c_name] = {"type": "intention", "function": expression}

    # capacité par (u, s)
    sat_capacity = {s.sid: s.capacity for s in instance.satellites}

    for (u_id, sat_id), vnames in vars_by_user_sat.items():
        c_name = f"c_cap_{u_id}_{sat_id}"
        cap = sat_capacity[sat_id]

        total_expr = " + ".join(vnames) if len(vnames) > 1 else vnames[0]
        expression = f"0 if {total_expr} <= {cap} else 1e9"

        constraints_section[c_name] = {"type": "intention", "function": expression}

    for v_name in variables_section.keys():
        _, u_id, oid = v_name.split("_", 2)
        o = obs_by_id[oid]
        rew = o.reward
        c_name = f"c_reward_{v_name}"

        # syntaxe expression directe pour unaire
        expression = f"{-rew} * {v_name}"
        constraints_section[c_name] = {"type": "intention", "function": expression}

    # Ajouter agents auxiliaires pour la distribution (contrainte PyDcop pour nb agents suffisant)
    nb_vars = len(variables_section)
    nb_constraints = len(constraints_section)
    nb_computations = nb_vars + nb_constraints

    real_agents = agents
    while len(real_agents) < nb_computations:
        real_agents.append(f"aux_{len(real_agents)}")

    # DCOP YAML
    dcop_dict = {
        "name": "esop_dcop",
        "objective": "min",
        "domains": {"binary": {"values": [0, 1]}},
        "agents": real_agents,
        "variables": variables_section,
        "constraints": constraints_section
    }

    yaml_str = yaml.dump(dcop_dict, sort_keys=False)
    return yaml_str

def generate_ESOP_instance(
    nb_satellites,
    nb_users, # nb d'utilisateurs exclusifs (hors u0)
    nb_tasks,
    horizon = 300,
    capacity = 20,
    seed = None,
    scenario = "generic", # "generic", "small_scale", "large_scale",
    one_exclusive_user_per_satellite=False # si True, chaque satellite a au plus un utilisateur exclusif par satellite
):
    """
    Génère une instance ESOP fidèle au modèle de l'article.

    - S : nb_satellites, horizon, capacity, transition_time = 1.
    - U : u0 (central, sans exclusives) + u1..u_nb_users (exclusifs, avec exclusive_windows).
    - R : nb_tasks requêtes, réparties entre u0 et les exclusifs selon le scénario.
    - O : opportunités pour chaque requête :
        - pour un exclusif ui : chaque observation est incluse dans UNE de ses exclusives,
          rewards élevés pour garantir la priorité.
        - pour u0 : mix d'opportunités dans les exclusives (partage) et hors exclusives,
          rewards plus faibles.

    Le paramètre `scenario` permet de se rapprocher des configurations de la section 6 :
      - "generic" : valeurs par défaut set en params.
      - "small_scale" : proche des "highly conflicting small-scale problems".
      - "large_scale" : proche des "realistic large-scale problems".
    
    Le paramètre one_exclusive_window_per_satellite permet de forcer au plus une un utilisateur exclusif par satellite.
    """
    if seed is not None:
        random.seed(seed)

    if scenario == "small_scale": # figure 4 article
        # 5 min horizon dans l'article, avec horizon=300 ici
        EXCL_WINDOWS_PER_USER = 8
        EXCL_WINDOW_LENGTH_RANGE = (15, 20)

        # 2 à 20 requêtes par exclusif, 8 à 80 pour u0 dans le papier.
        # Ici, nb_tasks est global : on approxime via la probabilité.
        PROB_TASK_FOR_U0 = 0.5

        # 10 opportunités par requête, durée ~5
        NB_OPPS_RANGE = (10, 10)
        DURATION_RANGE = (5, 5)

        # Fenêtre de requête
        TASK_WINDOW_MIN_LENGTH = 10

        # Rewards : exclusifs élevés, central faible
        REWARD_EXCLUSIVE_RANGE = (10, 50)
        REWARD_CENTRAL_RANGE = (1, 5)

        # probabilité qu'une opportunité de u0 soit dans une exclusive
        PROB_U0_IN_EXCLUSIVE = 0.7

        # Horizon
        horizon = 300
    elif scenario == "large_scale": # figure 5 article
        # 6 heures environ 21600s dans le papier ; horizon paramétrable
        EXCL_WINDOWS_PER_USER = 10
        EXCL_WINDOW_LENGTH_RANGE = (300, 600)

        # 20 à 100 requêtes par exclusif, 25 à 250 pour u0 (approximation via probas)
        PROB_TASK_FOR_U0 = 0.5

        # 5 opportunités par requête, durée 20
        NB_OPPS_RANGE = (5, 5)
        DURATION_RANGE = (20, 20)

        TASK_WINDOW_MIN_LENGTH = 40  # fenêtres obs ~40–60

        # Rewards, même logique : exclusifs >> central
        REWARD_EXCLUSIVE_RANGE = (10, 50)
        REWARD_CENTRAL_RANGE = (1, 5)

        PROB_U0_IN_EXCLUSIVE = 0.7
    elif scenario == "custom_scale":
        EXCL_WINDOWS_PER_USER = 8
        EXCL_WINDOW_LENGTH_RANGE = (30, 60) # plus de place par défaut
        
        # RÉDUIT le nombre d'obs des exclusifs
        PROB_TASK_FOR_U0 = 0.7 # + de tasks pour u0
        
        NB_OPPS_RANGE = (5, 10)
        DURATION_RANGE = (3, 5) 

        TASK_WINDOW_MIN_LENGTH = 10  # fenêtres obs env. 40–60
        
        REWARD_EXCLUSIVE_RANGE = (10, 30)
        REWARD_CENTRAL_RANGE = (15, 40)
        PROB_U0_IN_EXCLUSIVE = 0.7
    else: # generic
        EXCL_WINDOWS_PER_USER = 1
        EXCL_WINDOW_LENGTH_RANGE = (15, 60)

        PROB_TASK_FOR_U0 = 0.5

        NB_OPPS_RANGE = (2, 10)
        DURATION_RANGE = (3, 10)
        TASK_WINDOW_MIN_LENGTH = 10

        REWARD_EXCLUSIVE_RANGE = (50, 100)
        REWARD_CENTRAL_RANGE = (1, 10)

        PROB_U0_IN_EXCLUSIVE = 0.7
    
    satellites = []
    for i in range(nb_satellites):
        satellites.append(Satellite(sid=f"s{i}", t_start=0, t_end=horizon, capacity=capacity, transition_time=1))

    users = []
    users.append(User(uid="u0", exclusive_windows=[]))  # central
    attributed_sats = dict() # pour one_exclusive_window_per_satellite

    # pour garantir l'absence d'overlap des exclusives sur un même satellite,
    # on garde pour chaque sat la liste des intervalles déjà utilisés
    excl_by_sat = {s.sid: [] for s in satellites}

    def sample_non_overlapping_interval(sid, length_range):
        """
            Tente de générer une fenêtre [start, end] sur le satellite sid qui ne chevauche aucune fenêtre déjà présente dans excl_by_sat[sid].
        """
        existing = excl_by_sat[sid]
        # si aucun intervalle, on place librement
        if not existing:
            length = random.randint(*length_range)
            start = random.randint(0, max(0, horizon - length))
            end = start + length
            return (start, end)

        # on essaie quelques placements aléatoires compatibles
        for _ in range(20):
            length = random.randint(*length_range)
            start = random.randint(0, max(0, horizon - length))
            end = start + length
            if all(end <= s0 or start >= s1 for (s0, s1) in existing):
                return (start, end)
        # échec......
        return None

    for u_idx in range(nb_users):
        uid = f"u{u_idx+1}"
        exclusive_windows = []

        if one_exclusive_user_per_satellite:
            taken_sats = attributed_sats.get(uid, None)
            if taken_sats is not None: # cet utilisateur a déjà un satellite attribué
                sat = next(s for s in satellites if s.sid == attributed_sats[uid])
            else:
                available_sats = [s for s in satellites if s.sid not in attributed_sats.values()]
                if not available_sats:
                    # plus de satellites disponibles pour des exclusives
                    users.append(User(uid=uid, exclusive_windows=[]))
                    continue
                sat = random.choice(available_sats)
                attributed_sats[uid] = sat.sid

            # toutes les fenêtres de cet utilisateur seront sur ce satellite,
            # en s'assurant qu'elles ne se chevauchent pas entre elles
            for _ in range(EXCL_WINDOWS_PER_USER):
                interval = sample_non_overlapping_interval(sat.sid, EXCL_WINDOW_LENGTH_RANGE)
                if interval is None:
                    # pas possible de placer plus de fenêtres sur ce sat
                    break
                start, end = interval
                exclusive_windows.append(ExclusiveWindow(satellite=sat.sid, t_start=start, t_end=end))
                excl_by_sat[sat.sid].append((start, end))
        else:
            # cas général : l'utilisateur peut avoir plusieurs satellites,
            # mais les exclusives ne doivent pas se chevaucher sur un même sat
            for _ in range(EXCL_WINDOWS_PER_USER):
                sat = random.choice(satellites)
                interval = sample_non_overlapping_interval(sat.sid, EXCL_WINDOW_LENGTH_RANGE)
                if interval is None:
                    # impossible de placer une nouvelle fenêtre sur ce sat sans chevauchement
                    continue
                start, end = interval
                exclusive_windows.append(ExclusiveWindow(satellite=sat.sid, t_start=start, t_end=end))
                excl_by_sat[sat.sid].append((start, end))

        users.append(User(uid=uid, exclusive_windows=exclusive_windows))

    exclusive_users = [u for u in users if u.uid != "u0"]

    tasks = []
    observations = []
    for t_idx in range(nb_tasks):
        # Répartition des tasks entre u0 et les exclusifs
        if random.random() < PROB_TASK_FOR_U0 or nb_users == 0:
            owner = "u0"
        else:
            owner = f"u{random.randint(1, nb_users)}"

        tid = f"r_{t_idx}"

        # Fenêtre de la requête
        t_start = random.randint(0, max(0, horizon - TASK_WINDOW_MIN_LENGTH - 1))
        t_end = random.randint(t_start + TASK_WINDOW_MIN_LENGTH, horizon)

        # Durée d'une observation
        duration = random.randint(*DURATION_RANGE)

        # Reward de la requête (et des obs)
        if owner == "u0":
            reward = random.randint(*REWARD_CENTRAL_RANGE)
        else:
            reward = random.randint(*REWARD_EXCLUSIVE_RANGE)

        task = Task(tid=tid, owner=owner, t_start=t_start, t_end=t_end, duration=duration, reward=reward, opportunities=[])

        nb_opps = random.randint(*NB_OPPS_RANGE)

        for k in range(nb_opps):
            oid = f"o_{tid}_{k}"

            # cas où requête d'un utilisateur exclusif
            if owner != "u0":
                u_owner = next(u for u in users if u.uid == owner)
                if not u_owner.exclusive_windows:
                    # cas limite, on saute
                    continue

                # toutes les opportunités d'un exclusif doivent être dans UNE de ses exclusives
                windows = u_owner.exclusive_windows[:]
                random.shuffle(windows)
                placed = False
                sat = None

                for w in windows:
                    sat = next(s for s in satellites if s.sid == w.satellite)
                    # intersection de la fenêtre de la requête et de l'exclusive
                    win_start = max(t_start, w.t_start)
                    win_end = min(t_end, w.t_end)
                    # fenêtre obs incluse dans l'exclusive
                    if win_end - win_start >= duration + 1:
                        o_start = random.randint(win_start, win_end - duration - 1)
                        o_end = o_start + duration + 1
                        placed = True
                        break

                if not placed:
                    # aucune exclusive ne peut accueillir cette opportunité
                    continue

            # cas où requête du central u0
            else:
                if exclusive_users and random.random() < PROB_U0_IN_EXCLUSIVE:
                    # opportunité dans une exclusive d'un utilisateur
                    excl_user = random.choice(exclusive_users)
                    if excl_user.exclusive_windows:
                        w = random.choice(excl_user.exclusive_windows)
                        sat = next(s for s in satellites if s.sid == w.satellite)

                        win_start = max(t_start, w.t_start)
                        win_end = min(t_end, w.t_end)

                        if win_end - win_start >= duration + 1:
                            o_start = random.randint(win_start, win_end - duration - 1)
                            o_end = o_start + duration + 1
                        else:
                            # pas assez de place dans cette exclusive -> hors exclusives
                            sat = random.choice(satellites)
                            win_len = random.randint(duration + 1, max(duration + 2, t_end - t_start))
                            o_start = random.randint(t_start, max(t_start, t_end - win_len))
                            o_end = min(t_end, o_start + win_len)
                    else:
                        sat = random.choice(satellites)
                        win_len = random.randint(duration + 1, max(duration + 2, t_end - t_start))
                        o_start = random.randint(t_start, max(t_start, t_end - win_len))
                        o_end = min(t_end, o_start + win_len)
                else:
                    # opportunité de u0 hors de toute exclusive (au moins en intention)
                    sat = random.choice(satellites)
                    win_len = random.randint(duration + 1, max(duration + 2, t_end - t_start))
                    o_start = random.randint(t_start, max(t_start, t_end - win_len))
                    o_end = min(t_end, o_start + win_len)

            obs = Observation(oid=oid, task_id=tid, satellite=sat.sid, t_start=o_start, t_end=o_end, duration=duration, reward=reward, owner=owner)
            observations.append(obs)
            task.opportunities.append(obs)

        tasks.append(task)

    instance = ESOPInstance(nb_satellites=nb_satellites, nb_users=nb_users, nb_tasks=nb_tasks, horizon=horizon, satellites=satellites, users=users, tasks=tasks, observations=observations)

    # Sanity check : toutes les obs d'exclusifs sont dans leurs exclusives
    for o in instance.observations:
        if o.owner == "u0":
            continue
        u = next(u for u in instance.users if u.uid == o.owner)
        assert any(w.satellite == o.satellite and o.t_start >= w.t_start and o.t_end <= w.t_end for w in u.exclusive_windows), f"{o.oid} de {o.owner} hors exclusive"

    return instance

def generate_benchmark_instances(scenario="small_scale", num_instances=30):
    """
        génère 30 instances qui matchent les configurations expérimentales de l'article pour le benchmarking.
    """
    
    if scenario == "small_scale":
        params = {'nb_satellites': 3, 'nb_users': 4, 'nb_tasks': [25, 50, 75, 100, 125, 150]} # 250-1500 obs
    else: # large_scale
        params = {'nb_satellites': 8, 'nb_users': 5, 'nb_tasks': [100, 200, 300, 400, 500]} # 500-2500 obs
    
    instances = {obs_count: [] for obs_count in params['nb_tasks']}
    
    for obs_count in params['nb_tasks']:
        for seed in range(num_instances):
            instance = generate_ESOP_instance(nb_satellites=params['nb_satellites'], nb_users=params['nb_users'], nb_tasks=obs_count, scenario=scenario, seed=seed)
            instances[obs_count].append(instance)
    
    return instances

if __name__ == "__main__":
    from GreedySolver import greedy_schedule
    instances = generate_benchmark_instances(scenario="small_scale", num_instances=30)
    for i in instances:
        # solve with greedy and print plans
        for inst in instances[i]:
            plans = greedy_schedule(inst)
            print_user_plans(plans)
            score = assess_solution(inst, plans)
            print(f"Score total pour l'instance : {sum(score.values())}\n")