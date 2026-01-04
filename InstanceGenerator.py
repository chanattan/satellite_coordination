from ESOPInstance import *
from typing import Dict, List
import random
import yaml

def print_user_plans(user_plans):
    """
    Affiche les plannings utilisateurs.
    """
    if not user_plans:
        print("Aucun planning généré.")
        return
    
    for u_id, plan in user_plans.items():
        print(f"Planning pour l'utilisateur {u_id}:")
        for sat_id, observations in plan.items():
            print(f"  Sur le satellite {sat_id}:")
            for obs, _ in observations:
                print(f"    > Observation {obs.oid} (reward: {obs.reward})")
        
        # Calculer le score total pour cet utilisateur
        total_reward = sum(obs.reward for sat_obs in plan.values() for obs, _ in sat_obs)
        print(f"  Score total: {total_reward}\n")

def generate_DCOP_instance(instance: ESOPInstance) -> str:
    """
    Génère une instance DCOP à partir d'une instance ESOP donnée.
    Utilise la syntaxe "expression" directe de pyDCOP pour les contraintes.
    """

    # Agents : tous les utilisateurs exclusifs
    agents = [u.uid for u in instance.users if u.uid != "u0"]

    # Variables x_{u,o} pour les observations du central
    central_observations = [o for o in instance.observations if o.owner == "u0"]
    exclusives_by_user: Dict[str, List[ExclusiveWindow]] = {
        u.uid: u.exclusive_windows for u in instance.users if u.uid != "u0"
    }
    obs_by_id = {o.oid: o for o in instance.observations}

    variables_section = {}
    vars_by_obs: Dict[str, List[str]] = {}
    vars_by_user_sat: Dict[tuple, List[str]] = {}

    for o in central_observations:
        for u_id, windows in exclusives_by_user.items():
            has_excl = any(
                w.satellite == o.satellite and not (w.t_end <= o.t_start or w.t_start >= o.t_end)
                for w in windows
            )
            if not has_excl:
                continue

            v_name = f"x_{u_id}_{o.oid}"
            variables_section[v_name] = {
                "domain": "binary",
                "agent": u_id
            }

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

        constraints_section[c_name] = {
            "type": "intention",
            "function": expression
        }

    # capacité par (u, s)
    sat_capacity = {s.sid: s.capacity for s in instance.satellites}

    for (u_id, sat_id), vnames in vars_by_user_sat.items():
        c_name = f"c_cap_{u_id}_{sat_id}"
        cap = sat_capacity[sat_id]

        total_expr = " + ".join(vnames) if len(vnames) > 1 else vnames[0]
        expression = f"0 if {total_expr} <= {cap} else 1e9"

        constraints_section[c_name] = {
            "type": "intention",
            "function": expression
        }

    # rewards unaires
    for v_name in variables_section.keys():
        _, u_id, oid = v_name.split("_", 2)
        o = obs_by_id[oid]
        rew = o.reward
        c_name = f"c_reward_{v_name}"

        # syntaxe expression directe pour unaire
        expression = f"{-rew} * {v_name}"

        constraints_section[c_name] = {
            "type": "intention",
            "function": expression
        }

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
        "domains": {
            "binary": {
                "values": [0, 1]
            }
        },
        "agents": real_agents,
        "variables": variables_section,
        "constraints": constraints_section
    }

    yaml_str = yaml.dump(dcop_dict, sort_keys=False)
    return yaml_str

def generate_ESOP_instance(nb_satellites: int, nb_users: int, nb_tasks: int, horizon: int = 300, capacity: int = 20, seed: int = None):
    """
    Génère une instance aléatoire du problème ESOP.
    """
    if seed is not None:
        random.seed(seed)

    # Création des satellites
    satellites = []
    for i in range(nb_satellites):
        satellites.append(Satellite(sid=f"s{i}", t_start=0, t_end=horizon,
                                   capacity=capacity, transition_time=1))

    users = []
    # Planificateur central sans exclusives
    users.append(User(uid="u0", exclusive_windows=[]))

    # Utilisateurs exclusifs
    for u_idx in range(nb_users):
        uid = f"u{u_idx+1}"
        exclusive_windows = []
        for _ in range(2):
            sat = random.choice(satellites)
            length = random.randint(20, 60)
            start = random.randint(0, max(0, horizon - length))
            end = start + length
            exclusive_windows.append(
                ExclusiveWindow(satellite=sat.sid, t_start=start, t_end=end)
            )
        users.append(User(uid=uid, exclusive_windows=exclusive_windows))

    tasks = []
    observations = []

    exclusive_users = [u for u in users if u.uid != "u0"]

    for t_idx in range(nb_tasks):
        # Owner avec prob. 0.5 pour u0 : à vérifier
        if random.random() < 0.5 or nb_users == 0:
            owner = "u0"
        else:
            owner = f"u{random.randint(1, nb_users)}"

        tid = f"r_{t_idx}"
        t_start = random.randint(0, horizon // 2)
        t_end = random.randint(t_start + 10, horizon)
        duration = random.randint(3, 10)
        reward = random.randint(1, 10)

        task = Task(tid=tid, owner=owner, t_start=t_start, t_end=t_end,
                    duration=duration, reward=reward, opportunities=[])

        nb_opps = random.randint(2, 4)

        for k in range(nb_opps):
            oid = f"o_{tid}_{k}"

            if owner == "u0" and exclusive_users:
                # Forcer une opportunité dans une fenêtre exclusive
                excl_user = random.choice(exclusive_users)
                if excl_user.exclusive_windows:
                    w = random.choice(excl_user.exclusive_windows)
                    sat = next(s for s in satellites if s.sid == w.satellite)

                    win_start = max(t_start, w.t_start)
                    win_end = min(t_end, w.t_end)

                    if win_end - win_start > duration + 1:
                        o_start = random.randint(win_start, win_end - duration - 1)
                        o_end = o_start + duration + 1
                    else:
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
                # Cas standard
                sat = random.choice(satellites)
                win_len = random.randint(duration + 1, max(duration + 2, t_end - t_start))
                o_start = random.randint(t_start, max(t_start, t_end - win_len))
                o_end = min(t_end, o_start + win_len)

            obs = Observation(oid=oid, task_id=tid, satellite=sat.sid,
                              t_start=o_start, t_end=o_end,
                              duration=duration, reward=reward, owner=owner)
            observations.append(obs)
            task.opportunities.append(obs)

        tasks.append(task)

    instance = ESOPInstance(nb_satellites=nb_satellites, nb_users=nb_users,
                            nb_tasks=nb_tasks, horizon=horizon,
                            satellites=satellites, users=users,
                            tasks=tasks, observations=observations)
    return instance