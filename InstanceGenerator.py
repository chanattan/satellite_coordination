from ESOPInstance import *
import random

def generate_DCOP_instance(instance):
    pass

def generate_ESOP_instance(nb_satellites: int, nb_users: int, nb_tasks: int, horizon: int = 300, capacity: int = 20, seed: int = None) -> ESOPInstance:
    """
    Étant donné un nombre de satellites nb_satellites, nombre d'utilisateurs exclusifs nb_users et un nombre de tâches nb_tasks,
    cette fonction génère et retourne une instance aléatoire I du problème ESOP (EOSCSP avec utilisateurs exclusifs).

    L'instance est modélisée par un objet (classe Python) et contient également un format texte pour l'exportation via la méthode to_text.

    Format texte (exemple) :

        [Parameters]
        Satellites : nb_satellites
        Exclusive users : nb_users
        Tasks : nb_tasks

        [Satellites]
        s0 0 300 20 1
        ...

        [Users]
        u0 -
        u1 s0:10-40, s1:100-130
        ...

        [Tasks]
        r_0 owner=u0 window=[10,80] duration=5 reward=3
        ...

        [Observations]
        o_r_0_0 task=r_0 owner=u0 sat=s0 window=[15,40] duration=5 reward=3
        ...
    """
    if seed is not None:
        random.seed(seed)

    # création des satellites
    satellites = []
    for i in range(nb_satellites):
        satellites.append(Satellite(sid=f"s{i}", t_start=0, t_end=horizon, capacity=capacity, transition_time=1))

    users = []
    # planificateur central (artificiel) sans exclusives
    users.append(User(uid="u0", exclusive_windows=[]))

    # utilisateurs exclusifs u1...u_{nb_users}
    for u_idx in range(nb_users):
        uid = f"u{u_idx+1}"
        # fenêtres exclusives aléatoires
        exclusive_windows = []
        # choix : 2 par utilisateur
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
    # générationd des tâches et observations
    for t_idx in range(nb_tasks):
        # on tire aléatoirement le propriétaire :
        # central avec prob. 0.5, sinon un exclusif
        if random.random() < 0.5 or nb_users == 0:
            owner = "u0"
        else:
            owner = f"u{random.randint(1, nb_users)}"

        tid = f"r_{t_idx}"
        # fenêtre de validité de la tâche
        t_start = random.randint(0, horizon // 2)
        t_end = random.randint(t_start + 10, horizon)
        duration = random.randint(3, 10)
        reward = random.randint(1, 10)

        task = Task(tid=tid, owner=owner, t_start=t_start, t_end=t_end, duration=duration, reward=reward, opportunities=[])

        # on génère 2 à 4 opportunités d'observation par tâche
        nb_opps = random.randint(2, 4)
        for k in range(nb_opps):
            oid = f"o_{tid}_{k}"
            sat = random.choice(satellites)
            # fenêtre de l'opportunité incluse dans celle de la tâche
            win_len = random.randint(duration + 1, max(duration + 2, t_end - t_start))
            o_start = random.randint(t_start, max(t_start, t_end - win_len))
            o_end = min(t_end, o_start + win_len)

            obs = Observation(oid=oid, task_id=tid, satellite=sat.sid, t_start=o_start, t_end=o_end, duration=duration, reward=reward, owner=owner)
            observations.append(obs)
            task.opportunities.append(obs)

        tasks.append(task)

    instance = ESOPInstance(nb_satellites=nb_satellites, nb_users=nb_users, nb_tasks=nb_tasks, horizon=horizon, satellites=satellites, users=users, tasks=tasks, observations=observations)
    return instance
