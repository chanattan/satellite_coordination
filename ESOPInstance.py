from typing import List, Dict, Any

class Satellite():
    def __init__(self, sid, t_start, t_end, capacity, transition_time=None):
        self.sid = sid
        self.t_start = t_start
        self.t_end = t_end
        self.capacity = capacity
        self.transition_time = transition_time if transition_time is not None else 1 # simplification : tau constant

class ExclusiveWindow():
    def __init__(self, satellite, t_start, t_end):
        self.satellite = satellite
        self.t_start = t_start
        self.t_end = t_end

class User():
    def __init__(self, uid, exclusive_windows):
        self.uid = uid
        self.exclusive_windows = exclusive_windows

class Observation():
    def __init__(self, oid, task_id, satellite, t_start, t_end, duration, reward, owner):
        self.oid = oid
        self.task_id = task_id
        self.satellite = satellite
        self.t_start = t_start
        self.t_end = t_end
        self.duration = duration
        self.reward = reward
        self.owner = owner # uid

class Task():
    def __init__(self, tid, owner, t_start, t_end, duration, reward, opportunities):
        self.tid = tid
        self.owner = owner
        self.t_start = t_start
        self.t_end = t_end
        self.duration = duration
        self.reward = reward
        self.opportunities = opportunities

class ESOPInstance():
    nb_satellites: int
    nb_users: int # nb d'utilisateurs exclusifs (hors u0)
    nb_tasks: int
    horizon: int
    satellites: List[Satellite]
    users: List[User] # inclut u0
    tasks: List[Task]
    observations: List[Observation]
    def __init__(self, nb_satellites: int, nb_users: int, nb_tasks: int, horizon: int,
                 satellites: List[Satellite], users: List[User], tasks: List[Task], observations: List[Observation]):
        """
            Initialise une instance ESOP avec les paramètres donnés.

            nb_users est le nombre d'utilisateurs exclusifs (hors u0).
            users inclut l'utilisateur central u0.
        """
        self.nb_satellites = nb_satellites
        self.nb_users = nb_users
        self.nb_tasks = nb_tasks
        self.horizon = horizon
        self.satellites = satellites
        self.users = users
        self.tasks = tasks
        self.observations = observations

    def to_text(self) -> str:
        """
        Export de l'instance en format texte.
        """
        lines = []
        lines.append("[Parameters]")
        lines.append(f"Satellites : {self.nb_satellites}")
        lines.append(f"Exclusive users : {self.nb_users}")
        lines.append(f"Tasks : {self.nb_tasks}")
        lines.append("")
        lines.append("[Satellites]")
        for sat in self.satellites:
            lines.append(
                f"{sat.sid} {sat.t_start} {sat.t_end} {sat.capacity} {sat.transition_time}"
            )
        lines.append("")
        lines.append("[Users]")
        for u in self.users:
            win_strs = [
                f"{w.satellite}:{w.t_start}-{w.t_end}" for w in u.exclusive_windows
            ]
            wins = ", ".join(win_strs) if win_strs else "-"
            lines.append(f"{u.uid} {wins}")
        lines.append("")
        lines.append("[Tasks]")
        for task in self.tasks:
            lines.append(
                f"{task.tid} owner={task.owner} "
                f"window=[{task.t_start},{task.t_end}] "
                f"duration={task.duration} reward={task.reward}"
            )
        lines.append("")
        lines.append("[Observations]")
        for o in self.observations:
            lines.append(
                f"{o.oid} task={o.task_id} owner={o.owner} "
                f"sat={o.satellite} window=[{o.t_start},{o.t_end}] "
                f"duration={o.duration} reward={o.reward}"
            )
        return "\n".join(lines)

def assess_solution(instance: ESOPInstance, user_plans: Dict[str, Dict[str, List[Any]]]) -> Dict[str, int]:
    """
    Évalue une solution donnée (plannings par utilisateur) et retourne le score total par utilisateur.

    user_plans : dict uid -> dict sid -> list of (Observation, t_start)
    """
    scores = {u.uid: 0 for u in instance.users}
    for uid, plan in user_plans.items():
        total_reward = 0
        for sid, observations in plan.items():
            for obs, t_start in observations:
                total_reward += obs.reward
        scores[uid] = total_reward
    return scores

from typing import Dict, List, Tuple, Any


def estRealisable(instance: ESOPInstance,
                  user_plans: Dict[str, Dict[str, List[Tuple[Observation, int]]]]) -> bool:
    """
    Vérifie si un ensemble de plannings user_plans est réalisable pour l'instance donnée.

    user_plans : dict uid -> dict sid -> list of (Observation, t_start)
    Retourne True si toutes les contraintes simples sont respectées, False sinon.
    """

    ok = True

    # Index utiles
    sats_by_id = {s.sid: s for s in instance.satellites}
    users_by_id = {u.uid: u for u in instance.users}

    # 1) Unicité : une observation ne doit pas être planifiée plus d'une fois dans tout le système
    used_obs = set()
    for uid, plan in user_plans.items():
        for sid, obs_list in plan.items():
            for obs, t_start in obs_list:
                if obs in used_obs:
                    print(f"[ERREUR] Observation {obs.oid} planifiée plusieurs fois.")
                    ok = False
                used_obs.add(obs)

    # 2) Contraintes par utilisateur / satellite
    for uid, plan in user_plans.items():
        user = users_by_id.get(uid)
        if user is None:
            print(f"[ERREUR] Utilisateur inconnu: {uid}")
            ok = False
            continue

        for sid, obs_list in plan.items():
            sat = sats_by_id.get(sid)
            if sat is None:
                print(f"[ERREUR] Satellite inconnu: {sid}")
                ok = False
                continue

            # Capacité
            if len(obs_list) > sat.capacity:
                print(f"[ERREUR] Capacité dépassée sur {sid} pour {uid}: "
                      f"{len(obs_list)} > {sat.capacity}")
                ok = False

            # Tri par temps de début
            obs_list_sorted = sorted(obs_list, key=lambda p: p[1])

            # Vérité fenêtre observation + horizon satellite
            for obs, t_start in obs_list_sorted:
                t_end = t_start + obs.duration

                if t_start < sat.t_start or t_end > sat.t_end:
                    print(f"[ERREUR] {uid} / {sid} : {obs.oid} sort de l'horizon du satellite "
                          f"[{sat.t_start},{sat.t_end}] avec [{t_start},{t_end}].")
                    ok = False

                if t_start < obs.t_start or t_end > obs.t_end:
                    print(f"[ERREUR] {uid} / {sid} : {obs.oid} hors de sa fenêtre "
                          f"[{obs.t_start},{obs.t_end}] avec [{t_start},{t_end}].")
                    ok = False

            # Non-chevauchement + temps de transition
            tau = sat.transition_time
            for (obs1, t1), (obs2, t2) in zip(obs_list_sorted, obs_list_sorted[1:]):
                end1 = t1 + obs1.duration
                # on impose end1 + tau <= t2
                if end1 + tau > t2:
                    print(f"[ERREUR] {uid} / {sid} : transition insuffisante entre "
                          f"{obs1.oid} [{t1},{end1}] et {obs2.oid} commençant à t={t2} "
                          f"(tau={tau}).")
                    ok = False

    # 3) Fenêtres d'exclusivité des utilisateurs exclusifs (uid != "u0")
    for uid, plan in user_plans.items():
        if uid == "u0":
            continue  # le central n'a pas d'exclusives

        user = users_by_id.get(uid)
        if user is None:
            continue

        for sid, obs_list in plan.items():
            # Toutes les obs planifiées par cet utilisateur sur ce sat
            # doivent être dans au moins une de ses exclusive_windows
            for obs, t_start in obs_list:
                t_end = t_start + obs.duration
                in_excl = any(
                    (w.satellite == sid and
                     t_start >= w.t_start and
                     t_end <= w.t_end)
                    for w in user.exclusive_windows
                )
                if not in_excl:
                    print(f"[ERREUR] {uid} / {sid} : {obs.oid} planifiée en dehors "
                          f"de toute fenêtre d'exclusivité de {uid}.")
                    ok = False

    if ok:
        print("[OK] Le planning est réalisable selon les contraintes vérifiées.")
    else:
        print("[ECHEC] Le planning viole au moins une contrainte.")

    return ok