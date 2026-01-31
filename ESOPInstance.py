class Satellite():
    def __init__(self, sid, t_start, t_end, capacity, transition_time=None):
        self.sid = sid
        self.t_start = t_start
        self.t_end = t_end
        self.capacity = capacity
        self.transition_time = transition_time if transition_time is not None else 1 # article : tau constant

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
    def __init__(self, nb_satellites, nb_users, nb_tasks, horizon, satellites, users, tasks, observations):
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

    def to_text(self):
        lines = []
        lines.append("[Parameters]")
        lines.append(f"Satellites : {self.nb_satellites}")
        lines.append(f"Exclusive users : {self.nb_users}")
        lines.append(f"Tasks : {self.nb_tasks}")
        lines.append("")
        lines.append("[Satellites]")
        for sat in self.satellites:
            lines.append(f"{sat.sid} {sat.t_start} {sat.t_end} {sat.capacity} {sat.transition_time}")
        lines.append("")
        lines.append("[Users]")
        for u in self.users:
            win_strs = [f"{w.satellite}:{w.t_start}-{w.t_end}" for w in u.exclusive_windows]
            wins = ", ".join(win_strs) if win_strs else "-"
            lines.append(f"{u.uid} {wins}")
        lines.append("")
        lines.append("[Tasks]")
        for task in self.tasks:
            lines.append(f"{task.tid} owner={task.owner} window=[{task.t_start},{task.t_end}] duration={task.duration} reward={task.reward}")
        lines.append("")
        lines.append("[Observations]")
        for o in self.observations:
            lines.append(f"{o.oid} task={o.task_id} owner={o.owner} sat={o.satellite} window=[{o.t_start},{o.t_end}] duration={o.duration} reward={o.reward}")
        return "\n".join(lines)

def assess_solution(instance, user_plans):
    """
        Évalue une solution donnée (plannings par utilisateur) et retourne le score total par utilisateur.
    """
    scores = {u.uid: 0 for u in instance.users}
    for uid, plan in user_plans.items():
        total_reward = 0
        for sid, observations in plan.items():
            for obs, t_start in observations:
                total_reward += obs.reward
        scores[uid] = total_reward
    return scores

def estRealisable(instance, user_plans):
    """
        Vérifie si user_plans est réalisable pour l'instance donnée.
    """

    ok = True

    sats_by_id = {s.sid: s for s in instance.satellites}
    users_by_id = {u.uid: u for u in instance.users}

    # vérification de l'unicité des observations + au plus une obs par requête
    used_obs = set()
    used_tasks = set()
    for uid, plan in user_plans.items():
        for sid, obs_list in plan.items():
            for obs, t_start in obs_list:
                if obs in used_obs:
                    print(f"[ERREUR] Observation {obs.oid} planifiée plusieurs fois.")
                    ok = False
                used_obs.add(obs)

                # au plus 1 observation par requête / tâche
                if obs.task_id in used_tasks:
                    print(f"[ERREUR] Task {obs.task_id} satisfaite par plusieurs observations.")
                    ok = False
                used_tasks.add(obs.task_id)

    # capacité globale, délais, fenêtres et transitions, par satellite
    for sid, sat in sats_by_id.items():
        # rassembler toutes les obs planifiées sur ce sat (tous utilisateurs confondus)
        all_obs_on_sat = [(obs, t_start, uid) for uid, plan in user_plans.items() for s2, obs_list in plan.items() if s2 == sid for (obs, t_start) in obs_list]

        # capacité globale
        if len(all_obs_on_sat) > sat.capacity:
            print(f"[ERREUR] Capacité globale dépassée sur {sid}: "
                  f"{len(all_obs_on_sat)} > {sat.capacity}")
            ok = False

        all_obs_on_sat.sort(key=lambda p: p[1]) # tri par temps début

        # fenêtres / horizon + transitions globales
        tau = sat.transition_time
        for i, (obs, t_start, uid) in enumerate(all_obs_on_sat):
            t_end = t_start + obs.duration

            # horizon satellite
            if t_start < sat.t_start or t_end > sat.t_end:
                print(f"[ERREUR] {uid} / {sid} : {obs.oid} sort de l'horizon sat [{sat.t_start},{sat.t_end}] avec [{t_start},{t_end}].")
                ok = False

            # fenêtre observation
            if t_start < obs.t_start or t_end > obs.t_end:
                print(f"[ERREUR] {uid} / {sid} : {obs.oid} hors de sa fenêtre [{obs.t_start},{obs.t_end}] avec [{t_start},{t_end}].")
                ok = False

            if i < len(all_obs_on_sat) - 1: # transition avec l'observation suivante (globale)
                obs2, t2, uid2 = all_obs_on_sat[i + 1]
                end1 = t_end
                if end1 + tau > t2:
                    print(f"[ERREUR] Transition insuffisante sur {sid} entre {obs.oid} ({uid}) [{t_start},{end1}] et {obs2.oid} ({uid2}) commençant à t={t2} (tau={tau}).")
                    ok = False

    # Fenêtres d'exclusivité pour utilisateurs exclusifs (uid != "u0")
    for uid, plan in user_plans.items():
        if uid == "u0":
            continue  # le central peut utiliser les portions non exclusives

        user = users_by_id.get(uid)
        if user is None:
            print(f"[ERREUR] Utilisateur inconnu (dans exclusifs) : {uid}")
            ok = False
            continue

        for sid, obs_list in plan.items():
            for obs, t_start in obs_list:
                t_end = t_start + obs.duration
                in_excl = any((w.satellite == sid and t_start >= w.t_start and t_end <= w.t_end) for w in user.exclusive_windows)
                if not in_excl:
                    print(f"[ERREUR] {uid} / {sid} : {obs.oid} planifiée en dehors de toute fenêtre d'exclusivité de {uid}.")
                    ok = False

    if ok:
        print("[OK] Le planning est réalisable selon les contraintes vérifiées.")
    else:
        print("[ECHEC] Le planning viole au moins une contrainte.")
    return ok
