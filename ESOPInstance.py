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
