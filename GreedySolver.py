from typing import List, Tuple, Dict
from ESOPInstance import *


def greedy_schedule_for_user_on_satellite(instance: ESOPInstance, user_id: str, satellite_id: str):
    """
    Planifie de manière gloutonne (inspiré de l'algorithme 1 de l'article)
    les observations de l'utilisateur user_id sur le satellite satellite_id.

    Retourne une liste [(obs, t_start_assigné), ...] triée par temps.
    """

    candidate_obs = [
        o for o in instance.observations
        if o.owner == user_id and o.satellite == satellite_id
    ]

    sat = next(s for s in instance.satellites if s.sid == satellite_id)
    cap = sat.capacity
    tau = sat.transition_time
    sat_start = sat.t_start
    sat_end = sat.t_end

    # priorité approximée par reward décroissante, puis date de début croissante
    candidate_obs.sort(key=lambda o: (-o.reward, o.t_start))

    # liste [(obs, t_start)]
    plan: List[Tuple[Observation, int]] = []
    # ensemble des tâches déjà satisfaites sur ce satellite
    tasks_satisfied = set()

    def used_capacity():
        return len(plan)

    def try_insert(obs: Observation):
        """
        Tente de trouver le premier créneau valide pour obs.
        Retourne le t_start choisi ou None si pas possible.
        """

        if used_capacity() >= cap:
            return None

        if not plan:
            t0 = max(obs.t_start, sat_start)
            if t0 + obs.duration <= min(obs.t_end, sat_end):
                return t0
            return None

        sorted_plan = sorted(plan, key=lambda p: p[1])

        # 1) Trou avant la première observation planifiée
        first_obs, first_t = sorted_plan[0]
        earliest_start = max(obs.t_start, sat_start)
        latest_end = min(obs.t_end, first_t - tau)
        if earliest_start + obs.duration <= latest_end:
            return earliest_start

        # 2) Trous entre observations successives
        for (o_prev, t_prev), (o_next, t_next) in zip(sorted_plan, sorted_plan[1:]):
            end_prev = t_prev + o_prev.duration
            start_next = t_next

            window_start = max(obs.t_start, end_prev + tau, sat_start)
            window_end = min(obs.t_end, start_next - tau, sat_end)

            if window_start + obs.duration <= window_end:
                return window_start

        # 3) Trou après la dernière observation
        last_obs, last_t = sorted_plan[-1]
        end_last = last_t + last_obs.duration
        window_start = max(obs.t_start, end_last + tau, sat_start)
        window_end = min(obs.t_end, sat_end)
        if window_start + obs.duration <= window_end:
            return window_start

        return None

    for obs in candidate_obs:
        # nouvelle contrainte : au plus une observation par tâche
        if obs.task_id in tasks_satisfied:
            continue

        t_insert = try_insert(obs)
        if t_insert is not None:
            plan.append((obs, t_insert))
            tasks_satisfied.add(obs.task_id)

    plan.sort(key=lambda p: p[1])
    return plan


def greedy_schedule_for_user(instance: ESOPInstance, user_id: str):
    full_plan = {}
    for sat in instance.satellites:
        full_plan[sat.sid] = greedy_schedule_for_user_on_satellite(instance, user_id, sat.sid)
    return full_plan
