from typing import List, Tuple, Dict
from ESOPInstance import *


def greedy_schedule_for_user_on_satellite(instance: ESOPInstance, user_id: str, satellite_id: str):
    candidate_obs = [
        o for o in instance.observations
        if o.owner == user_id and o.satellite == satellite_id
    ]

    sat = next(s for s in instance.satellites if s.sid == satellite_id)
    user = next(u for u in instance.users if u.uid == user_id)

    cap = sat.capacity
    tau = sat.transition_time
    sat_start = sat.t_start
    sat_end = sat.t_end

    candidate_obs.sort(key=lambda o: (-o.reward, o.t_start))

    plan: List[Tuple[Observation, int]] = []
    tasks_satisfied = set()

    # fenêtres exclusives pertinentes pour cet utilisateur et ce satellite
    exclusive_windows = []
    if user_id != "u0":
        exclusive_windows = [
            w for w in user.exclusive_windows
            if w.satellite == satellite_id
        ]

    def used_capacity():
        return len(plan)

    def in_exclusive_window(t_start, duration):
        if user_id == "u0":
            return True  # pas de contrainte pour le central
        t_end = t_start + duration
        for w in exclusive_windows:
            if t_start >= w.t_start and t_end <= w.t_end:
                return True
        return False

    def try_insert(obs: Observation):
        if used_capacity() >= cap:
            return None

        # helper local pour tester un candidat t0
        def check_t0(t0):
            if not in_exclusive_window(t0, obs.duration):
                return None
            return t0

        if not plan:
            t0 = max(obs.t_start, sat_start)
            if t0 + obs.duration <= min(obs.t_end, sat_end):
                return check_t0(t0)
            return None

        sorted_plan = sorted(plan, key=lambda p: p[1])

        # 1) trou avant la première obs
        first_obs, first_t = sorted_plan[0]
        earliest_start = max(obs.t_start, sat_start)
        latest_end = min(obs.t_end, first_t - tau)
        if earliest_start + obs.duration <= latest_end:
            return check_t0(earliest_start)

        # 2) trous entre obs successives
        for (o_prev, t_prev), (o_next, t_next) in zip(sorted_plan, sorted_plan[1:]):
            end_prev = t_prev + o_prev.duration
            start_next = t_next

            window_start = max(obs.t_start, end_prev + tau, sat_start)
            window_end = min(obs.t_end, start_next - tau, sat_end)

            if window_start + obs.duration <= window_end:
                return check_t0(window_start)

        # 3) après la dernière obs
        last_obs, last_t = sorted_plan[-1]
        end_last = last_t + last_obs.duration
        window_start = max(obs.t_start, end_last + tau, sat_start)
        window_end = min(obs.t_end, sat_end)
        if window_start + obs.duration <= window_end:
            return check_t0(window_start)

        return None

    for obs in candidate_obs:
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
