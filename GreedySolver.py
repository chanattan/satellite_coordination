from typing import List, Tuple, Dict
from ESOPInstance import *

def greedy_schedule_for_user_on_satellite(instance: ESOPInstance, user_id: str, satellite_id: str):
    """
    Planifie de manière gloutonne (cf. article) les observations de l'utilisateur user_id sur le satellite satellite_id, dans l'instance associée.

    Retourne une liste [(obs, t_start_assigné), ...] triée par temps.
    """

    # on filtre les observations de cet utilisateur et de ce satellite
    candidate_obs = [o for o in instance.observations
                     if o.owner == user_id and o.satellite == satellite_id]

    sat = next(s for s in instance.satellites if s.sid == satellite_id)
    cap = sat.capacity
    tau = sat.transition_time
    sat_start = sat.t_start
    sat_end = sat.t_end

    # Trier par priorité puis fenêtre de début (ici on approxime la priorité par la reward décroissante et la date de début croissante)
    candidate_obs.sort(key=lambda o: (-o.reward, o.t_start))

    # planning en construction : liste [(obs, t_start), ...] triée par t_start
    plan = []
    def used_capacity():
        return len(plan)

    # fonction utilitaire : essayer d'insérer une observation dans le plan
    def try_insert(obs: Observation):
        """
        Tente de trouver le premier créneau valide pour obs.
        Retourne le t_start choisi ou None si pas possible.
        """

        if used_capacity() >= cap:
            return None

        # Cas où le plan est vide
        if not plan:
            # On peut commencer au max entre fenêtre de l'obs et début du satellite
            t0 = max(obs.t_start, sat_start)
            if t0 + obs.duration <= min(obs.t_end, sat_end):
                return t0
            return None

        # Cas général : on regarde les trous entre observations déjà planifiées
        # On construit une liste étendue avec bornes début / fin fictives pour simplifier la recherche des trous.
        # Plan est trié par t_start.
        # On va tester :
        #  - un trou avant la première observation
        #  - des trous entre observations successives
        #  - un trou après la dernière
        sorted_plan = sorted(plan, key=lambda p: p[1])

        # Trou avant la première obs planifiée
        first_obs, first_t = sorted_plan[0]
        earliest_start = max(obs.t_start, sat_start)
        latest_end = min(obs.t_end, first_t - tau)  # laisser la transition avant la première
        if earliest_start + obs.duration <= latest_end:
            return earliest_start

        # Trous entre obs successives
        for (o_prev, t_prev), (o_next, t_next) in zip(sorted_plan, sorted_plan[1:]):
            # Fin de o_prev + durée + tau
            window_start = max(
                obs.t_start,
                t_prev + o_prev.duration + tau
            )
            # On doit aussi respecter la transition de obs vers o_next
            window_end = min(
                obs.t_end,
                t_next - tau  # début de la suivante moins tau
            )
            if window_start + obs.duration <= window_end:
                return window_start

        # Trou après la dernière obs
        last_obs, last_t = sorted_plan[-1]
        window_start = max(
            obs.t_start,
            last_t + last_obs.duration + tau
        )
        window_end = min(
            obs.t_end,
            sat_end
        )
        if window_start + obs.duration <= window_end:
            return window_start

        # Pas de créneau valide trouvé
        return None

    # parcours glouton des observations candidates
    for obs in candidate_obs:
        t_insert = try_insert(obs)
        if t_insert is not None:
            plan.append((obs, t_insert))

    plan.sort(key=lambda p: p[1])
    return plan

def greedy_schedule_for_user(instance: ESOPInstance, user_id: str):
    full_plan = {}
    for sat in instance.satellites:
        full_plan[sat.sid] = greedy_schedule_for_user_on_satellite(
            instance, user_id, sat.sid
        )
    return full_plan
