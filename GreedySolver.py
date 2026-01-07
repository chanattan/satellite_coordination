from ESOPInstance import ESOPInstance, Observation, Task


def greedy_schedule_P_u(instance: ESOPInstance, user_id: str):
    """
    Résout P_u avec l'algorithme glouton pour un utilisateur donné :

    P_u = <S, U, R_u, O_u>
      - S : mêmes satellites que l'instance globale
      - U : on peut garder tous les users ou seulement u (ici on garde juste u, car
            les autres n'interviennent pas dans le solve local)
      - R_u : tâches dont le owner est u
      - O_u : observations dont le owner est u

    Retourne un plan {sat_id: [(obs, t_start), ...]} pour user_id.
    """

    # Utilisateur u
    u = next(user for user in instance.users if user.uid == user_id)

    # R_u : tâches de u
    tasks_u: list[Task] = [t for t in instance.tasks if t.owner == user_id]

    # O_u : observations de u
    obs_u: list[Observation] = [o for o in instance.observations if o.owner == user_id]

    # Sous-instance P_u (mêmes satellites et horizon)
    inst_u = ESOPInstance(
        nb_satellites=instance.nb_satellites,
        nb_users=1,  # nombre d'utilisateurs exclusifs dans cette sous-instance (ici juste u)
        nb_tasks=len(tasks_u),
        horizon=instance.horizon,
        satellites=instance.satellites,
        users=[u],          # on ne garde que u ici, suffisant pour le solve local
        tasks=tasks_u,
        observations=obs_u,
    )
    all_plans_u = greedy_schedule(inst_u)

    # On ne récupère que le plan de u
    return all_plans_u.get(user_id, {})


def greedy_schedule(instance: ESOPInstance):
    """
        Implémentation de l'algorithme 1 (greedy EOSCSP solver) de l'article :
        - tri global des observations O (reward décroissant, puis t_start croissant),
        - structure Rs par satellite : liste triée (obs, t_start),
        - fonction first_slot(o, P, Rs) avec parcours des intervalles,
        - au plus une observation par requête/tâche.
        Retour :
        user_plans[uid][sid] = [(obs, t_start), ...]
    """

    # Résultat final par utilisateur et satellite
    user_plans = {}

    # Rs : pour chaque satellite s, liste triée [(obs, t_start)] déjà planifiées sur s
    Rs = {sat.sid: [] for sat in instance.satellites}
    sat_by_id = {s.sid: s for s in instance.satellites}

    # Tri global des observations (remplace Osorted de l'article)
    Osorted = sorted(instance.observations, key=lambda o: (-o.reward, o.t_start))
    # Tâches déjà satisfaites (au plus une observation par requête)
    tasks_satisfied = set()

    def first_slot(o: Observation):
        """
        Version fidèle de first_slot(o, P, Rs) avec gestion de t_upper.
        On parcourt les "domaines" possibles de l'observation sur son satellite
        et on insère o à la position i dans Rs[s] si un créneau valide existe.
        """
        s = sat_by_id[o.satellite]
        sid = s.sid

        tstart_s = s.t_start
        tend_s = s.t_end
        cap = s.capacity
        tau = s.transition_time

        # Capacité déjà atteinte ?
        if len(Rs[sid]) >= cap:
            return None

        plan_s = Rs[sid]  # déjà trié par t_start

        # Si le satellite n'a encore rien planifié, on regarde le segment [tstart_s, tend_s]
        if not plan_s:
            # t_start candidat = max(horizon sat, fenêtre obs)
            t0 = max(tstart_s, o.t_start)
            if t0 + o.duration <= min(tend_s, o.t_end):
                # pas de conflit ni transition à gérer, on insère à la fin
                plan_s.append((o, t0))
                return t0
            return None

        # avant la première observation
        # On considère le domaine [tstart_s, t_end_first - tau]
        first_obs, first_t = plan_s[0]
        t_end_first = first_t
        # borne supérieure de ce domaine
        t_upper = t_end_first
        # fenêtre disponible = [tstart_s, t_upper - tau] pour le début d'o
        if tstart_s < t_upper:
            t0 = max(tstart_s, o.t_start)
            if t0 + o.duration <= min(o.t_end, t_upper - tau, tend_s):
                # insert en tête (i = 0)
                plan_s.insert(0, (o, t0))
                return t0

        # Domaines entre les observations successives
        # plan_s est trié, on regarde les intervalles entre (oi, ti) et (oi+1, ti+1)
        for i in range(len(plan_s) - 1):
            o_i, t_i = plan_s[i]
            o_ip1, t_ip1 = plan_s[i + 1]

            # début du domaine = fin de o_i + tau
            domain_start = t_i + o_i.duration + tau
            # t_upper = début de o_{i+1}
            t_upper = t_ip1

            if domain_start < t_upper - tau:
                # fenêtre possible pour le début de o
                t0 = max(domain_start, o.t_start)
                if t0 + o.duration <= min(o.t_end, t_upper - tau, tend_s):
                    # insérer o à la position i+1 (juste après o_i)
                    plan_s.insert(i + 1, (o, t0))
                    return t0

        # Domaine après la dernière observation
        o_last, t_last = plan_s[-1]
        domain_start = t_last + o_last.duration + tau
        # dernier domaine = [domain_start, tend_s]
        if domain_start < tend_s:
            t0 = max(domain_start, o.t_start)
            if t0 + o.duration <= min(o.t_end, tend_s):
                # insérer à la fin
                plan_s.append((o, t0))
                return t0

        # Aucun créneau
        return None

    # boucle principale
    for o in Osorted:
        # au plus une observation par tâche
        if o.task_id in tasks_satisfied:
            continue

        t = first_slot(o)
        if t is None:
            continue

        # marquer la tâche comme satisfaite
        tasks_satisfied.add(o.task_id)

        # remplir user_plans pour évaluation
        user_plans.setdefault(o.owner, {}).setdefault(o.satellite, []).append((o, t))

    # Tri par temps de début dans la sortie
    for uid in user_plans:
        for sid in user_plans[uid]:
            user_plans[uid][sid].sort(key=lambda p: p[1])

    return user_plans
