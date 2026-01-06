from ESOPInstance import ESOPInstance, User, Task, Observation
from typing import List, Dict, Tuple, Optional


class AuctionAgent:
    """
    Classe wrapper pour gérer la logique d'enchère et de planning d'un utilisateur.
    """
    def __init__(self, user: User, instance: ESOPInstance):
        self.user = user
        self.instance = instance
        self.plan: Dict[str, List[Tuple[Observation, int]]] = {
            s.sid: [] for s in instance.satellites
        }
        self.capacity_usage: Dict[str, int] = {s.sid: 0 for s in instance.satellites}

    def solve_local(self, initial_observations: List[Observation]):
        for obs in initial_observations:
            if obs.owner == self.user.uid:  # ✅ Ajouter cette condition
                success = self.add_observation(obs)
                if not success:
                    print(f"Warning: Impossible d'ajouter l'observation initiale {obs.oid}")


    def _plan_reward(self) -> float:
        """Somme des rewards des observations dans le plan courant."""
        total = 0.0
        for sid, obs_list in self.plan.items():
            for obs, _ in obs_list:
                total += obs.reward
        return total

    def _try_insert_marginal(self, obs: Observation) -> float:
        """
        Calcule le gain marginal de l'observation obs :
        reward(plan + obs) - reward(plan), en utilisant le même greedy local.
        """
        backup_plan = {sid: lst[:] for sid, lst in self.plan.items()}
        backup_cap = self.capacity_usage.copy()
        base_reward = self._plan_reward()

        gain = -1.0
        if self.add_observation(obs):
            gain = self._plan_reward() - base_reward

        # restauration de l'état
        self.plan = backup_plan
        self.capacity_usage = backup_cap
        return gain

    def calculate_bid(self, task: Task) -> Tuple[float, Optional[Observation]]:
        """
        Correspond à 'bid(r, M_u)'.
        Calcule le coût marginal pour une tâche donnée (gain de reward).
        Retourne (valeur_enchere, meilleure_opportunite).
        """
        best_bid_value = -1.0
        best_opportunity = None

        for opp in task.opportunities:
            if not self._is_in_exclusive_window(opp):
                continue
            if not self._is_feasible(opp):
                continue

            # Bid = gain marginal (approximation de solve(P_u ∪ {r}) - solve(P_u))
            gain = self._try_insert_marginal(opp)
            if gain > best_bid_value:
                best_bid_value = gain
                best_opportunity = opp

        return best_bid_value, best_opportunity

    def add_observation(self, obs: Observation) -> bool:
        """
        Tente d'ajouter une observation au plan local (M_w <- M_w + sigma).
        Retourne True si succès.
        """
        if not self._is_feasible(obs):
            return False

        self.plan[obs.satellite].append((obs, obs.t_start))
        self.plan[obs.satellite].sort(key=lambda x: x[1])
        self.capacity_usage[obs.satellite] += 1
        return True

    def _is_in_exclusive_window(self, obs: Observation) -> bool:
        """Vérifie si l'observation tombe dans une fenêtre exclusive de l'agent."""
        for window in self.user.exclusive_windows:
            if window.satellite == obs.satellite:
                if (obs.t_start >= window.t_start) and (obs.t_end <= window.t_end):
                    return True
        return False

    def _is_feasible(self, obs: Observation) -> bool:
        """Vérifie les contraintes physiques sans modifier le plan."""
        sid = obs.satellite
        sat_obj = next(s for s in self.instance.satellites if s.sid == sid)

        if self.capacity_usage[sid] >= sat_obj.capacity:
            return False

        existing_obs_list = self.plan[sid]
        trans_time = sat_obj.transition_time

        for existing_obs, t_start in existing_obs_list:
            # chevauchement temporel
            if not (obs.t_end <= existing_obs.t_start or obs.t_start >= existing_obs.t_end):
                return False

            # temps de transition après existing_obs
            if obs.t_start >= existing_obs.t_end:
                if obs.t_start < existing_obs.t_end + trans_time:
                    return False

            # temps de transition avant existing_obs
            if obs.t_end <= existing_obs.t_start:
                if existing_obs.t_start < obs.t_end + trans_time:
                    return False

        return True

def greedy_plan_for_u0(instance: ESOPInstance,
                       existing_plan: Optional[Dict[str, List[Tuple[Observation, int]]]] = None
                       ) -> Dict[str, List[Tuple[Observation, int]]]:
    """
    Planificateur greedy pour u0 SUR LES CRÉNEAUX NON-EXCLUSIFS UNIQUEMENT.
    """
    if existing_plan is None:
        plan = {s.sid: [] for s in instance.satellites}
    else:
        plan = {sid: lst.copy() for sid, lst in existing_plan.items()}
    capacity_usage = {s.sid: len(plan[s.sid]) for s in instance.satellites}

    def is_feasible_u0(obs: Observation) -> bool:
        sid = obs.satellite
        sat_obj = next(s for s in instance.satellites if s.sid == sid)
        if capacity_usage[sid] >= sat_obj.capacity:
            return False

        #Vérifier que ce n'est PAS dans une fenêtre exclusive d'un AUTRE utilisateur
        for user in instance.users:
            if user.uid != "u0":  # Seulement les exclusifs
                for window in user.exclusive_windows:
                    if (window.satellite == sid and 
                        obs.t_start >= window.t_start and 
                        obs.t_end <= window.t_end):
                        return False  # Violation d'exclusive !

        existing_obs_list = plan[sid]
        trans_time = sat_obj.transition_time

        for existing_obs, t_start in existing_obs_list:
            if not (obs.t_end <= existing_obs.t_start or obs.t_start >= existing_obs.t_end):
                return False
            if obs.t_start >= existing_obs.t_end:
                if obs.t_start < existing_obs.t_end + trans_time:
                    return False
            if obs.t_end <= existing_obs.t_start:
                if existing_obs.t_start < obs.t_end + trans_time:
                    return False
        return True

    remaining_obs: List[Observation] = []
    for task in instance.tasks:
        if task.owner == "u0":
            for opp in task.opportunities:
                if opp.owner == "u0":  # Encore détenues par u0
                    remaining_obs.append(opp)

    remaining_obs.sort(key=lambda o: o.t_start)

    for obs in remaining_obs:
        if is_feasible_u0(obs):  # Maintenant respecte les exclusives
            plan[obs.satellite].append((obs, obs.t_start))
            plan[obs.satellite].sort(key=lambda x: x[1])
            capacity_usage[obs.satellite] += 1

    return plan



def solve_psi(instance: ESOPInstance) -> Dict[str, Dict[str, List[Tuple[Observation, int]]]]:
    """
    Algorithme 2 : Parallel Single-Item (PSI)
    """
    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    agents = {u.uid: AuctionAgent(u, instance) for u in exclusive_users}

    global_plan = {u.uid: {} for u in instance.users}

    # Initialisation des plans locaux avec les observations propres à chaque utilisateur exclusif
    for uid, agent in agents.items():
        own_obs = [o for o in instance.observations if o.owner == uid]
        agent.solve_local(own_obs)

    # PHASE PARALLÈLE : chaque agent calcule ses bids pour toutes les requêtes de u0
    bids_registry = {}  # {task_id: [(bid_value, obs, agent_id)]}

    central_tasks = [t for t in instance.tasks if t.owner == "u0"]

    for task in central_tasks:
        bids_registry[task.tid] = []
        for uid, agent in agents.items():
            bid_val, best_obs = agent.calculate_bid(task)
            if best_obs:
                bids_registry[task.tid].append((bid_val, best_obs, uid))

    # Attribution des tâches à partir des bids calculés en parallèle
    for task in central_tasks:
        offers = bids_registry.get(task.tid, [])

        if not offers:
            continue

        # w <- argmax(B_u[r])
        offers.sort(key=lambda x: x[0], reverse=True)
        best_bid, winning_obs, winner_id = offers[0]

        if best_bid > 0:
            success = agents[winner_id].add_observation(winning_obs)
            if success:
                winning_obs.owner = winner_id
            else:
                # conflit PSI typique (état local différent au moment de l'allocation)
                pass

    # Construction du plan global
    for uid, agent in agents.items():
        global_plan[uid] = agent.plan

    # u0 planifie greedily les observations restantes
    global_plan["u0"] = greedy_plan_for_u0(instance)

    return global_plan


def solve_ssi(instance: ESOPInstance) -> Dict[str, Dict[str, List[Tuple[Observation, int]]]]:
    """
    Algorithme 3 : Sequential Single-Item (SSI)
    """
    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    agents = {u.uid: AuctionAgent(u, instance) for u in exclusive_users}
    global_plan = {u.uid: {} for u in instance.users}

    # Initialisation des plans locaux avec les observations propres à chaque utilisateur exclusif
    for uid, agent in agents.items():
        own_obs = [o for o in instance.observations if o.owner == uid]
        agent.solve_local(own_obs)

    # Tri séquentiel des requêtes de u0
    central_tasks = [t for t in instance.tasks if t.owner == "u0"]
    central_tasks.sort(key=lambda t: t.t_start)

    for task in central_tasks:
        best_global_bid = -1.0
        best_global_obs = None
        winner_id = None

        for uid, agent in agents.items():
            bid_val, obs = agent.calculate_bid(task)

            if bid_val > best_global_bid:
                best_global_bid = bid_val
                best_global_obs = obs
                winner_id = uid

        if winner_id and best_global_bid > 0 and best_global_obs is not None:
            success = agents[winner_id].add_observation(best_global_obs)
            if success:
                best_global_obs.owner = winner_id

    for uid, agent in agents.items():
        global_plan[uid] = agent.plan

    global_plan["u0"] = greedy_plan_for_u0(instance)

    return global_plan


def solve_ssi_regret(instance: ESOPInstance) -> Dict[str, Dict[str, List[Tuple[Observation, int]]]]:
    """
    Enchères séquentielles basées sur le regret (version simplifiée).
    Chaque agent garde une trace du meilleur gain marginal raté
    et ajuste ses bids en conséquence.
    """
    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    agents = {u.uid: AuctionAgent(u, instance) for u in exclusive_users}
    global_plan = {u.uid: {} for u in instance.users}

    # Initialisation des plans locaux avec les observations propres à chaque utilisateur exclusif
    for uid, agent in agents.items():
        own_obs = [o for o in instance.observations if o.owner == uid]
        agent.solve_local(own_obs)

    central_tasks = [t for t in instance.tasks if t.owner == "u0"]
    central_tasks.sort(key=lambda t: t.t_start)

    # mémoire du meilleur gain raté pour chaque agent
    missed_best_gain: Dict[str, float] = {u.uid: 0.0 for u in exclusive_users}

    for task in central_tasks:
        best_adjusted_bid = -1.0
        best_obs = None
        winner_id = None

        # calcul des bids ajustés par "regret" pour la tâche courante
        for uid, agent in agents.items():
            raw_bid, obs = agent.calculate_bid(task)
            if raw_bid <= 0 or obs is None:
                continue

            adjusted_bid = raw_bid - missed_best_gain[uid]

            if adjusted_bid > best_adjusted_bid:
                best_adjusted_bid = adjusted_bid
                best_obs = obs
                winner_id = uid

        if winner_id is not None and best_adjusted_bid > 0 and best_obs is not None:
            success = agents[winner_id].add_observation(best_obs)
            if success:
                best_obs.owner = winner_id
            else:
                # si l'insertion échoue, on considère ce gain comme "raté"
                missed_best_gain[winner_id] = max(missed_best_gain[winner_id], best_adjusted_bid)
        else:
            # si personne ne gagne cette tâche, on met à jour le "meilleur gain raté" pour ceux qui avaient un bid > 0
            for uid, agent in agents.items():
                raw_bid, obs = agent.calculate_bid(task)
                if raw_bid > missed_best_gain[uid]:
                    missed_best_gain[uid] = raw_bid

    for uid, agent in agents.items():
        global_plan[uid] = agent.plan

    global_plan["u0"] = greedy_plan_for_u0(instance)

    return global_plan
    
