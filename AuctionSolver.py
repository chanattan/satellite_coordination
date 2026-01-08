from ESOPInstance import ESOPInstance, User, Task, Observation
from GreedySolver import greedy_schedule
from typing import List, Dict, Tuple, Optional, Any

# =============================================================================
# FONCTIONS UTILITAIRES POUR TRANSITIONS
# =============================================================================

def check_transition_feasibility_global(
    merged_plan: Dict[str, List[Tuple[Observation, int]]],
    sat: Any, # Objet Satellite
    new_obs: Observation,
    new_start: int
) -> bool:
    """
    Vérifie si une observation respecte les transitions sur un satellite donné
    en tenant compte du plan déjà établi (merged_plan).
    Utilisé pour la fusion finale du plan u0.
    """
    sid = sat.sid
    tau = sat.transition_time 
    
    current_ops = merged_plan.get(sid, [])
    if not current_ops:
        return True
        
    new_end = new_start + new_obs.duration

    # On vérifie les conflits avec toutes les opérations existantes sur ce satellite
    for (existing_obs, existing_start) in current_ops:
        existing_end = existing_start + existing_obs.duration
        
        # 1. Chevauchement temporel strict
        if not (new_end <= existing_start or new_start >= existing_end):
            return False
            
        # 2. Temps de transition (avant ou après)
        # Si new est avant existing
        if new_end <= existing_start:
            if new_end + tau > existing_start:
                return False
        # Si new est après existing
        if existing_end <= new_start:
            if existing_end + tau > new_start:
                return False
                
    return True


# =============================================================================
# AGENT D'ENCHÈRES
# =============================================================================

class AuctionAgent:
    """
    Agent responsable de calculer les bids et gérer le plan local d'un utilisateur exclusif.
    """

    def __init__(self, user: User, instance: ESOPInstance):
        self.user = user
        self.instance = instance
        # Plan courant : {sat_id: [(obs, t_start), ...]}
        self.plan: Dict[str, List[Tuple[Observation, int]]] = {}

    def _restrict_obs_to_exclusive(self, obs: Observation) -> Optional[Observation]:
        """
        [CORRECTION CRITIQUE] 
        Crée une copie de l'observation dont la fenêtre [t_start, t_end] est 
        l'intersection stricte entre la fenêtre originale et les fenêtres 
        exclusives de l'utilisateur sur ce satellite.
        
        Empêche physiquement le GreedySolver de planifier hors des zones réservées.
        """
        valid_intervals = []
        for win in self.user.exclusive_windows:
            if win.satellite == obs.satellite:
                # Intersection mathématique des intervalles
                start = max(obs.t_start, win.t_start)
                end = min(obs.t_end, win.t_end)
                
                # Vérifie si l'observation rentre dans l'intersection
                if start + obs.duration <= end:
                    valid_intervals.append((start, end))
        
        if not valid_intervals:
            return None
            
        # On prend la première fenêtre valide (simplification)
        best_start, best_end = valid_intervals[0]
        
        # On renvoie une NOUVELLE instance d'Observation modifiée
        new_obs = Observation(
            oid=obs.oid, 
            task_id=obs.task_id, 
            satellite=obs.satellite, 
            t_start=best_start,   # Fenêtre restreinte
            t_end=best_end,       # Fenêtre restreinte
            duration=obs.duration, 
            reward=obs.reward, 
            owner=self.user.uid
        )
        return new_obs

    def _create_local_instance(self, extra_task: Task = None) -> ESOPInstance:
        """
        Construit une sous-instance P[u] contenant :
        1. Les tâches déjà possédées par l'utilisateur.
        2. (Optionnel) Une tâche supplémentaire 'extra_task' (pour le calcul du bid).
        3. LES OBSERVATIONS RESTREINTES (CLIPPÉES) correspondantes.
        """
        # 1. Identifier les tâches à considérer
        tasks_to_consider = [t for t in self.instance.tasks if t.owner == self.user.uid]
        if extra_task and extra_task not in tasks_to_consider:
            tasks_to_consider.append(extra_task)
            
        # 2. Rassembler et CLIPPER toutes les opportunités liées à ces tâches
        restricted_obs = []
        for task in tasks_to_consider:
            for opp in task.opportunities:
                # On ne garde que ce qui rentre dans nos fenêtres exclusives
                clipped_opp = self._restrict_obs_to_exclusive(opp)
                if clipped_opp:
                    restricted_obs.append(clipped_opp)
        
        # 3. Créer l'instance ESOP temporaire
        # Note : On met nb_users=1 car c'est une résolution locale
        inst_u = ESOPInstance(
            nb_satellites=self.instance.nb_satellites,
            nb_users=1,
            nb_tasks=len(tasks_to_consider),
            horizon=self.instance.horizon,
            satellites=self.instance.satellites,
            users=[self.user],
            tasks=tasks_to_consider,
            observations=restricted_obs # On passe la liste filtrée
        )
        return inst_u

    def solve_local(self) -> None:
        """
        Met à jour le plan local self.plan en résolvant P[u] avec les contraintes exclusives.
        """
        inst_u = self._create_local_instance()
        # Appel au GreedySolver générique
        full_plan = greedy_schedule(inst_u)
        self.plan = full_plan.get(self.user.uid, {})

    def calculate_bid(self, task: Task) -> Tuple[float, Optional[Observation]]:
        """
        Calcule le gain marginal si on ajoutait 'task' au plan courant.
        bid = Score(Plan + Task) - Score(Plan Actuel)
        """
        current_reward = sum(obs.reward for obs_list in self.plan.values() for obs, _ in obs_list)
        
        # Création d'une instance temporaire incluant la nouvelle tâche
        # Les observations seront automatiquement clippées par _create_local_instance
        temp_inst = self._create_local_instance(extra_task=task)
        
        # Résolution gloutonne sur l'instance temporaire
        temp_plans = greedy_schedule(temp_inst)
        temp_user_plan = temp_plans.get(self.user.uid, {})
        
        # Calcul du nouveau score
        new_reward = sum(obs.reward for obs_list in temp_user_plan.values() for obs, _ in obs_list)
        
        marginal_gain = new_reward - current_reward
        
        # Si gain positif, on doit identifier quelle observation a permis de satisfaire la tâche
        # (Utile pour PSI/SSI pour savoir où ça a été placé)
        if marginal_gain > 0:
            scheduled_obs = None
            for obs_list in temp_user_plan.values():
                for obs, _ in obs_list:
                    if obs.task_id == task.tid:
                        scheduled_obs = obs
                        break
                if scheduled_obs: break
            
            if scheduled_obs:
                return marginal_gain, scheduled_obs
                
        return -1.0, None

    def add_observation(self, obs: Observation) -> bool:
        """
        Valide l'ajout d'une tâche/observation suite à une enchère gagnée.
        Dans PSI/SSI, task.owner est mis à jour à l'extérieur.
        Ici, on relance solve_local() pour officialiser le nouveau plan.
        """
        # Note : obs est l'observation restreinte retournée par calculate_bid.
        # Pour être robuste, on fait confiance à 'solve_local' pour tout recalculer
        # proprement basé sur le fait que self.user possède maintenant la tâche.
        
        # On sauvegarde l'état actuel pour rollback si besoin (optionnel dans greedy mais prudent)
        old_plan = self.plan
        
        self.solve_local()
        
        # Vérification : est-ce que la tâche est bien dans le nouveau plan ?
        # (Elle devrait l'être car calculate_bid a dit que c'était possible)
        is_scheduled = False
        for obs_list in self.plan.values():
            for o, _ in obs_list:
                if o.task_id == obs.task_id:
                    is_scheduled = True
                    break
        
        if is_scheduled:
            return True
        else:
            # Cas rare : conflit d'ordre greedy imprévu (ex: en PSI parallèle)
            self.plan = old_plan
            return False

# =============================================================================
# ALGORITHMES PSI et SSI
# =============================================================================

def solve_psi(instance: ESOPInstance) -> Dict[str, Dict[str, List[Tuple[Observation, int]]]]:
    """
    Algorithme 2 : Parallel Single-Item Auction.
    """
    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    agents = {u.uid: AuctionAgent(u, instance) for u in exclusive_users}
    global_plan = {u.uid: {} for u in instance.users}

    # 1. Résolution locale initiale
    for agent in agents.values():
        agent.solve_local()

    bids_registry = [] 
    central_tasks = [t for t in instance.tasks if t.owner == "u0"]

    # 2. Bidding parallèle
    for task in central_tasks:
        for uid, agent in agents.items():
            bid_val, best_obs = agent.calculate_bid(task)
            if best_obs and bid_val > 0:
                bids_registry.append((bid_val, best_obs, uid, task))

    # 3. Winner Determination (Tri par meilleure offre globale)
    bids_registry.sort(key=lambda x: x[0], reverse=True)
    
    assigned_task_ids = set()
    
    for bid_val, obs, winner_id, task in bids_registry:
        if task.tid in assigned_task_ids:
            continue
            
        # Changement de propriétaire PROVISOIRE pour l'agent
        # L'agent recalculera son plan dans add_observation
        original_owner = task.owner
        task.owner = winner_id 
        
        if agents[winner_id].add_observation(obs):
            # Succès confirmé
            assigned_task_ids.add(task.tid)
            # obs.owner a été géré par la recréation dans l'agent, 
            # mais on s'assure que la tâche reste au winner.
        else:
            # Echec (conflit PSI), on remet le owner à u0
            task.owner = original_owner

    # 4. Construction du plan global
    for uid, agent in agents.items():
        global_plan[uid] = agent.plan

    # 5. Fusion et planification du reste pour u0
    merged_exclusive_plan = {s.sid: [] for s in instance.satellites}
    for agent in agents.values():
        for sid, obs_list in agent.plan.items():
            if sid in merged_exclusive_plan:
                merged_exclusive_plan[sid].extend(obs_list)
    
    global_plan["u0"] = greedy_plan_for_u0(instance, merged_exclusive_plan)
    
    return global_plan


def solve_ssi(instance: ESOPInstance) -> Dict[str, Dict[str, List[Tuple[Observation, int]]]]:
    """
    Algorithme 3 : Sequential Single-Item Auction.
    """
    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    agents = {u.uid: AuctionAgent(u, instance) for u in exclusive_users}
    global_plan = {u.uid: {} for u in instance.users}

    # 1. Résolution locale initiale
    for agent in agents.values():
        agent.solve_local()

    central_tasks = [t for t in instance.tasks if t.owner == "u0"]
    # Tri par temps de début (heuristique courante SSI)
    central_tasks.sort(key=lambda t: t.t_start)

    # 2. Boucle séquentielle
    for task in central_tasks:
        best_bid = -1.0
        best_obs = None
        winner_id = None
        
        # Enchères
        for uid, agent in agents.items():
            bid_val, obs = agent.calculate_bid(task)
            if obs and bid_val > best_bid:
                best_bid = bid_val
                best_obs = obs
                winner_id = uid
        
        # Allocation
        if winner_id and best_obs and best_bid > 0:
            task.owner = winner_id # Transfert de propriété
            if not agents[winner_id].add_observation(best_obs):
                # Si échec (ne devrait pas arriver en SSI pur), rollback
                task.owner = "u0"

    # 3. Construction résultat
    for uid, agent in agents.items():
        global_plan[uid] = agent.plan

    merged_exclusive_plan = {s.sid: [] for s in instance.satellites}
    for agent in agents.values():
        for sid, obs_list in agent.plan.items():
            if sid in merged_exclusive_plan:
                merged_exclusive_plan[sid].extend(obs_list)
                
    global_plan["u0"] = greedy_plan_for_u0(instance, merged_exclusive_plan)
    
    return global_plan

def solve_ssi_regret(instance: ESOPInstance) -> Dict[str, Dict[str, List[Tuple[Observation, int]]]]:
    """
    Enchères séquentielles basées sur le regret (extension de SSI).

    Idée :
      - On parcourt les requêtes de u0 séquentiellement (comme SSI).
      - Chaque agent calcule un bid brut via calculate_bid(task).
      - Chaque agent garde un "regret" (meilleur gain manqué jusqu'ici).
      - Les bids sont ajustés : bid_ajusté = bid_brut + regret[uid].
      - Si un agent gagne, son regret est légèrement réduit (il "rattrape" son retard).
      - Si une tâche n'est attribuée à personne, on met à jour le regret de ceux
        qui avaient un bid brut positif (ils ont manqué une opportunité).

    Cela favorise les agents qui ont souvent eu de bons bids mais n'ont pas gagné.
    """

    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    agents = {u.uid: AuctionAgent(u, instance) for u in exclusive_users}
    global_plan = {u.uid: {} for u in instance.users}

    # 1. Résolution locale initiale (comme SSI)
    for agent in agents.values():
        agent.solve_local()

    # Regret initial pour chaque agent
    regret: Dict[str, float] = {u.uid: 0.0 for u in exclusive_users}

    central_tasks = [t for t in instance.tasks if t.owner == "u0"]
    # ordre séquentiel (comme SSI)
    central_tasks.sort(key=lambda t: t.t_start)

    # 2. Boucle séquentielle avec regret
    for task in central_tasks:
        best_adjusted_bid = -1.0
        best_raw_bid = -1.0
        best_obs = None
        winner_id = None

        # On stocke aussi les bids bruts pour mettre à jour les regrets des perdants
        raw_bids: Dict[str, float] = {}

        # Enchères
        for uid, agent in agents.items():
            raw_bid, obs = agent.calculate_bid(task)
            raw_bids[uid] = raw_bid

            if obs is None or raw_bid <= 0:
                continue

            adjusted_bid = raw_bid + regret[uid]
            if adjusted_bid > best_adjusted_bid:
                best_adjusted_bid = adjusted_bid
                best_raw_bid = raw_bid
                best_obs = obs
                winner_id = uid

        # Allocation si quelqu'un gagne avec un bid ajusté positif
        if winner_id is not None and best_obs is not None and best_adjusted_bid > 0:
            original_owner = task.owner
            task.owner = winner_id  # transfert de propriété

            if agents[winner_id].add_observation(best_obs):
                # Succès : l'agent a enfin gagné quelque chose, on réduit son regret
                # (par ex. moitié du regret, ou "on consomme" la partie correspondante au raw_bid)
                regret[winner_id] = max(0.0, regret[winner_id] - best_raw_bid)
            else:
                # Si l'insertion échoue, rollback comme en SSI
                task.owner = original_owner

        else:
            # Aucun gagnant : tous les agents avec un bid brut positif "ratent" une opportunité
            for uid, raw_bid in raw_bids.items():
                if raw_bid > 0:
                    regret[uid] = max(regret[uid], raw_bid)

    # 3. Construction du plan global
    for uid, agent in agents.items():
        global_plan[uid] = agent.plan

    # Fusion des plans exclusifs pour construire un plan de base avant d'ajouter u0
    merged_exclusive_plan = {s.sid: [] for s in instance.satellites}
    for agent in agents.values():
        for sid, obs_list in agent.plan.items():
            if sid in merged_exclusive_plan:
                merged_exclusive_plan[sid].extend(obs_list)

    global_plan["u0"] = greedy_plan_for_u0(instance, merged_exclusive_plan)

    return global_plan


# =============================================================================
# PLANIFICATION U0 (FINALE)
# =============================================================================

def greedy_plan_for_u0(instance: ESOPInstance, existing_plan: Dict[str, List[Tuple[Observation, int]]]) -> Dict[str, List[Tuple[Observation, int]]]:
    """
    Planifie les tâches restantes de u0 dans les trous laissés par les exclusifs.
    """
    u0_plan = {s.sid: [] for s in instance.satellites}
    
    remaining_tasks = [t for t in instance.tasks if t.owner == "u0"]
    remaining_obs = []
    
    # On récupère toutes les opportunités u0 possibles
    for t in remaining_tasks:
        for opp in t.opportunities:
            if opp.owner == "u0":
                remaining_obs.append(opp)
    
    if not remaining_obs:
        return u0_plan
    
    # Instance u0 isolée
    u0_user = next(u for u in instance.users if u.uid == "u0")
    u0_inst = ESOPInstance(
        nb_satellites=instance.nb_satellites,
        nb_users=0,
        nb_tasks=len(remaining_tasks),
        horizon=instance.horizon,
        satellites=instance.satellites,
        users=[u0_user],
        tasks=remaining_tasks,
        observations=remaining_obs
    )
    
    # Planification idéale u0
    raw_u0_plan = greedy_schedule(u0_inst).get("u0", {})
    
    sat_map = {s.sid: s for s in instance.satellites}
    
    # Intégration en respectant les contraintes globales
    for sid, obs_list in raw_u0_plan.items():
        if sid not in sat_map: continue
        sat = sat_map[sid]
        
        sorted_u0_obs = sorted(obs_list, key=lambda x: x[1])
        
        for obs, start_time in sorted_u0_obs:
            # Vérif Capacité
            current_load = len(existing_plan.get(sid, [])) + len(u0_plan[sid])
            if current_load >= sat.capacity:
                continue 
            
            # Vérif Transitions (Global)
            temp_combined_plan = {
                sid: existing_plan.get(sid, []) + u0_plan[sid]
            }
            
            if check_transition_feasibility_global(temp_combined_plan, sat, obs, start_time):
                u0_plan[sid].append((obs, start_time))
                
    return u0_plan

