from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy
from ESOPInstance import ESOPInstance, Observation, Task, User, Satellite
from GreedySolver import greedy_schedule, greedy_schedule_P_u

# =============================================================================
# FONCTIONS UTILITAIRES (inchangées)
# =============================================================================

def safe_plan_reward(plan: Dict[str, Dict[str, List[Tuple[Observation, int]]]], 
                    user_id: str) -> float:
    """Calcule le reward total d'un plan utilisateur de façon safe"""
    total = 0
    user_plan = plan.get(user_id, {})
    for sat_plan in user_plan.values():
        if isinstance(sat_plan, list):
            total += sum(obs.reward for obs, _ in sat_plan)
    return total

def safe_get_schedule(plan: Dict[str, Dict[str, List[Tuple[Observation, int]]]], 
                     task_id: str) -> Optional[Tuple[Observation, int]]:
    """Extrait le schedule pour une tâche spécifique"""
    for user_plan in plan.values():
        for sat_plan in user_plan.values():
            if isinstance(sat_plan, list):
                for obs_t in sat_plan:
                    if len(obs_t) == 2 and obs_t[0].task_id == task_id:
                        return obs_t
    return None

def create_instance_with_fixed_observations(instance: ESOPInstance, 
                                          fixed_obs: Dict[str, List[Tuple[Observation, int]]]) -> ESOPInstance:
    """Crée une nouvelle instance où certaines observations sont pré-planifiées (P[∅|M])"""
    new_inst = deepcopy(instance)
    fixed_tasks = set()
    for sat_plans in fixed_obs.values():
        for obs, _ in sat_plans:
            fixed_tasks.add(obs.task_id)
    
    new_inst.tasks = [t for t in new_inst.tasks if t.tid not in fixed_tasks]
    new_inst.nb_tasks = len(new_inst.tasks)
    new_inst.observations = [o for o in new_inst.observations if o.task_id not in fixed_tasks]
    
    return new_inst

def bid(u_id: str, instance: ESOPInstance, request: Task, 
        all_user_plans: Dict[str, Dict[str, List[Tuple[Observation, int]]]]) -> Tuple[float, Optional[Tuple[Observation, int]]]:
    """
    bid(r, Mu) : gain marginal pour intégrer r dans le plan actuel Mu
    """
    u = next((user for user in instance.users if user.uid == u_id), None)
    if u is None:
        return 0.0, None
    
    # Instance actuelle P[u]
    tasks_u = [t for t in instance.tasks if t.owner == u_id]
    obs_u = [o for o in instance.observations if o.owner == u_id]
    
    inst_u = ESOPInstance(
        nb_satellites=instance.nb_satellites, nb_users=1, nb_tasks=len(tasks_u),
        horizon=instance.horizon, satellites=instance.satellites, users=[u], 
        tasks=tasks_u, observations=obs_u
    )
    
    old_plan_reward = safe_plan_reward(all_user_plans, u_id)
    
    # Instance AVEC r : P[u] ∪ P[r]
    tasks_u_r = tasks_u + [request]
    obs_u_r = obs_u + [o for o in instance.observations if o.task_id == request.tid]
    
    inst_u_r = ESOPInstance(
        nb_satellites=instance.nb_satellites, nb_users=1, nb_tasks=len(tasks_u_r),
        horizon=instance.horizon, satellites=instance.satellites, users=[u], 
        tasks=tasks_u_r, observations=obs_u_r
    )
    
    new_plan_u_r = greedy_schedule_P_u(inst_u_r, u_id)
    new_plan_reward = safe_plan_reward({"temp": new_plan_u_r}, u_id)
    bid_value = new_plan_reward - old_plan_reward
    
    schedule_r = safe_get_schedule({"temp": new_plan_u_r}, request.tid)
    return bid_value, schedule_r

def integrate_observation(current_plan: Dict[str, List[Tuple[Observation, int]]], 
                         new_obs_schedule: Tuple[Observation, int], 
                         instance: ESOPInstance) -> Dict[str, List[Tuple[Observation, int]]]:
    """Opérateur ⊕ : refait greedy sur instance mise à jour (Ligne 8 des algos)"""
    obs, t_start = new_obs_schedule
    u_id = obs.owner
    
    u = next((user for user in instance.users if user.uid == u_id), None)
    if u is None:
        return current_plan
    
    tasks_u = [t for t in instance.tasks if t.owner == u_id]
    obs_u = [o for o in instance.observations if o.owner == u_id and o.task_id != obs.task_id]
    
    inst_u = ESOPInstance(
        nb_satellites=instance.nb_satellites, nb_users=1, nb_tasks=len(tasks_u),
        horizon=instance.horizon, satellites=instance.satellites, users=[u], 
        tasks=tasks_u, observations=obs_u
    )
    
    return greedy_schedule_P_u(inst_u, u_id)

# =============================================================================
# PSI - Algorithme 2 de l'article (COMMENTÉ LIGNE PAR LIGNE)
# =============================================================================

def psi_solve(instance: ESOPInstance) -> Dict[str, Dict[str, List[Tuple[Observation, int]]]]:
    """
    PSI - Algorithme 2 EXACT de l'article
    
    Data: An EOSCSP P = ⟨S, U, R, O⟩
    Result: An assignment M
    """
    # Ligne 1: Mu0 ← ∅
    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    user_plans = {u.uid: greedy_schedule_P_u(instance, u.uid) for u in exclusive_users}
    Mu0 = {}  # Plan collecté par u0
    
    # Ligne 3-4: for each r ∈ R do Bu[r], σu[r] ← bid(r, Mu) // send Bu, σu to u0
    u0_tasks = [t for t in instance.tasks if t.owner == "u0"]
    for r in u0_tasks:
        bids_r = {u.uid: bid(u.uid, instance, r, user_plans) for u in exclusive_users}
        
        if not bids_r:
            continue
            
        # Ligne 6: w ← arg maxu∈Uex {Bu[r]}
        winner_id = max(bids_r, key=lambda uid: bids_r[uid][0])
        winner_bid, sigma_w = bids_r[winner_id]
        
        if sigma_w is None or winner_bid <= 0:
            continue
            
        # Ligne 7: Mu0 ← Mu0 ∪ {σw[r]}
        obs, t = sigma_w
        sid = obs.satellite
        Mu0.setdefault(sid, []).append(sigma_w)
        
        # Ligne 8: Mw ← Mw ⊕ σw[r] // send Mw[r] to w
        user_plans[winner_id] = integrate_observation(
            user_plans[winner_id], sigma_w, instance
        )
    
    # Ligne 9: Mu0 ← solve(P[u0|Mu0])
    inst_u0_fixed = create_instance_with_fixed_observations(instance, Mu0)
    final_u0_plan = greedy_schedule(inst_u0_fixed).get("u0", {})
    
    # Ligne 10: return ∪u∈U Mu
    return {**user_plans, "u0": final_u0_plan}

# =============================================================================
# SSI - Algorithme 3 de l'article (COMMENTÉ LIGNE PAR LIGNE)
# =============================================================================

def ssi_solve(instance: ESOPInstance, sort_key=lambda r: r.t_end) -> Dict[str, Dict[str, List[Tuple[Observation, int]]]]:
    """
    SSI - Algorithme 3 EXACT de l'article
    
    Data: An EOSCSP P = ⟨S, U, R, O⟩
    Result: An assignment M
    """
    # Ligne 1: Mu0 ← ∅
    # Ligne 2: for each u ∈ Uex do concurrently Mu ← solve(P[u])
    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    user_plans = {u.uid: greedy_schedule_P_u(instance, u.uid) for u in exclusive_users}
    Mu0 = {}
    
    # Ligne 3: for each r ∈ sorted(R) do
    u0_tasks = sorted([t for t in instance.tasks if t.owner == "u0"], key=sort_key)
    
    # Lignes 4-5: for each u ∈ Uex do Bu[r], σu[r] ← bid(r, Mu) // send Bu[r], σu[r] to u0
    for r in u0_tasks:
        bids_r = {u.uid: bid(u.uid, instance, r, user_plans) for u in exclusive_users}
        
        if not bids_r:
            continue
            
        # Ligne 6: w ← arg maxu∈Uex {Bu[r]}
        winner_id = max(bids_r, key=lambda uid: bids_r[uid][0])
        winner_bid, sigma_w = bids_r[winner_id]
        
        if sigma_w is None or winner_bid <= 0:
            continue
            
        # Ligne 7: Mu0 ← Mu0 ∪ {σw[r]}
        obs, t = sigma_w
        sid = obs.satellite
        Mu0.setdefault(sid, []).append(sigma_w)
        
        # Ligne 8: Mw ← Mw ⊕ σw[r] // send Mw[r] to w
        user_plans[winner_id] = integrate_observation(
            user_plans[winner_id], sigma_w, instance
        )
    
    # Ligne 9: Mu0 ← solve(P[u0|Mu0])
    inst_u0_fixed = create_instance_with_fixed_observations(instance, Mu0)
    final_u0_plan = greedy_schedule(inst_u0_fixed).get("u0", {})
    
    # Ligne 10: return ∪u∈U Mu
    return {**user_plans, "u0": final_u0_plan}

# =============================================================================
# 🆕 REGRET AUCTION (NOUVEAU)
# =============================================================================

def regret_bid(u_id: str, instance: ESOPInstance, request: Task, 
               all_user_plans: Dict[str, Dict[str, List[Tuple[Observation, int]]]],
               history_bids: Dict[str, float], alpha: float = 0.1) -> Tuple[float, Optional[Tuple[Observation, int]]]:
    """
    Enchère par regret : bid = gain_marginal + α * regret_passé
    - Favorise les utilisateurs qui ont souvent été "frustrés" (bons bids perdus)
    """
    # Bid classique (gain marginal)
    classic_bid, schedule = bid(u_id, instance, request, all_user_plans)
    
    # Regret passé : moyenne des enchères perdues (non-gagnées)
    past_regret = history_bids.get(u_id, 0.0)
    
    # Enchère finale = bid actuel + bonus regret
    regret_bonus = alpha * past_regret
    final_bid = classic_bid + regret_bonus
    
    return final_bid, schedule

def regret_auction_solve(instance: ESOPInstance, sort_key=lambda r: r.t_end, 
                        alpha: float = 0.1, n_rounds: int = 3) -> Dict[str, Dict[str, List[Tuple[Observation, int]]]]:
    """
    Auction par regret sur n_rounds itérations
    - Chaque round : enchères regret → allocation → mise à jour historique
    - Converge vers une allocation plus équilibrée/équitable
    """
    exclusive_users = [u for u in instance.users if u.uid != "u0"]
    
    # Historique des regrets (enchères perdues)
    history_bids = {u.uid: 0.0 for u in exclusive_users}
    
    # Plans initiaux
    user_plans = {u.uid: greedy_schedule_P_u(instance, u.uid) for u in exclusive_users}
    best_plans = user_plans.copy()
    best_score = 0
    
    # Requêtes u0 triées
    u0_tasks = sorted([t for t in instance.tasks if t.owner == "u0"], key=sort_key)
    
    for round_num in range(n_rounds):
        print(f"🔄 Regret Auction - Round {round_num+1}/{n_rounds}")
        Mu0 = {}
        
        # Enchères séquentielles avec regret
        for r in u0_tasks:
            bids_r = {}
            for u in exclusive_users:
                bid_val, schedule = regret_bid(u.uid, instance, r, user_plans, 
                                            history_bids, alpha)
                bids_r[u.uid] = (bid_val, schedule)
            
            if not bids_r:
                continue
                
            # Gagnant (max bid regret)
            winner_id = max(bids_r, key=lambda uid: bids_r[uid][0])
            winner_bid, sigma_w = bids_r[winner_id]
            
            if sigma_w is None or (winner_bid - history_bids[winner_id]) <= 0:
                continue
                
            # Allocation
            obs, t = sigma_w
            sid = obs.satellite
            Mu0.setdefault(sid, []).append(sigma_w)
            
            # Mise à jour plan gagnant
            user_plans[winner_id] = integrate_observation(
                user_plans[winner_id], sigma_w, instance
            )
            
            # 🔄 MISE À JOUR REGRET : les perdants accumulent du regret
            for loser_id, (loser_bid, _) in bids_r.items():
                if loser_id != winner_id:
                    history_bids[loser_id] += loser_bid * 0.1  # Regret pondéré
        
        # Évaluation du round
        inst_u0_fixed = create_instance_with_fixed_observations(instance, Mu0)
        final_u0_plan = greedy_schedule(inst_u0_fixed).get("u0", {})
        round_plans = {**user_plans, "u0": final_u0_plan}
        
        round_score = sum(sum(sum(obs.reward for obs, _ in obslist or []) 
                             for obslist in sat_plans.values()) 
                         for sat_plans in round_plans.values())
        
        print(f"   Round score: {round_score:.1f}")
        
        # Garder meilleur plan vu
        if round_score > best_score:
            best_score = round_score
            best_plans = deepcopy(round_plans)
    
    print(f"✅ Regret Auction final score: {best_score:.1f}")
    return best_plans

# =============================================================================
# TEST COMPARATIF
# =============================================================================

"""
Test complet des 3 algorithmes :
1. PSI : parallèle (Algo 2)
2. SSI : séquentiel (Algo 3) 
3. Regret : multi-rounds équitable
"""
