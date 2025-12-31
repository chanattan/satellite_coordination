from InstanceGenerator import *
from GreedySolver import *
from DCOP import *
import time
import numpy as np

def greedy_solve(inst):
    """
    Résout l'instance avec l'algorithme glouton et affiche les résultats.
    """
    print("\n" + "="*60)
    print("=== RÉSOLUTION GLOUTONNE ===")
    print("="*60 + "\n")

    # Résolution gloutonne pour tous les utilisateurs
    user_plans = {}
    for user in inst.users:
        user_plans[user.uid] = greedy_schedule_for_user(inst, user.uid)

    # Affichage des plannings
    for user_id, plan in user_plans.items():
        total_reward = 0
        print(f"\nPlanning pour l'utilisateur {user_id}:")
        for sat_id, observations in plan.items():
            if observations:
                print(f"  Sur le satellite {sat_id}:")
                for obs, t_start in observations:
                    print(f"    > Observation {obs.oid} démarrant à t={t_start} (reward: {obs.reward})")
                    total_reward += obs.reward
        print(f"  Score total: {total_reward}")

    # Score global
    scores = assess_solution(inst, user_plans)
    global_score = sum(scores.values())
    print(f"\n{'='*60}")
    print(f"Score global (glouton): {global_score}")
    print(f"{'='*60}\n")

    return user_plans

def compare_solutions(inst):
    """
    Compare les solutions gloutonne et DCOP.
    """
    print("\n" + "="*60)
    print("=== COMPARAISON DES MÉTHODES ===")
    print("="*60 + "\n")
    
    # Solution gloutonne
    print("1. Calcul de la solution gloutonne...")
    user_plans_greedy = {}
    # Time measure
    time_start = time.time()
    for user in inst.users:
        user_plans_greedy[user.uid] = greedy_schedule_for_user(inst, user.uid)
    time_end = time.time()
    scores_greedy = assess_solution(inst, user_plans_greedy)
    score_global_greedy = sum(scores_greedy.values())
    print("\nPlannings gloutons:")
    print_user_plans(user_plans_greedy)
    print(f"   TOTAL : {sum(len(obs) for plan in user_plans_greedy.values() for obs in plan.values())} observations allouées")
    print(f"   Score global (glouton): {score_global_greedy}")
    print(f"   Temps de calcul (glouton): {time_end - time_start:.4f} secondes")
    
    print("\n" + "-"*60 + "\n")

    # Solution DCOP
    print("\n2. Calcul de la solution DCOP...")
    solve_dcop(inst, print_output=False)
    
    print("\n" + "="*60)
    print("Comparaison terminée")
    print("="*60 + "\n")

def compare_greedy_vs_sdcop(inst):
    print("\n" + "="*60)
    print("=== COMPARAISON GLOUTON vs S-DCOP (DPOP) ===")
    print("="*60 + "\n")

    # Glouton
    time_start = time.time()
    user_plans_greedy = {u.uid: greedy_schedule_for_user(inst, u.uid) for u in inst.users}
    time_end = time.time()
    scores_greedy = assess_solution(inst, user_plans_greedy)
    score_global_greedy = sum(scores_greedy.values())
    print("Solution gloutonne :")
    print_user_plans(user_plans_greedy)
    print(f"Temps de calcul (glouton): {time_end - time_start:.4f} secondes")
    print(f"Score global (glouton): {score_global_greedy}\n")

    # s-dcop(dpop)
    time_start = time.time()
    sdcop_plans, sdcop_assignments, avg_time_per_request = sdcop_with_pydcop(inst)
    time_end = time.time()
    scores_sdcop = {u.uid: 0 for u in inst.users}
    for u_id, plan in sdcop_plans.items():
        scores_sdcop[u_id] = sum(
            obs.reward for sat_plan in plan.values() for (obs, _) in sat_plan
        )
    score_global_sdcop = sum(scores_sdcop.values())

    print(f"Temps de calcul (s-dcop): {time_end - time_start:.4f} secondes (moyenne par requête: {avg_time_per_request:.4f} secondes)")
    print("Solution s-dcop (plans exclusifs après coordination) :")
    print_user_plans(sdcop_plans)
    print("Assignments (r, u, o) :")
    for r_id, u_id, obs in sdcop_assignments:
        print(f"  - {r_id} -> {u_id} via {obs.oid} (reward {obs.reward})")

    print(f"\nScore global (s-dcop): {score_global_sdcop}")
    print("="*60 + "\n")

def extensive_compare_greedy_vs_sdcop(nb_instances: int = 10):
    """
        Compare les deux méthodes sur plusieurs instances générées.
    """
    nb_satellites = 3
    nb_users = 2
    nb_tasks = 10
    times_greedy = []
    times_sdcop = []
    scores_g = []
    scores_d = []

    for i in range(nb_instances):
        print("\n" + "="*60)
        print(f"=== INSTANCE {i+1} / {nb_instances} ===")
        print("="*60 + "\n")

        inst = generate_ESOP_instance(
            nb_satellites=nb_satellites + np.random.randint(2,5),
            nb_users=nb_users + np.random.randint(2,5),
            nb_tasks=nb_tasks + np.random.randint(20,100),
            seed=None
        )

        # Glouton
        time_start = time.time()
        user_plans_greedy = {u.uid: greedy_schedule_for_user(inst, u.uid) for u in inst.users}
        time_end = time.time()
        times_greedy.append(time_end - time_start)
        scores_greedy = assess_solution(inst, user_plans_greedy)
        score_global_greedy = sum(scores_greedy.values())
        scores_g.append(score_global_greedy)

        # s-dcop(dpop)
        sdcop_plans, sdcop_assignments, avg_time_per_request = sdcop_with_pydcop(inst)
        times_sdcop.append(avg_time_per_request)
        scores_sdcop = {u.uid: 0 for u in inst.users}
        for u_id, plan in sdcop_plans.items():
            scores_sdcop[u_id] = sum(
                obs.reward for sat_plan in plan.values() for (obs, _) in sat_plan
            )
        score_global_sdcop = sum(scores_sdcop.values())
        scores_d.append(score_global_sdcop)
        
    print(f"\n=== RÉSULTATS SUR {nb_instances} INSTANCES ===")
    avg_time_greedy = sum(times_greedy) / nb_instances
    print(f"Temps moyen glouton : {avg_time_greedy:.4f} secondes | Score moyen : {sum(scores_g) / nb_instances:.2f}")
    avg_time_sdcop = sum(times_sdcop) / nb_instances
    print(f"Temps moyen s-dcop : {avg_time_sdcop:.4f} secondes | Score moyen : {sum(scores_d) / nb_instances:.2f}")
    print("="*60 + "\n")

if __name__ == '__main__':
    print("="*60)
    print("PROJET COCOMA - Coordination de Satellites")
    print("="*60)
    
    # Génération d'une instance
    nb_satellites = 3
    nb_users = 2
    nb_tasks = 5
    seed = None
    print("\nGénération de l'instance ESOP...")
    print(f"Paramètres: {nb_satellites} satellites, {nb_users} utilisateurs exclusifs, {nb_tasks} tâches, seed={seed}")
    
    inst = generate_ESOP_instance(
        nb_satellites=nb_satellites,
        nb_users=nb_users, 
        nb_tasks=nb_tasks, 
        seed=seed
    )
    
    print("\n" + "="*60)
    print("Instance générée:")
    print("="*60)
    print(inst.to_text())
    print("="*60 + "\n")

    print("Modes disponibles:")
    print("1. Résolution gloutonne seule")
    print("2. Résolution DCOP seule")
    print("3. Comparaison des deux méthodes")
    print("4. Comparaison glouton vs s-dcop Python")
    
    # résolution DCOP (partie 1 du projet)
    mode = 4
    
    if mode == 1:
        greedy_solve(inst)
    elif mode == 2:
        solve_dcop(inst)
    elif mode == 3:
        compare_solutions(inst)
    elif mode == 4:
        extensive_compare_greedy_vs_sdcop(nb_instances=3)
    
    print("\n" + "="*60)
    print("Exécution terminée")
    print("="*60)

    # Visualisation animée
    # anim = animate_user_schedule_on_satellite(inst, user_id="u0", satellite_id="s0", interval_ms=500)
    
    #anim  # Jupyter affiche l’animation

