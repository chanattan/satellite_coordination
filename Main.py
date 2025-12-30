from InstanceGenerator import *
from GreedySolver import *
from DCOP import *
import time

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

if __name__ == '__main__':
    print("="*60)
    print("PROJET COCOMA - Coordination de Satellites")
    print("="*60)
    
    # Génération d'une instance
    nb_satellites = 3
    nb_users = 5
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
    
    # résolution DCOP (partie 1 du projet)
    mode = 3
    
    if mode == 1:
        greedy_solve(inst)
    elif mode == 2:
        solve_dcop(inst)
    elif mode == 3:
        compare_solutions(inst)
    
    print("\n" + "="*60)
    print("Exécution terminée")
    print("="*60)

    # Visualisation animée
    # anim = animate_user_schedule_on_satellite(inst, user_id="u0", satellite_id="s0", interval_ms=500)
    
    #anim  # Jupyter affiche l’animation

