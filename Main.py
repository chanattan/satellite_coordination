from InstanceGenerator import *
from GreedySolver import *

if __name__ == '__main__':
    inst = generate_ESOP_instance(3, 2, 10, seed=0)
    print(inst.to_text())

    print("\n--- Greedy Solver Output ---\n")

    # Résolution gloutonne pour un utilisateur exclusif u1 - à vérifier
    user_id = 'u1'
    plan = greedy_schedule_for_user(inst, user_id)
    for sat_id, observations in plan.items():
        print(f"Planning for user {user_id} on satellite {sat_id}:")
        for obs, t_start in observations:
            print(f"  > Observation {obs.oid} starting at {t_start}")
    
