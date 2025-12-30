import subprocess
import re
from InstanceGenerator import generate_DCOP_instance
from ESOPInstance import ESOPInstance

def save_dcop_instance(dcop):
    """
    Sauvegarde l'instance DCOP au format YAML dans un fichier "esop_dcop.yaml".
    """
    with open("esop_dcop.yaml", "w") as f:
        f.write(dcop)

def run_pydcop_solve(yaml_path: str, algo: str = "dpop") -> str:
    """
    Lance 'pydcop solve --algo {algo} {yaml_path}' et renvoie stdout sous forme de string.
    """
    cmd = ["pydcop", "solve", "--algo", algo, yaml_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de pydcop: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Erreur: pydcop n'est pas installé ou n'est pas dans le PATH")
        print("Installez-le avec: pip install pydcop")
        return None

import json
import re
from typing import Dict


def parse_assignment_from_output(output: str) -> Dict[str, int]:
    """
    Parse l'assignement des variables depuis la sortie JSON de pydcop solve.
    
    La sortie de pyDCOP est capturée au format JSON pour réafficher proprement la section assignment.
    """
    if output is None or output.strip() == "":
        return {}
    
    assignment = {}
    
    try:
        data = json.loads(output)
        
        # extraction de la section assignment
        if "assignment" in data:
            assignment = data["assignment"]
            assignment = {k: int(v) for k, v in assignment.items()}
        
        return assignment
    
    except json.JSONDecodeError:
        print("Parsing JSON échoué -> parsing manuel")
        
        pattern = re.compile(r'^\s*"?([A-Za-z0-9_]+)"?\s*:\s*([0-9]+)')
        
        in_assign_section = False
        for line in output.splitlines():
            if "assignment" in line.lower():
                in_assign_section = True
                continue
            
            if not in_assign_section:
                continue
            if line.strip().startswith("}"):
                break
            
            m = pattern.match(line)
            if m:
                var_name, val = m.group(1), int(m.group(2))
                assignment[var_name] = val
        
        return assignment


def print_assignment_summary(instance, assignment: Dict[str, int]):
    """
    Affiche un résumé clair de l'assignement du DCOP.
    """
    print("="*70)
    print("RÉSUMÉ DES RÉSULTATS DU DCOP")
    print("="*70 + "\n")
    
    # map observation_id -> observation
    obs_by_id = {o.oid: o for o in instance.observations}
    
    # Grouper par utilisateur exclusif
    user_allocations = {}
    total_reward = 0
    
    for var_name, value in assignment.items():
        if value != 1:
            continue
        
        # Parser : x_{u_id}_{o_id}
        try:
            _, u_id, o_id = var_name.split("_", 2)
        except ValueError:
            continue
        
        if o_id not in obs_by_id:
            continue
        
        obs = obs_by_id[o_id]
        
        if u_id not in user_allocations:
            user_allocations[u_id] = []
        
        user_allocations[u_id].append(obs)
        total_reward += obs.reward
    
    if not user_allocations:
        print("  Aucune observation allouée aux utilisateurs exclusifs.\n")
    else:
        print(f"Observations allouées aux utilisateurs exclusifs:")
        print(f"{'-'*70}\n")
        
        for u_id in sorted(user_allocations.keys()):
            observations = user_allocations[u_id]
            user_reward = sum(o.reward for o in observations)
            
            print(f"  {u_id}:")
            print(f"    Nombre d'observations : {len(observations)}")
            print(f"    Récompense totale : {user_reward}")
            print(f"    Observations : ")
            
            for obs in observations:
                print(f"      - {obs.oid} (task={obs.task_id}, reward={obs.reward}, sat={obs.satellite})")
            print()
    
    print(f"TOTAL : {sum(len(obs_list) for obs_list in user_allocations.values())} observations allouées")
    print(f"RÉCOMPENSE TOTALE : {total_reward}\n")
    print("="*70 + "\n")
    
    return user_allocations, total_reward


def print_dcop_metrics(output: str):
    """
    Affiche les métriques de résolution du DCOP.
    """
    try:
        data = json.loads(output)
        
        print("="*70)
        print("MÉTRIQUES DE RÉSOLUTION DCOP")
        print("="*70 + "\n")
        
        print(f"  Statut: {data.get('status', 'UNKNOWN')}")
        print(f"  Coût final: {data.get('cost', 'N/A')}")
        print(f"  Violation: {data.get('violation', 'N/A')}")
        print(f"  Cycles: {data.get('cycle', 'N/A')}")
        print(f"  Temps de calcul: {data.get('time', 'N/A')} secondes")
        print(f"  Nombre de messages: {data.get('msg_count', 'N/A')}")
        print(f"  Taille des messages: {data.get('msg_size', 'N/A')} bytes\n")
        
        print("="*70 + "\n")
    except:
        pass

def assignment_to_user_plans(instance: ESOPInstance, assignment: dict):
    """
    Convertit un assignement DCOP en plannings utilisateurs.
    """
    user_plans = {}
    for var_name, value in assignment.items():
        if value != 1:
            continue
        # variable x_{u_id}_{o_id}
        try:
            _, u_id, o_id = var_name.split("_", 2)
        except ValueError:
            # variable qui ne respecte pas ce schéma
            continue

        # retrouver l'observation
        try:
            obs = next(o for o in instance.observations if o.oid == o_id)
        except StopIteration:
            continue

        user_plans.setdefault(u_id, {}).setdefault(obs.satellite, []).append((obs, None))

    return user_plans

def print_user_plans(user_plans):
    """
    Affiche les plannings utilisateurs.
    """
    if not user_plans:
        print("Aucun planning généré.")
        return
    
    for u_id, plan in user_plans.items():
        print(f"Planning pour l'utilisateur {u_id}:")
        for sat_id, observations in plan.items():
            print(f"  Sur le satellite {sat_id}:")
            for obs, _ in observations:
                print(f"    > Observation {obs.oid} (reward: {obs.reward})")
        
        # Calculer le score total pour cet utilisateur
        total_reward = sum(obs.reward for sat_obs in plan.values() for obs, _ in sat_obs)
        print(f"  Score total: {total_reward}\n")

def validate_dcop_functions(dcop_yaml: str) -> bool:
    """
    Tests pour valider que toutes les fonctions du DCOP retournent bien des valeurs.
    """
    print("Validation des fonctions de contraintes...")
    
    import re
    
    # extraction de toutes les fonctions
    func_pattern = re.compile(r'function: ["\'](.+?)["\']', re.DOTALL)
    functions = func_pattern.findall(dcop_yaml)
    
    errors = []
    for i, func_str in enumerate(functions):
        # check qu'il y a au moins un return
        if 'return' not in func_str:
            errors.append(f"Fonction {i+1}: Aucune instruction return trouvée")
        
        # check qu'il n'y a pas de chemins sans return
        lines = func_str.split('\\n')
        if_count = sum(1 for line in lines if 'if ' in line or 'elif ' in line)
        return_count = sum(1 for line in lines if 'return' in line)
        
        if if_count > 0 and return_count < if_count + 1:
            errors.append(f"Fonction {i+1}: Chemins possibles sans return")
    
    if errors:
        print("Erreurs :")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Toutes les fonctions sont valides")
    return True

def solve_dcop(inst):
    """
    Résout l'instance ESOP en la transformant en instance DCOP puis en utilisant l'algorithme DPOP avec PyDcop.
    """
    print("\n=== Résolution DCOP avec DPOP ===\n")
    
    yaml_path = "esop_dcop.yaml"
    
    # Générer le DCOP à partir de l'instance ESOP
    print("> Génération du DCOP...")
    dcop_yaml = generate_DCOP_instance(inst)
    
    # Vérification pour valider le format yaml, dont les fonctions
    validate_dcop_functions(dcop_yaml)
    
    # Sauvegarde du DCOP en yaml pour exécution manuelle, ici on fera appel à la commande dans Python
    save_dcop_instance(dcop_yaml)
    print(f"> DCOP sauvegardé dans {yaml_path}\n")
    
    lines = dcop_yaml.split('\n')
    nb_vars = sum(1 for line in lines if line.strip().startswith('x_'))
    nb_constraints = sum(1 for line in lines if line.strip().startswith('c_'))
    print(f"> Informations du DCOP:")
    print(f"  - Nombre de variables: {nb_vars}")
    print(f"  - Nombre de contraintes: {nb_constraints}\n")

    # On résout avec PyDcop et on capture la sortie
    print("Lancement de DPOP...")
    output = run_pydcop_solve(yaml_path, algo="dpop")
    
    if output is None:
        print("\n!!! Échec de la résolution DCOP.")
        return
    
    print("> Sortie de DPOP:")
    print("-" * 50)
    print(output)
    print("-" * 50)

    # On parse le résultat pour un affichage plus clair.
    print("\n> Parsing de la solution...")
    assignment = parse_assignment_from_output(output)
    
    if not assignment:
        print("!!! Aucun résultat trouvé dans la sortie de DPOP.")
        return
    
    # Résultats
    print_dcop_metrics(output)
    print_assignment_summary(inst, assignment)
    print("=== Fin de la résolution DCOP ===\n")