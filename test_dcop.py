"""
Script de test pour vérifier la génération du DCOP et les expressions de contraintes.
"""

import yaml
import re
from InstanceGenerator import generate_ESOP_instance, generate_DCOP_instance


def test_constraint_expressions():
    """
    Teste que toutes les expressions de contraintes retournent bien des valeurs numériques.
    Utilise la syntaxe "expression" directe de pyDCOP (pas de "def").
    """
    print("="*60)
    print("TEST DES EXPRESSIONS DE CONTRAINTES")
    print("="*60 + "\n")
    
    # Générer une petite instance
    print("Génération d'une instance de test...")
    inst = generate_ESOP_instance(
        nb_satellites=2,
        nb_users=2,
        nb_tasks=3,
        horizon=100,
        seed=123
    )
    print(f"  - {len(inst.observations)} observations")
    print(f"  - {len([o for o in inst.observations if o.owner == 'u0'])} observations du central\n")
    
    # Générer le DCOP
    print("Génération du DCOP...")
    dcop_yaml = generate_DCOP_instance(inst)
    dcop = yaml.safe_load(dcop_yaml)
    
    print(f"  - {len(dcop.get('variables', {}))} variables")
    print(f"  - {len(dcop.get('constraints', {}))} contraintes\n")
    
    # Tester chaque expression de contrainte
    print("Test des expressions de contraintes:")
    constraints = dcop.get('constraints', {})
    
    all_passed = True
    for c_name, c_data in constraints.items():
        expr_str = c_data.get('function', '')
        
        try:
            # Extraire les noms de variables de l'expression avec regex
            # Cherche les variables de la forme x_<user>_<obs_id>
            var_names = re.findall(r'\b(x_\w+)\b', expr_str)
            unique_vars = list(set(var_names))
            
            if not unique_vars:
                # Expression sans variables (rare, mais possible)
                result = eval(expr_str, {'__builtins__': {}})
                if result is None:
                    print(f"   {c_name}: Expression retourne None")
                    all_passed = False
                elif not isinstance(result, (int, float)):
                    print(f"   {c_name}: Retourne {type(result).__name__} au lieu de int/float")
                    all_passed = False
                else:
                    print(f"   {c_name}")
            else:
                # Tester avec différentes combinaisons de valeurs
                test_cases = []
                if len(unique_vars) == 1:
                    test_cases = [
                        {unique_vars[0]: 0},
                        {unique_vars[0]: 1}
                    ]
                elif len(unique_vars) == 2:
                    test_cases = [
                        {unique_vars[0]: 0, unique_vars[1]: 0},
                        {unique_vars[0]: 0, unique_vars[1]: 1},
                        {unique_vars[0]: 1, unique_vars[1]: 0},
                        {unique_vars[0]: 1, unique_vars[1]: 1}
                    ]
                else:
                    # Pour plus de 2 variables, tester quelques cas
                    test_cases = [
                        {v: 0 for v in unique_vars},
                        {v: 1 if i == 0 else 0 for i, v in enumerate(unique_vars)},
                        {v: 1 for v in unique_vars}
                    ]
                
                errors = []
                for test_vals in test_cases:
                    try:
                        result = eval(expr_str, {'__builtins__': {}, **test_vals})
                        if result is None:
                            errors.append(f"retourne None pour {test_vals}")
                        elif not isinstance(result, (int, float)):
                            errors.append(f"retourne {type(result).__name__} pour {test_vals}")
                    except Exception as e:
                        errors.append(f"erreur lors de l'évaluation avec {test_vals}: {str(e)[:50]}")
                
                if errors:
                    print(f"   {c_name}:")
                    for error in errors:
                        print(f"      {error}")
                    all_passed = False
                else:
                    print(f"   {c_name}")
        
        except Exception as e:
            print(f"   {c_name}: Erreur - {str(e)[:60]}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print(" TOUS LES TESTS RÉUSSIS")
    else:
        print(" CERTAINS TESTS ONT ÉCHOUÉ")
    print("="*60 + "\n")
    
    return all_passed


def test_dcop_structure():
    """
    Vérifie la structure du DCOP généré.
    """
    print("="*60)
    print("TEST DE LA STRUCTURE DU DCOP")
    print("="*60 + "\n")
    
    inst = generate_ESOP_instance(
        nb_satellites=2,
        nb_users=3,
        nb_tasks=4,
        seed=456
    )
    
    dcop_yaml = generate_DCOP_instance(inst)
    dcop = yaml.safe_load(dcop_yaml)
    
    checks = []
    
    # Vérifier les champs obligatoires
    checks.append(("Champ 'name'", 'name' in dcop))
    checks.append(("Champ 'objective'", 'objective' in dcop))
    checks.append(("Champ 'domains'", 'domains' in dcop))
    checks.append(("Champ 'agents'", 'agents' in dcop))
    checks.append(("Champ 'variables'", 'variables' in dcop))
    checks.append(("Champ 'constraints'", 'constraints' in dcop))
    
    # Vérifier que l'objectif est 'min'
    checks.append(("Objectif = 'min'", dcop.get('objective') == 'min'))
    
    # Vérifier qu'il y a un domaine 'binary'
    checks.append(("Domaine 'binary' existe", 'binary' in dcop.get('domains', {})))
    
    # Vérifier que binary a les valeurs [0, 1]
    binary_values = dcop.get('domains', {}).get('binary', {}).get('values', [])
    checks.append(("Domaine binary = [0, 1]", binary_values == [0, 1]))
    
    # Vérifier que toutes les variables ont un agent
    all_vars_have_agent = all(
        'agent' in v for v in dcop.get('variables', {}).values()
    )
    checks.append(("Toutes les variables ont un agent", all_vars_have_agent))
    
    # Vérifier que toutes les contraintes ont une fonction
    all_constraints_have_func = all(
        'function' in c for c in dcop.get('constraints', {}).values()
    )
    checks.append(("Toutes les contraintes ont une fonction", all_constraints_have_func))
    
    # Vérifier que les expressions de contraintes sont des strings (pas des defs)
    all_expressions_are_strings = all(
        isinstance(c.get('function', ''), str) 
        for c in dcop.get('constraints', {}).values()
    )
    checks.append(("Toutes les contraintes sont des expressions", all_expressions_are_strings))
    
    for check_name, passed in checks:
        symbol = "" if passed else ""
        print(f"  {symbol} {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    
    print("\n" + "="*60)
    if all_passed:
        print(" STRUCTURE VALIDE")
    else:
        print(" STRUCTURE INVALIDE")
    print("="*60 + "\n")
    
    return all_passed


def test_yaml_validity():
    """
    Vérifie que le YAML généré est bien formé et parsable.
    """
    print("="*60)
    print("TEST DE VALIDITÉ YAML")
    print("="*60 + "\n")
    
    inst = generate_ESOP_instance(
        nb_satellites=2,
        nb_users=2,
        nb_tasks=3,
        seed=789
    )
    
    try:
        dcop_yaml = generate_DCOP_instance(inst)
        dcop = yaml.safe_load(dcop_yaml)
        
        print(f"   YAML parsable")
        print(f"   {len(dcop.get('agents', []))} agents")
        print(f"   {len(dcop.get('variables', {}))} variables")
        print(f"   {len(dcop.get('constraints', {}))} contraintes")
        print("\n" + "="*60)
        print(" YAML VALIDE")
        print("="*60 + "\n")
        return True
    except Exception as e:
        print(f"   Erreur YAML: {str(e)[:100]}")
        print("\n" + "="*60)
        print(" YAML INVALIDE")
        print("="*60 + "\n")
        return False

def show_dcop_sample():
    """
    Affiche un aperçu du DCOP généré pour vérification visuelle.
    """
    print("="*60)
    print("APERÇU DU DCOP GÉNÉRÉ")
    print("="*60 + "\n")
    
    inst = generate_ESOP_instance(
        nb_satellites=2,
        nb_users=2,
        nb_tasks=2,
        seed=999
    )
    
    dcop_yaml = generate_DCOP_instance(inst)
    dcop = yaml.safe_load(dcop_yaml)
    
    print(f"Nom: {dcop.get('name')}")
    print(f"Objectif: {dcop.get('objective')}\n")
    
    print(f"Agents ({len(dcop.get('agents', []))}): ")
    agents = dcop.get('agents', [])
    real_agents = [a for a in agents if not a.startswith('aux_')]
    aux_agents = [a for a in agents if a.startswith('aux_')]
    print(f"  Réels: {', '.join(real_agents)}")
    if aux_agents:
        print(f"  Auxiliaires: {len(aux_agents)} (aux_0 ... aux_{len(aux_agents)-1})")
    
    print(f"\nVariables ({len(dcop.get('variables', {}))}): ")
    for var_name in list(dcop.get('variables', {}).keys())[:3]:
        var_data = dcop['variables'][var_name]
        print(f"  - {var_name}: domain={var_data.get('domain')}, agent={var_data.get('agent')}")
    if len(dcop.get('variables', {})) > 3:
        print(f"  ... ({len(dcop.get('variables', {})) - 3} de plus)")
    
    print(f"\nContraintes ({len(dcop.get('constraints', {}))}): ")
    for const_name in list(dcop.get('constraints', {}).keys())[:3]:
        const_data = dcop['constraints'][const_name]
        func_str = const_data.get('function', '')
        func_short = (func_str[:70] + "...") if len(func_str) > 70 else func_str
        print(f"  - {const_name}")
        print(f"      {func_short}")
    if len(dcop.get('constraints', {})) > 3:
        print(f"  ... ({len(dcop.get('constraints', {})) - 3} de plus)")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTS DE VALIDATION DU DCOP")
    print("="*60 + "\n")
    
    # Test 1: Validité YAML
    yaml_ok = test_yaml_validity()
    
    # Test 2: Structure
    struct_ok = test_dcop_structure()
    
    # Test 3: Expressions
    expr_ok = test_constraint_expressions()
    
    # Test 4: Aperçu
    show_dcop_sample()
    
    # Résumé
    print("="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    print(f"  Validité YAML: {'OK' if yaml_ok else 'ÉCHEC'}")
    print(f"  Structure du DCOP: {'OK' if struct_ok else 'ÉCHEC'}")
    print(f"  Expressions de contraintes: {'OK' if expr_ok else 'ÉCHEC'}")
    
    if yaml_ok and struct_ok and expr_ok:
        print("\n Tous les tests sont réussis : le DCOP est valide et peut être résout.")
    else:
        print("\n Certains tests ont échoué : DCOP invalide.")
    print("="*60 + "\n")
