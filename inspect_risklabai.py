import os
import RiskLabAI

# 1. Get the installation path
package_path = os.path.dirname(RiskLabAI.__file__)
print(f"📍 RiskLabAI is installed at: {package_path}\n")

# 2. Define what we are looking for
targets = [
    "mean_decrease_impurity",
    "mean_decrease_accuracy",
    "cusum_filter_events_dynamic_threshold",
    "triple_barrier",
    "vertical_barrier",
    "fractional_difference_fixed",
    "get_optimal_portfolio_weights",
    "PurgedKFold"
]

print("🔍 Searching for functions...\n" + "="*60)

# 3. Walk through the files and find definitions
for root, dirs, files in os.walk(package_path):
    for file in files:
        if file.endswith(".py"):
            full_path = os.path.join(root, file)
            
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for target in targets:
                    # Look for function or class definitions
                    if f"def {target}" in content or f"class {target}" in content:
                        # Convert file path to Python import path
                        rel_path = os.path.relpath(full_path, os.path.dirname(package_path))
                        module_path = rel_path.replace(os.path.sep, ".").replace(".py", "")
                        
                        # Fix for __init__ files
                        if module_path.endswith(".__init__"):
                            module_path = module_path[:-9]
                            
                        print(f"✅ FOUND: {target}")
                        print(f"   Import: from {module_path} import {target}")
                        print("-" * 60)
            except Exception as e:
                print(f"Could not read {file}: {e}")