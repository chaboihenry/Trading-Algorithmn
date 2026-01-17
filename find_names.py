import os
import RiskLabAI

# Get the installation directory
package_dir = os.path.dirname(RiskLabAI.__file__)
print(f"📂 Scanning Library at: {package_dir}")

# Define the specific files we are interested in
target_files = {
    "MDI": os.path.join(package_dir, "features", "feature_importance", "feature_importance_mdi.py"),
    "MDA": os.path.join(package_dir, "features", "feature_importance", "feature_importance_mda.py")
}

print("=" * 60)

for name, path in target_files.items():
    print(f"\n🔍 Reading {name} file: {os.path.basename(path)}")
    
    if not os.path.exists(path):
        print("❌ File not found!")
        continue
        
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        found = False
        for line in lines:
            # Check for function or class definitions
            clean_line = line.strip()
            if clean_line.startswith("def ") or clean_line.startswith("class "):
                # Print the definition line (e.g., "def mean_decrease_impurity(...)")
                print(f"   👉 FOUND: {clean_line.split('(')[0]}")
                found = True
        
        if not found:
            print("   ⚠️ No functions or classes found in text.")
            
    except Exception as e:
        print(f"   ❌ Error reading text: {e}")

print("\n" + "=" * 60)