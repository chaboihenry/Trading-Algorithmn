import os
import RiskLabAI

# Get the installation directory
package_dir = os.path.dirname(RiskLabAI.__file__)
print(f"📂 Deep Scanning Library at: {package_dir}")
print("=" * 60)

# Keywords to hunt for
keywords = ["entropy", "regime", "structural", "break", "sadf", "cadf"]

found_files = {}

# Walk through every single file in the library
for root, dirs, files in os.walk(package_dir):
    for file in files:
        if file.endswith(".py"):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, package_dir)
            
            try:
                # Read the file content
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
                
                # Check if any keyword exists in the file content OR filename
                matches = [k for k in keywords if k in content or k in file.lower()]
                
                if matches:
                    found_files[rel_path] = matches
                    
            except Exception as e:
                pass

# Print results
if found_files:
    print(f"✅ Found {len(found_files)} potential matches:\n")
    for path, tags in found_files.items():
        print(f"📄 {path}")
        print(f"   └── Keywords: {', '.join(tags)}")
        
        # If it's a promising file, try to peek at class/def names
        if "entropy" in tags or "structural" in tags:
            try:
                with open(os.path.join(package_dir, path), "r") as f:
                    defs = [line.strip() for line in f if line.strip().startswith(("def ", "class "))]
                    if defs:
                        print(f"       └── Definitions: {defs[:3]}...") # Show first 3
            except: pass
        print("-" * 40)
else:
    print("❌ No files found containing 'entropy', 'regime', or 'structural'.")
    print("   -> The installed version of RiskLabAI likely excludes these modules.")

print("=" * 60)