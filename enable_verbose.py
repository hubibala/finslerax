import os
import glob

for fpath in glob.glob("experiments/wildfire/synthetic/*.py"):
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    new_content = content.replace("verbose=False", "verbose=True")
    
    if new_content != content:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {fpath}")
