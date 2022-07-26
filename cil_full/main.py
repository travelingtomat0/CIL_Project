import glob
import importlib

import prepare

if __name__ == '__main__':
    print("Preparing data...")
    prepare.run()

    print("Running models...")
    all_models = glob.glob("_*.py")
    for i, m in enumerate(all_models):
        print(f"### Model {i+1} / {len(all_models)} ###")
        importlib.import_module(m[:-3]).run()
