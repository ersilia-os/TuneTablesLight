from pathlib import Path
import glob, shutil, os

base_path = Path(__file__).parent.parent.resolve() / "tunetables" / "logs"
print(base_path)
files = glob.glob(os.path.join(base_path, "*.cpkt"))
print(files)
