
import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/KaiyangZhou/deep-person-reid.git"
TEMP_DIR = Path("temp_torchreid_install")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def check_installed():
    try:
        import torchreid
        logger.info(f"torchreid is already installed. Version: {torchreid.__version__}")
        return True
    except ImportError:
        logger.info("torchreid is not installed.")
        return False

def install_prerequisites():
    logger.info("Installing prerequisites...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "cython", "opencv-python", "gdown", "tensorboard", "future", "setuptools", "wheel"])

def install_torchreid():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    
    logger.info(f"Cloning {REPO_URL} into {TEMP_DIR}...")
    subprocess.check_call(["git", "clone", REPO_URL, str(TEMP_DIR)])
    
    setup_path = TEMP_DIR / "setup.py"
    if not setup_path.exists():
        logger.error(f"setup.py not found in {TEMP_DIR}")
        return False
        
    logger.info("Patching setup.py to disable Cython extensions...")
    with open(setup_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        stripped = line.strip()
        # Comment out Cython import
        if 'from Cython.Build import cythonize' in line:
            new_lines.append(f"# {line}")
        # Comment out ext_modules assignment in setup()
        elif 'ext_modules=cythonize(ext_modules)' in line:
             new_lines.append(f"# {line}")
        elif 'ext_modules=ext_modules' in line:
             new_lines.append(f"# {line}")
        # Comment out ext_modules definition list (safeguard)
        # We can't easily comment out a multi-line list without parsing, 
        # but commenting out the usage in setup() is sufficient.
        else:
            new_lines.append(line)
            
    with open(setup_path, 'w') as f:
        f.writelines(new_lines)
        
    logger.info("Installing torchreid from patched source...")
    # Use --no-build-isolation to ensure it uses the installed numpy/cython from current env
    # Using 'cwd' to install from the temp directory
    subprocess.check_call([sys.executable, "-m", "pip", "install", ".", "--no-build-isolation"], cwd=str(TEMP_DIR))
    
    logger.info("Cleaning up...")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        
    logger.info("Successfully installed torchreid!")
    return True

def main():
    if check_installed():
        print("torchreid is already installed.")
        # We don't exit here because the user might want to re-install if the previous one was broken
        # But for now, let's assume if it attempts to import successfully, it is fine.
        # However, the user specifically has an installation that MIGHT be broken or partially installed.
        # Let's verify if we can actually import it *outside* of this script?
        # Actually, if check_installed returns True, we probably don't need to do anything.
        return

    try:
        install_prerequisites()
        install_torchreid()
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
