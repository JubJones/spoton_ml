
import os
import sys
import shutil
import subprocess
import logging
import stat
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

def ensure_pip():
    """Ensure pip is available in the current environment."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        logger.warning("pip module not found. Attempting to bootstrap with ensurepip...")
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"])
            logger.info("pip installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to bootstrap pip: {e}")
            return False

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.
    """
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        logger.warning(f"Could not remove {path}: {exc_info}")

def cleanup_temp():
    if TEMP_DIR.exists():
        logger.info(f"Cleaning up {TEMP_DIR}...")
        try:
            shutil.rmtree(TEMP_DIR, onerror=on_rm_error)
        except Exception as e:
            logger.warning(f"Cleanup failed (non-critical): {e}")

def install_package(package_name):
    logger.info(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def install_requirements_from_file(req_path):
    logger.info(f"Installing requirements from {req_path}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_path)])

def install_torchreid():
    cleanup_temp() # Ensure clean start
    
    logger.info(f"Cloning {REPO_URL} into {TEMP_DIR}...")
    subprocess.check_call(["git", "clone", REPO_URL, str(TEMP_DIR)])
    
    setup_path = TEMP_DIR / "setup.py"
    req_path = TEMP_DIR / "requirements.txt"
    
    if not setup_path.exists():
        logger.error(f"setup.py not found in {TEMP_DIR}")
        return False
        
    # ROBUSTNESS FIX: Install all requirements from the repo FIRST
    # plus explicit trouble-makers like scipy/cython which might be needed during import
    logger.info("Pre-installing dependencies to avoid import errors during setup...")
    
    # 1. Core build deps
    install_package("numpy") 
    install_package("cython")
    install_package("setuptools")
    install_package("wheel")
    
    # 2. Dependencies often missed
    install_package("scipy")
    install_package("opencv-python")
    install_package("gdown")
    install_package("tensorboard")
    install_package("future")
    install_package("imageio")
    install_package("yacs")
    install_package("isort")
    install_package("yapf")
    
    # 3. Everything else in requirements.txt
    if req_path.exists():
        install_requirements_from_file(req_path)
        
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
        else:
            new_lines.append(line)
            
    with open(setup_path, 'w') as f:
        f.writelines(new_lines)
        
    logger.info("Installing torchreid from patched source...")
    # Use --no-build-isolation to ensure it uses the installed numpy/cython from current env
    subprocess.check_call([sys.executable, "-m", "pip", "install", ".", "--no-build-isolation"], cwd=str(TEMP_DIR))
    
    cleanup_temp()
        
    logger.info("Successfully installed torchreid!")
    return True

def main():
    if check_installed():
        print("torchreid is already installed.")
        return

    try:
        install_torchreid()
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
