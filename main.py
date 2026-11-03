# =====================================================================
# MAIN PROGRAM FOR OPTIMIZING MONOCLONAL ANTIBODY PRODUCTION

# run project from here
# =====================================================================

from pathlib import Path
from src.functions import run_project

if __name__ == "__main__":
    run_project(Path(__file__).resolve().parent)