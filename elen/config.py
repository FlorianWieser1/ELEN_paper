# elen/config.py
import os

# Default paths can be set here
PATH_PROJECT = "/home/florian_wieser/projects/ELEN/elen/"
PATH_PROJECT = os.getenv('PROJECT_PATH', PATH_PROJECT) # Overwrite if available in environment
PATH_DP_SCRIPTS = os.path.join(PATH_PROJECT, "scripts/data_preparation/")
PATH_MQA = os.path.join(PATH_PROJECT, "compare_mqa/")
PATH_INFERENCE = os.path.join(PATH_PROJECT, "inference/")
PATH_ROSETTA_TOOLS = "/home/florian_wieser/Rosetta_10-2022/main/tools/protein_tools/scripts"
PATH_ROSETTA_BIN = "/home/florian_wieser/Rosetta_10-2022/main/source/bin"
PATH_PYTHON = "/home/florian_wieser/miniconda3/envs/elen_test/bin/python"
PATH_DSSP = "/home/florian_wieser/miniconda3/envs/elen_test/bin/mkdssp"
PATH_SOFTWARE = "/home/florian_wieser/software/"
PATH_ELEN_MODELS = "/home/florian_wieser/projects/ELEN/models"
