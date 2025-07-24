import os

PATH_PROJECT = os.getenv("PROJECT_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PATH_DP_SCRIPTS      = os.path.join(PATH_PROJECT, "scripts", "data_preparation")
PATH_INFERENCE       = os.path.join(PATH_PROJECT, "inference")
PATH_ROSETTA_TOOLS   = os.getenv("ROSETTA_TOOLS", os.path.join(PATH_PROJECT, "rosetta_tools"))  # fallback
PATH_PYTHON          = os.getenv("PYTHON_PATH",   "python")  # just use 'python' in path by default
PATH_DSSP            = os.path.join(PATH_PROJECT, )
PATH_ELEN_MODELS     = os.getenv("ELEN_MODELS",   os.path.join(PATH_PROJECT, "models"))
