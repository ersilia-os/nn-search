import warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
