from pathlib import Path
# Rutas base
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
# Nombre del archivo de datos
DATASET_NAME ="diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
RAW_DATA_PATH = RAW_DATA_DIR / DATASET_NAME
# Traget y features
TARGET = "Diabetes_binary"
RANDOM_STATE = 42
TEST_SIZE = 0.2
#Crear directorios si no existen
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    