"""
Configuración centralizada del proyecto.

Todas las rutas y constantes viven aquí para que ningún módulo tenga que
calcular rutas relativas por su cuenta (que fue fuente de errores durante
la fase de experimentación en notebooks).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

# config.py está en src/, así que la raíz del proyecto es dos niveles arriba
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
MITDB_DIR: Path = RAW_DIR / "mitdb"
PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MEMAE_CLINICO_PATH: Path = MODELS_DIR / "memae_clinico.pt"
MEMAE_HIBRIDO_PATH: Path = MODELS_DIR / "memae_hibrido.pt"
BASELINE_PATH: Path = MODELS_DIR / "baseline_classifier.pt"

APP_DIR: Path = PROJECT_ROOT / "app"
EJEMPLOS_DIR: Path = APP_DIR / "ejemplos"

# ---------------------------------------------------------------------------
# Parámetros de señal y segmentación
# ---------------------------------------------------------------------------

FS_MITBIH: int = 360                 # frecuencia de muestreo del dataset (Hz)
VENTANA_MS: int = 600                # ventana por latido (ms)
LONGITUD_SENAL: int = 216            # muestras por latido: 360 Hz * 0.6 s
UMBRAL_STD_PLANO: float = 0.01       # descarte de latidos planos (mV)
N_RR_FEATURES: int = 4               # RR previo, RR siguiente, y sus dos ratios
VENTANA_RR_LOCAL: int = 10           # latidos previos para el ritmo basal local

# ---------------------------------------------------------------------------
# Arquitectura del modelo
# ---------------------------------------------------------------------------

LATENT_DIM: int = 14
MEM_DIM: int = 100                   # prototipos en el módulo de memoria
SHRINK_THRES: float = 0.0025         # umbral de sparsity de la atención

# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------
# W_RR y UMBRAL son parámetros de producción, ajustables sin reentrenar.
# El checkpoint guarda los valores con los que se evaluó el modelo en el
# notebook (w_rr=0.25); aquí se usa 0.15 por dos motivos: fue el óptimo del
# barrido de AUROC global, y reduce el peso del componente de ritmo, que es
# el más sensible a errores de detección automática de picos R en producción.
W_RR: float = 0.15                   # peso del componente de ritmo en el score
FPR_OBJETIVO: float = 0.10           # tasa de falsa alarma sobre latidos normales

# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------
# Umbral de producción, recalibrado tras fijar W_RR=0.15.
# Obtenido como mediana de los umbrales al 10% de FPR sobre cinco registros
# MIT-BIH predominantemente normales (100, 101, 103, 112, 115). Se usa la
# mediana por robustez frente a la notable variabilidad inter-paciente
# observada (rango: -0.83 a 0.48), que motiva además la opción de
# recalibración por paciente disponible en la interfaz.
UMBRAL_PRODUCCION: float = 0.1501




# ---------------------------------------------------------------------------
# Anotaciones AAMI
# ---------------------------------------------------------------------------

NO_LATIDO: set[str] = {"+", "~", "!", '"', "|", "[", "]"}

AAMI_MAP: dict[str, str] = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

CLASES_AAMI: list[str] = ["N", "S", "F", "V", "Q"]
CLASE_NORMAL: str = "N"              # definición clínica: solo N es normal

DESCRIPCION_CLASES: dict[str, str] = {
    "N": "Latido normal (sinusal o con conducción alterada)",
    "S": "Ectópico supraventricular",
    "F": "Latido de fusión",
    "V": "Ectópico ventricular",
    "Q": "Marcapasos o no clasificable",
}