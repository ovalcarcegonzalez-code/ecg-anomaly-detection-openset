"""
Preprocesamiento de señal ECG: segmentación en latidos y extracción de
características de ritmo.

Este módulo es compartido por el pipeline de entrenamiento (sobre registros
MIT-BIH anotados) y el de inferencia (sobre señales arbitrarias sin anotar),
garantizando que ambas rutas apliquen exactamente el mismo tratamiento.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.config import (
    AAMI_MAP,
    NO_LATIDO,
    UMBRAL_STD_PLANO,
    VENTANA_MS,
    VENTANA_RR_LOCAL,
)

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
StrArray = npt.NDArray[np.str_]


# ---------------------------------------------------------------------------
# Anotaciones (solo entrenamiento, en inferencia no hay etiquetas)
# ---------------------------------------------------------------------------

def filtrar_y_mapear(
    simbolos: list[str] | StrArray, picos: list[int] | IntArray
) -> tuple[IntArray, StrArray]:
    
    simbolos_arr = np.asarray(simbolos)
    picos_arr = np.asarray(picos)

    mask = ~np.isin(simbolos_arr, list(NO_LATIDO))
    simbolos_validos = simbolos_arr[mask]
    picos_validos = picos_arr[mask]

    etiquetas = np.array([AAMI_MAP.get(s, "Q") for s in simbolos_validos])
    return picos_validos, etiquetas


# ---------------------------------------------------------------------------
# Detección de picos R (solo inferencia; en entrenamiento vienen anotados)
# ---------------------------------------------------------------------------

def detectar_picos_r(senal: FloatArray, fs: int) -> IntArray:
    
    import neurokit2 as nk

    try:
        limpia = nk.ecg_clean(senal, sampling_rate=fs)
        _, info = nk.ecg_peaks(limpia, sampling_rate=fs)
        picos = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception as exc:
        raise RuntimeError(f"Fallo en la detección de picos R: {exc}") from exc

    if picos.size == 0:
        raise RuntimeError(
            "No se detectó ningún pico R. Verifique que la señal sea un ECG "
            "válido y que la frecuencia de muestreo sea correcta."
        )
    return picos


# ---------------------------------------------------------------------------
# Features de ritmo
# ---------------------------------------------------------------------------

def calcular_features_rr(
    picos: IntArray, indice: int, fs: int, ventana_local: int = VENTANA_RR_LOCAL
) -> list[float]:
    
    rr_previo = (picos[indice] - picos[indice - 1]) / fs
    rr_siguiente = (picos[indice + 1] - picos[indice]) / fs

    inicio = max(0, indice - ventana_local)
    rr_local = np.diff(picos[inicio : indice + 1]) / fs
    rr_medio = float(np.median(rr_local)) if rr_local.size else rr_previo

    return [
        float(rr_previo),
        float(rr_siguiente),
        float(rr_previo / (rr_medio + 1e-8)),
        float(rr_siguiente / (rr_medio + 1e-8)),
    ]


# ---------------------------------------------------------------------------
# Segmentación
# ---------------------------------------------------------------------------

def segmentar_latidos(
    senal: FloatArray,
    picos: IntArray,
    fs: int,
    etiquetas: StrArray | None = None,
    ventana_ms: int = VENTANA_MS,
    umbral_std_plano: float = UMBRAL_STD_PLANO,
) -> tuple[FloatArray, FloatArray, StrArray | None, IntArray]:
    
    media_ventana = int(fs * ventana_ms / 1000 / 2)

    latidos: list[FloatArray] = []
    features: list[list[float]] = []
    etiquetas_out: list[str] = []
    picos_out: list[int] = []

    for i in range(1, len(picos) - 1):
        pico = int(picos[i])

        if pico - media_ventana < 0 or pico + media_ventana > len(senal):
            continue

        crudo = senal[pico - media_ventana : pico + media_ventana]
        if float(crudo.std()) < umbral_std_plano:
            continue

        normalizado = (crudo - crudo.mean()) / (crudo.std() + 1e-8)

        latidos.append(normalizado)
        features.append(calcular_features_rr(picos, i, fs))
        picos_out.append(pico)
        if etiquetas is not None:
            etiquetas_out.append(str(etiquetas[i]))

    if not latidos:
        raise RuntimeError(
            "No se pudo extraer ningún latido válido de la señal. "
            "Puede deberse a una señal demasiado corta o de mala calidad."
        )

    return (
        np.asarray(latidos, dtype=np.float32),
        np.asarray(features, dtype=np.float32),
        np.asarray(etiquetas_out) if etiquetas is not None else None,
        np.asarray(picos_out, dtype=int),
    )


def procesar_senal_cruda(
    senal: FloatArray, fs: int
) -> tuple[FloatArray, FloatArray, IntArray]:
    
    picos = detectar_picos_r(senal, fs)
    latidos, features, _, picos_validos = segmentar_latidos(senal, picos, fs)
    return latidos, features, picos_validos