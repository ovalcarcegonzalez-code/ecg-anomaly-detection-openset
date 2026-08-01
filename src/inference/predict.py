"""
Inferencia: carga del modelo entrenado y puntuación de anomalía sobre señales
ECG arbitrarias.

El score de anomalía combina dos componentes estandarizados:
  - Error de reconstrucción de la morfología del latido.
  - Error de reconstrucción de las características de ritmo.

La estandarización usa las estadísticas calculadas sobre el conjunto de
validación durante el entrenamiento, de modo que un score de valor 2 significa
"dos desviaciones típicas por encima de lo observado en latidos normales".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from src.config import MEMAE_CLINICO_PATH, UMBRAL_PRODUCCION, W_RR
from src.data.preprocessing import procesar_senal_cruda
from src.models.memae import MemAEHibrido

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]


# ---------------------------------------------------------------------------
# Estructuras de salida
# ---------------------------------------------------------------------------







@dataclass
class ResultadoLatido:
    """Resultado del análisis de un latido individual."""

    indice: int
    posicion_muestra: int
    score: float
    es_anomalo: bool
    error_morfologia: float
    error_ritmo: float
    z_morfologia: float
    z_ritmo: float
    senal: FloatArray = field(repr=False)
    reconstruccion: FloatArray = field(repr=False)
    features_rr: FloatArray = field(repr=False)

    @property
    def causa(self) -> str:
        """
        Indica qué componente del error predomina en el latido.

        No identifica el tipo de arritmia: el modelo detecta desviaciones de la
        normalidad, no clasifica patologías.
        """
        if not self.es_anomalo:
            return "normal"
        morf_alta = self.z_morfologia > 1.0
        ritmo_alto = self.z_ritmo > 1.0
        if morf_alta and ritmo_alto:
            return "morfología y ritmo"
        if ritmo_alto:
            return "ritmo"
        if morf_alta:
            return "morfología"
        return "combinada"

    @property
    def causa_descripcion(self) -> str:
        """Descripción legible del componente del error que predomina."""
        descripciones = {
            "morfología y ritmo": "Desviación en forma y ritmo",
            "morfología": "Desviación en la forma del latido",
            "ritmo": "Desviación en el ritmo",
            "combinada": "Desviación combinada leve",
            "normal": "Dentro de la normalidad",
        }
        return descripciones[self.causa]

    @property
    def error_puntual(self) -> FloatArray:
        """Error de reconstrucción muestra a muestra, para localizar la anomalía."""
        return (self.senal - self.reconstruccion) ** 2


@dataclass
class ResultadoAnalisis:
    """Resultado agregado del análisis de una señal completa."""

    latidos: list[ResultadoLatido]
    umbral: float
    frecuencia_muestreo: int

    @property
    def n_latidos(self) -> int:
        return len(self.latidos)

    @property
    def n_anomalos(self) -> int:
        return sum(1 for lat in self.latidos if lat.es_anomalo)

    @property
    def porcentaje_anomalos(self) -> float:
        return 100.0 * self.n_anomalos / self.n_latidos if self.n_latidos else 0.0

    @property
    def scores(self) -> FloatArray:
        return np.array([lat.score for lat in self.latidos])

    @property
    def tiempos(self) -> FloatArray:
        """Instante (en segundos) de cada latido dentro de la señal."""
        return np.array(
            [lat.posicion_muestra / self.frecuencia_muestreo for lat in self.latidos]
        )

    @property
    def frecuencia_cardiaca_media(self) -> float:
        """Frecuencia cardíaca estimada en latidos por minuto."""
        if self.n_latidos < 2:
            return float("nan")
        duracion_min = (self.tiempos[-1] - self.tiempos[0]) / 60.0
        return (self.n_latidos - 1) / duracion_min if duracion_min > 0 else float("nan")

    def top_anomalos(self, k: int = 10) -> list[ResultadoLatido]:
        """Los k latidos con mayor score, para revisión priorizada."""
        return sorted(self.latidos, key=lambda lat: lat.score, reverse=True)[:k]

    def detectar_rachas(self, min_longitud: int = 3) -> list[tuple[int, int]]:
        """
        Localiza secuencias de latidos anómalos consecutivos.

        Clínicamente relevante: una extrasístole aislada suele ser benigna,
        mientras que tres o más consecutivas constituyen un evento de mayor
        gravedad.

        Returns:
            Lista de tuplas (indice_inicio, indice_fin) de cada racha.
        """
        rachas: list[tuple[int, int]] = []
        inicio: int | None = None

        for i, lat in enumerate(self.latidos):
            if lat.es_anomalo and inicio is None:
                inicio = i
            elif not lat.es_anomalo and inicio is not None:
                if i - inicio >= min_longitud:
                    rachas.append((inicio, i - 1))
                inicio = None

        if inicio is not None and self.n_latidos - inicio >= min_longitud:
            rachas.append((inicio, self.n_latidos - 1))

        return rachas

    def resumen(self) -> dict[str, float | int]:
        return {
            "n_latidos": self.n_latidos,
            "n_anomalos": self.n_anomalos,
            "porcentaje_anomalos": round(self.porcentaje_anomalos, 2),
            "frecuencia_cardiaca_bpm": round(self.frecuencia_cardiaca_media, 1),
            "score_medio": round(float(self.scores.mean()), 4),
            "score_maximo": round(float(self.scores.max()), 4),
            "umbral": round(self.umbral, 4),
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DetectorAnomalias:
    """
    Envuelve el modelo entrenado y todos los artefactos necesarios para
    reproducir el cálculo de score empleado en la evaluación.

    Los parámetros de producción (`w_rr`, `umbral`) se toman de `config.py`
    y no del checkpoint, de modo que puedan ajustarse sin reentrenar. Los
    valores originales del checkpoint se conservan como referencia.
    """

    def __init__(self, checkpoint_path: Path = MEMAE_CLINICO_PATH) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {checkpoint_path}. "
                "Verifique que el fichero .pt está en el directorio models/."
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.modelo = MemAEHibrido(**ckpt["config"])
        self.modelo.load_state_dict(ckpt["model_state_dict"])
        self.modelo.eval()

        # Estandarización de las features de ritmo (estadísticas de train)
        self.rr_mean = np.asarray(ckpt["rr_mean"], dtype=np.float32)
        self.rr_std = np.asarray(ckpt["rr_std"], dtype=np.float32)

        # Estandarización de los errores de reconstrucción (estadísticas de validación)
        self.mu_sig = float(ckpt["err_mu_sig"])
        self.sd_sig = float(ckpt["err_sd_sig"])
        self.mu_rr = float(ckpt["err_mu_rr"])
        self.sd_rr = float(ckpt["err_sd_rr"])

        # Parámetros de producción (config.py) y sus valores originales
        self.w_rr = W_RR
        self.w_rr_checkpoint = float(ckpt["w_rr"])
        self.umbral = UMBRAL_PRODUCCION
        self.umbral_checkpoint = float(ckpt["umbral"])

        self.fpr_objetivo = float(ckpt.get("fpr_objetivo", 0.10))
        self.definicion = str(ckpt.get("definicion", "desconocida"))

    # -- Cálculo interno ----------------------------------------------------

    def _puntuar_lote(
        self, latidos: FloatArray, features_rr: FloatArray, batch_size: int = 256
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Ejecuta el modelo sobre un conjunto de latidos ya preprocesados.

        Returns:
            reconstrucciones, error de morfología por latido, error de ritmo
            por latido.
        """
        rr_norm = (features_rr - self.rr_mean) / (self.rr_std + 1e-8)

        n = len(latidos)
        recons = np.zeros_like(latidos)
        err_sig = np.zeros(n, dtype=np.float32)
        err_rr = np.zeros(n, dtype=np.float32)

        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb_s = torch.tensor(latidos[i : i + batch_size]).unsqueeze(1)
                xb_r = torch.tensor(rr_norm[i : i + batch_size], dtype=torch.float32)

                out_s, out_r, _ = self.modelo(xb_s, xb_r)

                recons[i : i + batch_size] = out_s.squeeze(1).numpy()
                err_sig[i : i + batch_size] = (
                    ((out_s - xb_s) ** 2).mean(dim=(1, 2)).numpy()
                )
                err_rr[i : i + batch_size] = ((out_r - xb_r) ** 2).mean(dim=1).numpy()

        return recons, err_sig, err_rr

    def _combinar_scores(
        self, err_sig: FloatArray, err_rr: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Estandariza y combina ambos componentes del error.

        La estandarización previa es imprescindible: sin ella, el componente
        con mayor magnitud numérica domina la suma con independencia de su
        capacidad discriminativa real.

        Returns:
            score combinado, z-score de morfología, z-score de ritmo.
        """
        z_sig = (err_sig - self.mu_sig) / (self.sd_sig + 1e-8)
        z_rr = (err_rr - self.mu_rr) / (self.sd_rr + 1e-8)
        score = (1 - self.w_rr) * z_sig + self.w_rr * z_rr
        return score, z_sig, z_rr

    # -- API pública --------------------------------------------------------

    def calcular_scores(self, senal: FloatArray, fs: int) -> FloatArray:
        """
        Devuelve el score de anomalía de cada latido, sin aplicar umbral.

        Útil para calibración, donde solo interesa la distribución de scores.
        """
        latidos, features_rr, _ = procesar_senal_cruda(senal, fs)
        _, err_sig, err_rr = self._puntuar_lote(latidos, features_rr)
        score, _, _ = self._combinar_scores(err_sig, err_rr)
        return score

    def calibrar_umbral(
        self, senal: FloatArray, fs: int, fpr_objetivo: float | None = None
    ) -> float:
        """
        Calibra el umbral sobre una señal de referencia asumida mayoritariamente
        normal.

        Permite adaptar el sistema a un paciente concreto usando un tramo basal
        suyo, lo que mitiga la variabilidad inter-paciente observada en los
        experimentos.

        Args:
            senal: señal ECG de referencia.
            fs: frecuencia de muestreo en Hz.
            fpr_objetivo: proporción de latidos de referencia que se aceptará
                marcar como anómalos. Por defecto, el valor de configuración.
        """
        fpr = self.fpr_objetivo if fpr_objetivo is None else fpr_objetivo
        scores = self.calcular_scores(senal, fs)
        return float(np.percentile(scores, 100 * (1 - fpr)))

    def calibrar_umbral_pooled(
        self,
        senales: list[tuple[FloatArray, int]],
        fpr_objetivo: float | None = None,
    ) -> float:
        """
        Calibra el umbral agrupando los scores de varias señales de referencia.

        Preferible a promediar umbrales individuales: dado que cada registro
        presenta su propia distribución de scores, la mediana de los umbrales
        individuales no reproduce la tasa de falsos positivos deseada sobre el
        conjunto.

        Args:
            senales: lista de pares (señal, frecuencia de muestreo).
            fpr_objetivo: tasa de falsos positivos objetivo sobre el conjunto.
        """
        fpr = self.fpr_objetivo if fpr_objetivo is None else fpr_objetivo
        pool = np.concatenate([self.calcular_scores(s, fs) for s, fs in senales])
        return float(np.percentile(pool, 100 * (1 - fpr)))

    def analizar(
        self, senal: FloatArray, fs: int, umbral: float | None = None
    ) -> ResultadoAnalisis:
        """
        Analiza una señal ECG completa.

        Args:
            senal: señal ECG cruda, 1D.
            fs: frecuencia de muestreo en Hz.
            umbral: umbral de decisión. Si es None, se usa el de configuración.

        Returns:
            Resultado con un objeto por latido y métricas agregadas.
        """
        latidos, features_rr, picos = procesar_senal_cruda(senal, fs)
        recons, err_sig, err_rr = self._puntuar_lote(latidos, features_rr)
        scores, z_sig, z_rr = self._combinar_scores(err_sig, err_rr)

        umbral_usado = self.umbral if umbral is None else umbral

        resultados = [
            ResultadoLatido(
                indice=i,
                posicion_muestra=int(picos[i]),
                score=float(scores[i]),
                es_anomalo=bool(scores[i] > umbral_usado),
                error_morfologia=float(err_sig[i]),
                error_ritmo=float(err_rr[i]),
                z_morfologia=float(z_sig[i]),
                z_ritmo=float(z_rr[i]),
                senal=latidos[i],
                reconstruccion=recons[i],
                features_rr=features_rr[i],
            )
            for i in range(len(latidos))
        ]

        return ResultadoAnalisis(
            latidos=resultados, umbral=umbral_usado, frecuencia_muestreo=fs
        )


@lru_cache(maxsize=1)
def cargar_detector(checkpoint_path: Path = MEMAE_CLINICO_PATH) -> DetectorAnomalias:
    """
    Carga el detector una sola vez y lo reutiliza.

    El caché evita releer el checkpoint en cada llamada, lo que en Streamlit
    (que reejecuta el script completo en cada interacción) supondría una
    penalización notable.
    """
    return DetectorAnomalias(checkpoint_path)