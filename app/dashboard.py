"""
Interfaz interactiva para la detección de anomalías en ECG.

Permite cargar una señal, ejecutar el detector y explorar los resultados.
Contiene: resumen global, serie temporal con los latidos sospechosos resaltados, revisión
navegable de cada anomalía con su reconstrucción, y análisis agregado.

Nota sobre la interpretación: el sistema es un detector de anomalías, no un
clasificador de arritmias. Para cada latido sospechoso se indica qué componente
del error predomina (forma o ritmo), pero no se infiere el tipo concreto de
ectopia.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# pyarrow usa mimalloc por defecto. El inicializador de heap por hilo falla
# en macOS ARM cuando Streamlit crea un hilo nuevo en cada reejecución del
# script
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


from src.config import EJEMPLOS_DIR, FPR_OBJETIVO, UMBRAL_PRODUCCION
from src.inference.predict import ResultadoAnalisis, ResultadoLatido, cargar_detector

# ---------------------------------------------------------------------------
# Configuración y estilo
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Detección de anomalías en ECG",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

C_SENAL = "#4C9EEB"
C_ANOMALO = "#E63946"
C_RECON = "#F4A261"
C_GRID = "rgba(250,250,250,0.08)"
C_TEXTO = "#B8BCC4"

COLOR_CAUSA: dict[str, str] = {
    "morfología y ritmo": "#E63946",
    "ritmo": "#F4A261",
    "morfología": "#9D6BC7",
    "combinada": "#8D99AE",
}

ORDEN_CAUSAS: list[str] = ["morfología y ritmo", "ritmo", "morfología", "combinada"]


def layout_base(alto: int = 320, **kwargs) -> dict:
    """Layout común a todos los gráficos, coherente con el tema oscuro."""
    base = dict(
        height=alto,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXTO, size=12),
        xaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID),
        yaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID),
        margin=dict(l=50, r=20, t=30, b=45),
        hoverlabel=dict(bgcolor="#1A1D24", font_size=12),
    )
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Carga de datos y modelo
# ---------------------------------------------------------------------------

@st.cache_resource
def obtener_detector():
    #El cache evita recargar el modelo en cada interacción del usuario
    return cargar_detector()


@st.cache_data(show_spinner=False)
def analizar_cacheado(senal: np.ndarray, fs: int, umbral: float) -> ResultadoAnalisis:
    return obtener_detector().analizar(senal, fs, umbral=umbral)


def cargar_csv(fichero) -> np.ndarray:
    # Extrae la señal de un CSV tomando la primera columna numérica
    df = pd.read_csv(fichero)
    numericas = df.select_dtypes(include="number")
    if numericas.empty:
        raise ValueError("El fichero no contiene ninguna columna numérica.")
    return numericas.iloc[:, 0].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def grafico_serie_temporal(
    res: ResultadoAnalisis, senal: np.ndarray, fs: int, ventana_ms: int = 600
) -> go.Figure:
    #Serie temporal completa con los tramos sospechosos resaltados en rojo
    t = np.arange(len(senal)) / fs
    media_ventana = int(fs * ventana_ms / 1000 / 2)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=senal, mode="lines", name="ECG",
        line=dict(color=C_SENAL, width=1.1),
        hovertemplate="t=%{x:.2f}s<br>%{y:.2f} mV<extra></extra>",
    ))

    anomalos = [lat for lat in res.latidos if lat.es_anomalo]
    if anomalos:
        x_seg: list[float | None] = []
        y_seg: list[float | None] = []
        for lat in anomalos:
            ini = max(0, lat.posicion_muestra - media_ventana)
            fin = min(len(senal), lat.posicion_muestra + media_ventana)
            x_seg.extend(t[ini:fin].tolist() + [None])
            y_seg.extend(senal[ini:fin].tolist() + [None])

        fig.add_trace(go.Scatter(
            x=x_seg, y=y_seg, mode="lines",
            name=f"Sospechoso ({len(anomalos)})",
            line=dict(color=C_ANOMALO, width=1.8),
            hoverinfo="skip",
        ))

        pos = np.array([lat.posicion_muestra for lat in anomalos])
        fig.add_trace(go.Scatter(
            x=pos / fs, y=senal[pos], mode="markers", showlegend=False,
            marker=dict(
                color=C_ANOMALO, size=8, symbol="triangle-down",
                line=dict(color="#0E1117", width=1),
            ),
            customdata=[[lat.indice, lat.score, lat.causa_descripcion] for lat in anomalos],
            hovertemplate=(
                "Latido %{customdata[0]}<br>t=%{x:.2f}s<br>"
                "score=%{customdata[1]:.2f}<br>%{customdata[2]}<extra></extra>"
            ),
        ))

    fig.update_layout(**layout_base(
        340,
        xaxis_title="Tiempo (s)", yaxis_title="Amplitud (mV)",
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


def grafico_scores(res: ResultadoAnalisis, indice_activo: int | None = None) -> go.Figure:
    #Evolución del score con el umbral y el latido en revisión destacado
    fig = go.Figure()

    normales = [lat for lat in res.latidos if not lat.es_anomalo]
    anomalos = [lat for lat in res.latidos if lat.es_anomalo]

    for grupo, nombre, color in [
        (normales, "Normal", C_SENAL),
        (anomalos, "Sospechoso", C_ANOMALO),
    ]:
        if not grupo:
            continue
        fig.add_trace(go.Scatter(
            x=[lat.posicion_muestra / res.frecuencia_muestreo for lat in grupo],
            y=[lat.score for lat in grupo],
            mode="markers", name=nombre,
            marker=dict(color=color, size=5.5, opacity=0.85),
            customdata=[[lat.indice, lat.causa_descripcion] for lat in grupo],
            hovertemplate=(
                "Latido %{customdata[0]}<br>t=%{x:.2f}s<br>"
                "score=%{y:.2f}<br>%{customdata[1]}<extra></extra>"
            ),
        ))

    if indice_activo is not None:
        lat = res.latidos[indice_activo]
        fig.add_trace(go.Scatter(
            x=[lat.posicion_muestra / res.frecuencia_muestreo], y=[lat.score],
            mode="markers", showlegend=False,
            marker=dict(color="rgba(0,0,0,0)", size=16,
                        line=dict(color="#FAFAFA", width=2)),
            hoverinfo="skip",
        ))

    fig.add_hline(
        y=res.umbral, line_dash="dash", line_color="rgba(250,250,250,0.4)",
        annotation_text=f"umbral {res.umbral:.2f}",
        annotation_position="top right",
        annotation_font=dict(color=C_TEXTO, size=11),
    )

    fig.update_layout(**layout_base(
        260, xaxis_title="Tiempo (s)", yaxis_title="Score de anomalía",
        legend=dict(orientation="h", y=1.16, x=0, bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


def grafico_reconstruccion(lat: ResultadoLatido, fs: int) -> go.Figure:
    #Señal original frente a la reconstrucción, con el error puntual como área
    t = np.arange(len(lat.senal)) / fs * 1000

    fig = go.Figure()

    err = lat.error_puntual
    rango = float(lat.senal.max() - lat.senal.min())
    err_esc = err / (err.max() + 1e-8) * rango * 0.30 + float(lat.senal.min())

    fig.add_trace(go.Scatter(
        x=t, y=err_esc, fill="tozeroy", name="Error local",
        fillcolor="rgba(230,57,70,0.18)", line=dict(width=0), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=lat.senal, mode="lines", name="Señal real",
        line=dict(color=C_SENAL, width=2.6),
        hovertemplate="%{x:.0f} ms<br>%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=lat.reconstruccion, mode="lines", name="Esperado por el modelo",
        line=dict(color=C_RECON, width=2, dash="dash"),
        hovertemplate="%{x:.0f} ms<br>%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(**layout_base(
        360, xaxis_title="Tiempo dentro del latido (ms)",
        yaxis_title="Amplitud normalizada",
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


def grafico_dispersion(res: ResultadoAnalisis, indice_activo: int | None = None) -> go.Figure:
    #Plano forma-ritmo, coloreado por componente predominante del error.
    fig = go.Figure()

    normales = [lat for lat in res.latidos if not lat.es_anomalo]
    if normales:
        fig.add_trace(go.Scatter(
            x=[lat.z_morfologia for lat in normales],
            y=[lat.z_ritmo for lat in normales],
            mode="markers", name="Normal",
            marker=dict(color=C_SENAL, size=5, opacity=0.45),
            hovertemplate="forma=%{x:.2f}σ<br>ritmo=%{y:.2f}σ<extra></extra>",
        ))

    for causa in ORDEN_CAUSAS:
        grupo = [lat for lat in res.latidos if lat.es_anomalo and lat.causa == causa]
        if not grupo:
            continue
        fig.add_trace(go.Scatter(
            x=[lat.z_morfologia for lat in grupo],
            y=[lat.z_ritmo for lat in grupo],
            mode="markers", name=causa.capitalize(),
            marker=dict(color=COLOR_CAUSA[causa], size=7, opacity=0.85),
            customdata=[[lat.indice] for lat in grupo],
            hovertemplate=(
                "Latido %{customdata[0]}<br>forma=%{x:.2f}σ<br>"
                "ritmo=%{y:.2f}σ<extra></extra>"
            ),
        ))

    if indice_activo is not None:
        lat = res.latidos[indice_activo]
        fig.add_trace(go.Scatter(
            x=[lat.z_morfologia], y=[lat.z_ritmo], mode="markers", showlegend=False,
            marker=dict(color="rgba(0,0,0,0)", size=18,
                        line=dict(color="#FAFAFA", width=2)),
            hoverinfo="skip",
        ))

    fig.update_layout(**layout_base(
        400, xaxis_title="Desviación en la forma (σ)",
        yaxis_title="Desviación en el ritmo (σ)",
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------

if "idx_revision" not in st.session_state:
    st.session_state.idx_revision = 0


def reiniciar_revision() -> None:
    # Vuelve al primer latido al cambiar de señal o de umbral.
    st.session_state.idx_revision = 0


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Datos de entrada")

    fuente = st.radio(
        "Origen", ["Ejemplo precargado", "Subir CSV"],
        label_visibility="collapsed", on_change=reiniciar_revision,
    )

    senal: np.ndarray | None = None
    fs = 360

    if fuente == "Ejemplo precargado":
        ejemplos = {
            "Registro con arritmias (208)": "ejemplo_mixto.csv",
            "Registro predominantemente normal (100)": "ejemplo_normal.csv",
        }
        elegido = st.selectbox("Ejemplo", list(ejemplos), on_change=reiniciar_revision)
        ruta = EJEMPLOS_DIR / ejemplos[elegido]
        if ruta.exists():
            senal = cargar_csv(ruta)
        else:
            st.error(f"No se encontró el fichero {ruta.name}")
    else:
        subido = st.file_uploader(
            "Fichero CSV", type=["csv"], on_change=reiniciar_revision,
            help="Se utilizará la primera columna numérica del fichero.",
        )
        fs = st.number_input("Frecuencia de muestreo (Hz)", 100, 2000, 360, 10)
        if subido is not None:
            try:
                senal = cargar_csv(subido)
            except Exception as exc:
                st.error(f"No se pudo leer el fichero: {exc}")

    st.markdown("---")
    st.markdown("### Sensibilidad")

    umbral = st.slider(
        "Umbral de decisión", -2.0, 5.0, float(UMBRAL_PRODUCCION), 0.05,
        help=(
            "Valores más bajos detectan más anomalías a costa de más falsas "
            "alarmas. El valor por defecto corresponde a un 10 % de falsos "
            "positivos sobre una población de referencia."
        ),
        on_change=reiniciar_revision,
    )
    if abs(umbral - UMBRAL_PRODUCCION) > 1e-6:
        st.caption(f"Modificado · valor por defecto {UMBRAL_PRODUCCION:.3f}")

    st.markdown("---")
    st.caption(
        "**MemAE híbrido** — autoencoder con memoria entrenado únicamente con "
        "latidos normales. Detecta desviaciones de la normalidad sin haber visto "
        "ejemplos de la anomalía durante el entrenamiento."
    )

# ---------------------------------------------------------------------------
# Cuerpo principal
# ---------------------------------------------------------------------------

st.markdown("## 🫀 Detección de anomalías en ECG")

if senal is None:
    st.info("Seleccione un ejemplo o cargue un fichero CSV para comenzar.")
    st.stop()

with st.spinner("Analizando señal…"):
    try:
        res = analizar_cacheado(senal, int(fs), float(umbral))
    except RuntimeError as exc:
        st.error(f"No se pudo analizar la señal: {exc}")
        st.stop()

anomalos = [lat for lat in res.latidos if lat.es_anomalo]
rachas = res.detectar_rachas(min_longitud=3)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latidos analizados", res.n_latidos)
c2.metric("Sospechosos", len(anomalos), f"{res.porcentaje_anomalos:.1f} %")
c3.metric("Frecuencia cardíaca", f"{res.frecuencia_cardiaca_media:.0f} lpm")
c4.metric("Rachas de ≥3", len(rachas))

if rachas:
    st.warning(
        f"Se han detectado **{len(rachas)}** secuencias de tres o más latidos "
        "sospechosos consecutivos, patrón de mayor relevancia clínica que las "
        "anomalías aisladas."
    )

tab1, tab2, tab3 = st.tabs(["Serie temporal", "Revisión de latidos", "Análisis"])

# -- Pestaña 1: serie temporal ---------------------------------------------

with tab1:
    st.plotly_chart(grafico_serie_temporal(res, senal, int(fs)), use_container_width=True)
    st.caption(
        "Los tramos resaltados en rojo corresponden a latidos cuyo score supera "
        "el umbral. Pase el cursor sobre los marcadores para ver el detalle de "
        "cada uno."
    )
    st.plotly_chart(grafico_scores(res), use_container_width=True)

# -- Pestaña 2: revisión ---------------------------------------------------

with tab2:
    if not anomalos:
        st.success("No se han detectado latidos sospechosos con el umbral actual.")
    else:
        orden = sorted(anomalos, key=lambda lat: lat.score, reverse=True)
        total = len(orden)
        st.session_state.idx_revision = min(st.session_state.idx_revision, total - 1)

        etiquetas = [
            f"#{lat.indice} · score {lat.score:.2f} · {lat.causa_descripcion}"
            for lat in orden
        ]

        # Los callbacks se ejecutan antes del rerun automático de Streamlit, por
        # lo que actualizan el estado sin necesidad de forzar st.rerun(), cuya
        # interrupción del script provoca conflictos con el runtime de PyTorch.
        def ir_anterior() -> None:
            st.session_state.idx_revision = max(0, st.session_state.idx_revision - 1)

        def ir_siguiente() -> None:
            st.session_state.idx_revision = min(
                total - 1, st.session_state.idx_revision + 1
            )

        def desde_selector() -> None:
            st.session_state.idx_revision = etiquetas.index(
                st.session_state.selector_latido
            )

        nav1, nav2, nav3, nav4 = st.columns([1, 1, 4, 2])

        nav1.button(
            "◀ Anterior", use_container_width=True, on_click=ir_anterior,
            disabled=st.session_state.idx_revision == 0,
        )
        nav2.button(
            "Siguiente ▶", use_container_width=True, on_click=ir_siguiente,
            disabled=st.session_state.idx_revision >= total - 1,
        )
        nav3.selectbox(
            "Ir a", etiquetas, index=st.session_state.idx_revision,
            label_visibility="collapsed", key="selector_latido",
            on_change=desde_selector,
        )
        nav4.markdown(
            f"<div style='text-align:right;padding-top:6px;color:{C_TEXTO};'>"
            f"<b style='font-size:1.1rem;color:#FAFAFA;'>"
            f"{st.session_state.idx_revision + 1}</b> / {total}</div>",
            unsafe_allow_html=True,
        )

        lat = orden[st.session_state.idx_revision]

        izq, der = st.columns([3, 1])

        with izq:
            st.plotly_chart(grafico_reconstruccion(lat, int(fs)), use_container_width=True)

        with der:
            color = COLOR_CAUSA.get(lat.causa, "#8D99AE")
            st.markdown(
                f"<div style='background:{color}22;border-left:3px solid {color};"
                f"padding:10px 14px;border-radius:4px;margin-bottom:14px;'>"
                f"<div style='font-size:0.7rem;letter-spacing:0.08em;color:{C_TEXTO};'>"
                f"COMPONENTE DOMINANTE</div>"
                f"<div style='font-size:1.02rem;font-weight:600;line-height:1.35;'>"
                f"{lat.causa_descripcion}</div></div>",
                unsafe_allow_html=True,
            )

            st.metric("Score de anomalía", f"{lat.score:.2f}")
            st.metric("Desviación en la forma", f"{lat.z_morfologia:.2f} σ")
            st.metric("Desviación en el ritmo", f"{lat.z_ritmo:.2f} σ")
            st.metric("Instante", f"{lat.posicion_muestra / fs:.1f} s")

        st.caption(
            "La línea naranja representa la morfología que el modelo esperaría de "
            "un latido normal; el área sombreada señala dónde se concentra la "
            "discrepancia. El componente indicado describe si la desviación se "
            "manifiesta en la forma del latido, en su ritmo o en ambos: el sistema "
            "detecta anomalías, no identifica el tipo de arritmia."
        )

        st.plotly_chart(
            grafico_scores(res, indice_activo=lat.indice), use_container_width=True
        )

# -- Pestaña 3: análisis ---------------------------------------------------

with tab3:
    izq, der = st.columns(2)

    with izq:
        st.markdown("#### Forma frente a ritmo")
        st.caption(
            "Cada punto es un latido. El eje horizontal mide la desviación de "
            "forma y el vertical la de ritmo, lo que permite distinguir el "
            "origen de cada anomalía."
        )
        st.plotly_chart(grafico_dispersion(res), use_container_width=True)

    with der:
        st.markdown("#### Distribución de scores")
        st.caption(
            "La mayoría de latidos se concentra por debajo del umbral; la cola "
            "derecha corresponde a los casos sospechosos."
        )
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=res.scores, nbinsx=60, marker_color=C_SENAL, opacity=0.85,
            hovertemplate="score %{x:.2f}<br>%{y} latidos<extra></extra>",
        ))
        fig.add_vline(
            x=res.umbral, line_dash="dash", line_color=C_ANOMALO,
            annotation_text="umbral", annotation_font=dict(color=C_TEXTO, size=11),
        )
        fig.update_layout(**layout_base(
            400, xaxis_title="Score", yaxis_title="Nº de latidos", showlegend=False,
        ))
        st.plotly_chart(fig, use_container_width=True)

    if anomalos:
        st.markdown("#### Reparto por componente dominante")
        conteo = pd.Series([lat.causa for lat in anomalos]).value_counts()
        cols = st.columns(max(len(conteo), 1))
        for col, (causa, n) in zip(cols, conteo.items()):
            col.metric(str(causa).capitalize(), n)

    st.markdown("#### Resultados detallados")
    tabla = pd.DataFrame([{
        "latido": lat.indice,
        "tiempo_s": round(lat.posicion_muestra / fs, 3),
        "score": round(lat.score, 4),
        "sospechoso": lat.es_anomalo,
        "z_forma": round(lat.z_morfologia, 4),
        "z_ritmo": round(lat.z_ritmo, 4),
        "componente": lat.causa,
    } for lat in res.latidos])

    st.dataframe(tabla.head(50), use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar resultados completos (CSV)",
        tabla.to_csv(index=False).encode("utf-8"),
        file_name="resultados_ecg.csv", mime="text/csv",
    )

st.markdown("---")
st.caption(
    f"MemAE híbrido (forma + ritmo) · w_rr = {obtener_detector().w_rr} · "
    f"umbral por defecto calibrado al {FPR_OBJETIVO:.0%} de falsos positivos sobre "
    "población de referencia · Herramienta de apoyo al análisis; no sustituye al "
    "criterio clínico."
)