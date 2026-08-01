# Detección de anomalías en ECG mediante autoencoder con memoria

Sistema de detección de latidos anómalos en electrocardiograma entrenado
**exclusivamente con latidos normales**. A diferencia de un clasificador
supervisado, es capaz de identificar arritmias que nunca vio durante el
entrenamiento (*open-set detection*).

**Aplicación desplegada:** [enlace pendiente]

---

## Motivación

Un clasificador supervisado solo puede responder con las clases que aprendió.
Ante una arritmia que no estaba en su conjunto de entrenamiento, la asigna
forzosamente a alguna categoría conocida, sin emitir ninguna señal de alerta.
En un escenario clínico real —donde no es posible anticipar todas las
patologías— esta limitación es relevante.

Este trabajo aborda el problema desde la detección de anomalías: en lugar de
aprender fronteras entre categorías, el modelo aprende **qué es un latido
normal** y señala cualquier desviación, con independencia de su tipo.

Para validarlo se excluyó deliberadamente la clase V (extrasístoles
ventriculares) del conjunto de entrenamiento, evaluando después la capacidad
del sistema para detectarla como anómala.

---

## Resultados

Evaluación sobre el conjunto de test de MIT-BIH (36.881 latidos de pacientes
no vistos durante el entrenamiento):

| Modelo | AUROC (V) | AUPRC (V) | Recall V @ FPR 10 % |
|---|---|---|---|
| Autoencoder convolucional | 0,795 | 0,448 | 65,0 % |
| Híbrido (forma + ritmo) | 0,910 | 0,583 | 73,6 % |
| **MemAE (modelo final)** | **0,957** | **0,743** | **93,9 %** |

Bajo la definición clínica de normalidad (únicamente latidos N), el modelo
final alcanza un AUROC global de 0,950 y detecta el 96,2 % de las
extrasístoles ventriculares, el 92,1 % de las supraventriculares y el 95,9 %
de los latidos de marcapasos, con un 10 % de falsas alarmas sobre latidos
sinusales normales.

---

## Arquitectura

El modelo combina dos fuentes de información complementarias:

```
   Señal (216 puntos, 600 ms)      Intervalos RR (4 características)
              │                                │
      Encoder convolucional 1D          Encoder denso
              │                                │
              └──────────┬─────────────────────┘
                         │
                  Espacio latente (14)
                         │
                  ┌──────▼──────┐
                  │   MEMORIA   │   100 prototipos de normalidad
                  └──────┬──────┘
                         │
              ┌──────────┴─────────────────────┐
              │                                │
      Decoder convolucional             Decoder denso
              │                                │
      Señal reconstruida            Intervalos RR reconstruidos
```

**Rama de morfología.** Convoluciones 1D que capturan la forma del latido:
patrones locales invariantes a pequeños desplazamientos temporales.

**Rama de ritmo.** Cuatro características derivadas de los intervalos RR
(previo, siguiente y sus ratios respecto al ritmo basal local del paciente).
Aportan la información de prematuridad y pausa compensatoria, ausente al
procesar cada latido de forma aislada.

**Módulo de memoria.** Un autoencoder convencional puede generalizar tanto que
reconstruye correctamente las propias anomalías, reduciendo su error y
dificultando la detección. El módulo de memoria (Gong et al., 2019) impide que
el decoder reconstruya libremente: solo puede combinar prototipos de
normalidad aprendidos durante el entrenamiento, lo que amplifica el error
sobre entradas atípicas.

**Score de anomalía.** Los errores de reconstrucción de ambas ramas se
estandarizan por separado antes de combinarse. Sin esta estandarización, la
componente de mayor magnitud numérica domina la suma con independencia de su
capacidad discriminativa real.

---

## Instalación

Requiere Python 3.12.

```bash
git clone https://github.com/<usuario>/ecg-anomaly-detection.git
cd ecg-anomaly-detection

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Uso

### Interfaz web

```bash
streamlit run app/dashboard.py
```

Se abre en `http://localhost:8501`. Incluye dos registros de ejemplo
precargados, por lo que puede probarse sin descargar datos.

La aplicación permite:

- Cargar una señal ECG en formato CSV o utilizar los ejemplos incluidos.
- Visualizar la señal completa con los latidos sospechosos resaltados.
- Revisar cada anomalía comparando la señal real con la reconstrucción del
  modelo, con la zona de mayor discrepancia señalada.
- Ajustar el umbral de decisión según la sensibilidad deseada.
- Descargar los resultados por latido en CSV.

### Reproducir el entrenamiento

```bash
# 1. Descargar el dataset MIT-BIH (unos 100 MB)
python src/data/download_data.py

# 2. Ejecutar el notebook
jupyter notebook notebooks/01_exploracion.ipynb
```

El notebook recorre el proceso completo: exploración, preprocesamiento,
clasificador baseline, y las tres iteraciones del modelo de detección.

### Uso programático

```python
import wfdb
from src.inference.predict import cargar_detector

detector = cargar_detector()
registro = wfdb.rdrecord("data/raw/mitdb/208")
resultado = detector.analizar(registro.p_signal[:, 0], registro.fs)

print(resultado.resumen())

for latido in resultado.top_anomalos(5):
    print(f"t={latido.posicion_muestra / registro.fs:.1f}s  "
          f"score={latido.score:.2f}  ({latido.causa})")
```

---

## Estructura del proyecto

```
├── app/
│   ├── dashboard.py           Interfaz Streamlit
│   └── ejemplos/              Señales de demostración
├── data/
│   ├── raw/mitdb/             Dataset (no versionado)
│   └── processed/             Splits preprocesados (no versionado)
├── models/
│   ├── memae_clinico.pt       Modelo final desplegado
│   ├── memae_hibrido.pt       Variante open-set (apartado 5)
│   └── baseline_classifier.pt Clasificador supervisado de referencia
├── notebooks/
│   └── 01_exploracion.ipynb   Experimentación completa
├── src/
│   ├── config.py              Rutas y parámetros centralizados
│   ├── data/
│   │   ├── download_data.py   Descarga del dataset
│   │   └── preprocessing.py   Segmentación y características de ritmo
│   ├── models/memae.py        Arquitectura MemAE
│   └── inference/predict.py   Carga del modelo y cálculo de scores
└── requirements.txt
```

---

## Datos

**MIT-BIH Arrhythmia Database** (PhysioNet): 48 registros de media hora
procedentes de 47 pacientes, con cada latido anotado independientemente por
dos cardiólogos.

Los latidos se agrupan según el estándar **AAMI EC57** en cinco superclases:

| Clase | Descripción |
|---|---|
| N | Latido normal o con conducción alterada |
| S | Ectópico supraventricular |
| V | Ectópico ventricular |
| F | Latido de fusión |
| Q | Marcapasos o no clasificable |

### Decisiones metodológicas

**División por paciente, no por latido.** Los latidos de un mismo paciente son
muy similares entre sí; repartirlos aleatoriamente produciría fuga de
información y métricas artificialmente optimistas. La división se realiza a
nivel de paciente, de modo que la evaluación refleja el escenario real de
enfrentarse a un corazón desconocido.

**Exclusión de la clase V del entrenamiento.** Es la arritmia objetivo del
experimento open-set: el modelo debe detectarla sin haberla visto nunca.

**Exclusión de la clase Q del entrenamiento.** Los latidos de marcapasos
presentan una espiga de estimulación eléctrica de amplitud muy superior a
cualquier morfología fisiológica (el 97 % de los valores atípicos de amplitud
del dataset pertenecen a esta clase), que contaminaría la noción de
normalidad aprendida.

**Normalización por latido.** Cada segmento se normaliza con su propia media y
desviación típica, eliminando las diferencias de amplitud entre pacientes sin
introducir fuga de información entre conjuntos.

---

## Limitaciones

**Detector, no clasificador.** El sistema señala desviaciones de la normalidad
pero no identifica el tipo de arritmia. Se evaluó la posibilidad de inferirlo
a partir de la descomposición del error en sus componentes de forma y ritmo,
obteniendo un acierto del 26,7 % —insuficiente para su uso—, lo que confirma
que un score unidimensional no contiene la información necesaria para
discriminar entre tipos de ectopia.

**Variabilidad entre pacientes.** El umbral calibrado sobre una población de
referencia produce tasas de falsos positivos dispares según el registro
(2,1 % – 24,1 %). Los valores más altos se concentran en pacientes con ectopia
ventricular muy frecuente, donde los intervalos RR de los latidos normales
adyacentes se ven alterados por proximidad. La interfaz permite ajustar el
umbral manualmente para mitigarlo.

**Latidos de fusión.** La clase F se detecta con un rendimiento limitado
(AUROC 0,68). Es un resultado esperable: por definición fisiológica se trata de
una morfología intermedia entre normal y ventricular, situada en la frontera
de decisión.

**Detección automática de picos R.** El rendimiento en producción es
ligeramente inferior al medido con las anotaciones de referencia del dataset,
ya que la localización automática de los picos introduce variabilidad en el
cálculo de los intervalos RR.

---

## Aviso

Herramienta desarrollada con fines académicos. No constituye un dispositivo
médico ni sustituye al criterio clínico profesional.

---

## Referencias

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database*.
  IEEE Eng in Med and Biol 20(3):45-50, 2001.
- Gong D, et al. *Memorizing Normality to Detect Anomaly: Memory-augmented
  Deep Autoencoder for Unsupervised Anomaly Detection*. ICCV, 2019.
- ANSI/AAMI EC57. *Testing and reporting performance results of cardiac rhythm
  and ST segment measurement algorithms*, 1998.

---

## Autor

Óscar Valcárcel González — Grado en Inteligencia Artificial, Universidad Rey
Juan Carlos. Aplicaciones de la Inteligencia Artificial, curso 2026-27.