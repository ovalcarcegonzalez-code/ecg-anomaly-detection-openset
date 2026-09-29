# ECG Anomaly Detection with a Memory-Augmented Autoencoder

A system for detecting anomalous heartbeats in electrocardiograms, trained
**exclusively on normal beats**. Unlike a supervised classifier, it can
identify arrhythmias it never saw during training (*open-set detection*).

**Deployed application:** https://ecg-anomaly-detection-openset-ovg.streamlit.app

---

## Motivation

A supervised classifier can only answer with the classes it has learned.
Faced with an arrhythmia that was not in its training set, it is forced to
assign it to some known category, without raising any warning signal. In a
real clinical setting, where it is impossible to anticipate every pathology,
this limitation matters.

This work approaches the problem from the anomaly detection angle: instead of
learning boundaries between categories, the model learns **what a normal beat
looks like** and flags any deviation, regardless of its type.

To validate this, class V (premature ventricular contractions) was
deliberately excluded from the training set, and the system's ability to
detect it as anomalous was evaluated afterwards.

---

## Results

Evaluation on the MIT-BIH test set (36,881 beats from patients unseen during
training):

| Model | AUROC (V) | AUPRC (V) | Recall V @ FPR 10% |
|---|---|---|---|
| Convolutional autoencoder | 0.795 | 0.448 | 65.0% |
| Hybrid (shape + rhythm) | 0.910 | 0.583 | 73.6% |
| **MemAE (final model)** | **0.957** | **0.743** | **93.9%** |

Under the clinical definition of normality (N beats only), the final model
achieves an overall AUROC of 0.950 and detects 96.2% of premature ventricular
contractions, 92.1% of supraventricular beats and 95.9% of paced beats, with
a 10% false alarm rate on normal sinus beats.

---

## Architecture

The model combines two complementary sources of information:

```
   Signal (216 points, 600 ms)        RR intervals (4 features)
              │                                │
      1D convolutional encoder            Dense encoder
              │                                │
              └──────────┬─────────────────────┘
                         │
                  Latent space (14)
                         │
                  ┌──────▼──────┐
                  │   MEMORY    │   100 prototypes of normality
                  └──────┬──────┘
                         │
              ┌──────────┴─────────────────────┐
              │                                │
      Convolutional decoder              Dense decoder
              │                                │
      Reconstructed signal        Reconstructed RR intervals
```

**Morphology branch.** 1D convolutions that capture the shape of the beat:
local patterns that are invariant to small temporal shifts.

**Rhythm branch.** Four features derived from RR intervals (previous, next,
and their ratios with respect to the patient's local baseline rhythm). They
provide information about prematurity and compensatory pause, which is absent
when each beat is processed in isolation.

**Memory module.** A conventional autoencoder can generalize so well that it
correctly reconstructs the anomalies themselves, lowering their error and
making detection harder. The memory module (Gong et al., 2019) prevents the
decoder from reconstructing freely: it can only combine prototypes of
normality learned during training, which amplifies the error on atypical
inputs.

**Anomaly score.** The reconstruction errors of both branches are
standardized separately before being combined. Without this standardization,
the component with the larger numerical magnitude dominates the sum
regardless of its actual discriminative power.

---

## Installation

Requires Python 3.12.

```bash
git clone https://github.com/ovalcarcegonzalez-code/ecg-anomaly-detection.git
cd ecg-anomaly-detection

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Web interface

```bash
streamlit run app/dashboard.py
```

It opens at `http://localhost:8501`. It includes two preloaded example
recordings, so it can be tried without downloading any data.

The application allows you to:

- Upload an ECG signal in CSV format or use the included examples.
- View the full signal with suspicious beats highlighted.
- Review each anomaly by comparing the real signal with the model's
  reconstruction, with the area of greatest discrepancy marked.
- Adjust the decision threshold according to the desired sensitivity.
- Download per-beat results as CSV.

### Reproducing the training

```bash
# 1. Download the MIT-BIH dataset (about 100 MB)
python src/data/download_data.py

# 2. Run the notebook
jupyter notebook notebooks/01_exploracion.ipynb
```

The notebook walks through the full process: exploration, preprocessing,
baseline classifier, and the three iterations of the detection model.

### Programmatic use

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

## Project structure

```
├── app/
│   ├── dashboard.py           Streamlit interface
│   └── ejemplos/              Demo signals
├── data/
│   ├── raw/mitdb/             Dataset
│   └── processed/             Preprocessed splits
├── models/
│   ├── memae_clinico.pt       Final deployed model
│   ├── memae_hibrido.pt       Open-set variant
│   └── baseline_classifier.pt Reference supervised classifier
├── notebooks/
│   └── 01_exploracion.ipynb   Full experimentation
├── src/
│   ├── config.py              Centralized paths and parameters
│   ├── data/
│   │   ├── download_data.py   Dataset download
│   │   └── preprocessing.py   Segmentation and rhythm features
│   ├── models/memae.py        MemAE architecture
│   └── inference/predict.py   Model loading and score computation
└── requirements.txt
```

---

## Data

**MIT-BIH Arrhythmia Database** (PhysioNet): 48 half-hour recordings from 47
patients, with each beat independently annotated by two cardiologists.

Beats are grouped according to the **AAMI EC57** standard into five
superclasses:

| Class | Description |
|---|---|
| N | Normal beat or beat with altered conduction |
| S | Supraventricular ectopic |
| V | Ventricular ectopic |
| F | Fusion beat |
| Q | Paced or unclassifiable beat |

### Methodological decisions

**Split by patient, not by beat.** Beats from the same patient are very
similar to each other; distributing them randomly would cause information
leakage and artificially optimistic metrics. The split is done at the patient
level, so the evaluation reflects the real-world scenario of facing an unseen
heart.

**Exclusion of class V from training.** It is the target arrhythmia of the
open-set experiment: the model must detect it without ever having seen it.

**Exclusion of class Q from training.** Paced beats show an electrical
pacing spike with an amplitude far above any physiological morphology (97% of
the amplitude outliers in the dataset belong to this class), which would
contaminate the learned notion of normality.

**Per-beat normalization.** Each segment is normalized with its own mean and
standard deviation, removing amplitude differences between patients without
introducing information leakage between sets.

---

## Disclaimer

Tool developed for academic purposes. It is not a medical device and does not
replace professional clinical judgment.

---

## References

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database*.
  IEEE Eng in Med and Biol 20(3):45-50, 2001.
- Gong D, et al. *Memorizing Normality to Detect Anomaly: Memory-augmented
  Deep Autoencoder for Unsupervised Anomaly Detection*. ICCV, 2019.
- ANSI/AAMI EC57. *Testing and reporting performance results of cardiac rhythm
  and ST segment measurement algorithms*, 1998.

---

## Author

Óscar Valcarce González.
