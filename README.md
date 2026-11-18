# PDAN8412 POE Part 3 – CNN Facial Recognition (LFW)

**Student:** Adivhaho Randima (ST10128395)

Recognises known people from face images for the book press's premises, using a convolutional neural network trained on Labeled Faces in the Wild. Apache Spark is used for the exploratory analysis and for comparing per-person results.

## Files
| File | Description |
|---|---|
| `PDAN8412 Part3 CNN LFW.ipynb` | Full analysis notebook (run, with outputs) |
| `PDAN8412 Part3 Report ST10128395.pdf` | Client report |
| `model_v1_baseline.keras` | Baseline CNN |
| `model_v2_improved.keras` | Retrained CNN (augmentation + regularisation) |

## Dataset
Huang, G.B., Ramesh, M., Berg, T. and Learned-Miller, E. 2007. *Labeled Faces in the Wild*. Technical Report 07-49, University of Massachusetts, Amherst. The notebook downloads it automatically via `sklearn.datasets.fetch_lfw_people`.

## How to run
1. Python 3.11+ with `pyspark tensorflow scikit-learn pandas matplotlib seaborn` and Java 17+ (`JAVA_HOME` set).
2. Kernel → Restart & Run All (first run downloads LFW, about 200 MB; full run about 15 minutes on a laptop CPU).

## Results (test set, 454 images, 62 people)
| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Baseline CNN | 0.692 | 0.599 | 0.682 |
| Improved CNN (augmentation + dropout + L2 + mild class weights) | 0.793 | 0.743 | 0.792 |

## Environment notes (Windows)
- Spark 4.2 needs Java 17/21. The first notebook cell points `JAVA_HOME` to a JDK 17 inside the conda env (installed with `pip install install-jdk` then `jdk.install('17', path=<env>/jdk)`).
- Tested with TensorFlow 2.21, PySpark 4.2, pandas 2.3, numpy 2.4 (pip wheel).
