# HeartECG: a patient-disjoint benchmark of ECG-image models with zero-shot testing on real printouts

Code and results for the manuscript

> Q. M. Rizvi, S. Chakravarty, S. Talat, *Heart-disease classification and myocardial-infarction detection from
> 12-lead ECG images: a patient-disjoint benchmark of classical, convolutional, transformer and fusion models
> with zero-shot testing on real printouts* (submitted to Biomedical Signal Processing and Control, 2026).

PTB-XL 12-lead signals are rendered as realistic 3x4 printout images with printout randomisation, and classical
machine-learning pipelines (HOG/LBP/ink-geometry descriptors with LR, SVM, RF, XGBoost), four pretrained deep
backbones (ResNet-50, EfficientNetV2-S, ConvNeXt-T, ViT-S/16), a three-stream fusion network (TriStream-ECG)
and a 1-D signal reference are compared under one patient-disjoint protocol, then applied zero-shot to 928 real
ECG printouts from an independent hospital.

Headline results: ConvNeXt-T reaches a macro-AUROC of 0.936 on the PTB-XL test fold (MI AUROC 0.939) and a
zero-shot AUROC of 0.96 for infarction vs normal on real printouts; the fusion network gives no gain over the
single backbone, while printout randomisation is decisive for robustness and transfer.

## Contents

| Path | Description |
|---|---|
| `code/heartecg.py` | The complete pipeline: data preparation, rendering, training, evaluation, figures |
| `code/HeartECG_Colab.ipynb` | Self-contained Google Colab notebook that runs every stage on a GPU runtime |
| `results/results.json` | All numbers reported in the paper, as produced by `heartecg.py analyse` |
| `figures/` | The figures of the paper as produced by the pipeline |

## Reproducing the study

Open `code/HeartECG_Colab.ipynb` in Google Colab on a GPU runtime (the paper used an NVIDIA RTX PRO 6000
Blackwell; a T4 works but is several times slower) and run the cells top to bottom. The notebook

1. downloads PTB-XL v1.0.3 (PhysioNet) and the ECG Images dataset of Cardiac Patients (Mendeley Data),
2. runs a smoke test, the classical-ML stage, all deep trainings (3 seeds + ablations), Grad-CAM, and
3. writes `HEART_RESULTS.zip` with `results.json` and all figures to your Google Drive.

Stages are resumable; finished runs are skipped. The same stages can be run locally:

```
python heartecg.py data && python heartecg.py ml && python heartecg.py train_all && \
python heartecg.py explain && python heartecg.py analyse
```

## Data

- PTB-XL v1.0.3: https://doi.org/10.13026/kfzx-aw45 (Creative Commons; see PhysioNet)
- ECG Images dataset of Cardiac Patients: https://doi.org/10.17632/gwbz3fsgp8.2 (CC BY 4.0)

Neither dataset is redistributed here; the notebook downloads both from their official sources.

## License

MIT for the code in this repository. The datasets keep their own licenses.
