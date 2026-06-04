Privacy-Preserving Federated Learning for Healthcare IoT Using CKKS Homomorphic Encryption and Patient-Level Differential Privacy

What this work is about

A comprehensive simulation framework comparing 12 federated learning algorithms for privacy-preserving healthcare IoT analytics. The proposed HE-FL-IoT framework combines CKKS Homomorphic Encryption, Gradient Compression, and Adaptive Differential Privacy for secure collaborative model training across healthcare IoT devices.

We have used following algorithms to compare our work

 
| No | Algorithm | Citation |
|---|-----------|----------|
| 1 | Vanilla FedAvg | McMahan et al., AISTATS 2017 |
| 2 | FedProx | Li et al., MLSys 2020 |
| 3 | SCAFFOLD | Karimireddy et al., ICML 2020 |
| 4 | DP-FedAvg | McMahan et al., ICLR 2018 |
| 5 | FedBN | Li et al., ICLR 2021 |
| 6 | FedNova | Wang et al., NeurIPS 2020 |
| 7 | FLASHE | Li et al., IEEE INFOCOM 2022 |
| 8 | BatchCrypt | Zhang et al., USENIX ATC 2020 |

--> Ablation Study  varients
| NO | Variant |
|---|---------|
| 9 | FL + CKKS HE Only |
| 10 | FL + Gradient Compression Only |
| 11 | FL + Adaptive DP Only |

--> Proposed( Our method) (1)
| No| Method |
|---|--------|
| 12 | HE-FL-IoT (Proposed) |

Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Output Files
```
results/
├── comprehensive_metrics.json    # Raw simulation data
├── radar_metrics.json            # Radar chart data
├── statistical_report.txt        # T-test + Cohen's d report
├── comparison_accuracy.png       # Accuracy convergence
├── comparison_f1.png             # F1-Score convergence
├── comparison_auc.png            # AUC convergence
├── comparison_mae.png            # MAE convergence
├── comparison_radar.png          # Holistic radar chart
├── comparison_tradeoff.png       # Privacy vs Accuracy bubble
├── Privacy_Fairness_Table.csv    # Fairness comparison
└── Cryptographic_Overhead_Metrics.csv # Overhead analysis
```

## Project Structure
```
HE_FL_IoT/
├── main.py                       # Entry point
├── requirements.txt
├── README.md
├── src/
│   ├── config.py                 # SimulationConfig
│   ├── algorithms/               # All 12 algorithm implementations
│   ├── simulation/               # Engine + Scenarios + Dataset
│   ├── analysis/                 # Statistical tests
│   └── visualization/            # Publication-quality plots
├── results/                      # Generated outputs
└── Supplementary_Data_Sheet/     # Raw CSV data
```

--> Privacy Engine (4 Layers)
1. Adaptive Gradient Clipping — per-round percentile-based sensitivity calibration
2. CKKS Approximate HE — encrypted model aggregation with floating-point support
3. Gradient Compression — top-k sparsification + quantization for bandwidth reduction
4. Patient-Level DP — per-patient noise injection with Renyi privacy accounting
