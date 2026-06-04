
# Main CLI Entry Point
# HE-FL-IoT: Privacy-Preserving Federated Learning with CKKS Homomorphic
# Encryption for Secure Healthcare IoT Analytics
# Orchestrates the simulation, analysis, and visualization.

import sys
import os
import json

# Ensure active directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import SimulationConfig
from src.simulation.generator import SimulationEngine
from src.analysis.stats import StatisticalAnalyzer
from src.visualization.plots import Visualizer

def main():
    print("==========================================")
    print("   HE-FL-IoT Comparative Analysis System  ")
    print("   v1.0.0 - Enterprise Simulation Framework")
    print("==========================================\n")
    
    # 1. Initialize Configuration
    config = SimulationConfig()
    print(f"[INFO] Initialized configuration. Output Dir: {config.OUTPUT_DIR}")
    print(f"[INFO] Simulation Mode: Monte Carlo ({config.NUM_SEEDS} seeds) for Error Bars.")
    print(f"[INFO] Algorithms: {len(config.ALGORITHMS)} total (8 baselines + 3 ablation + 1 proposed)")
    
    # 2. Init Engine
    engine = SimulationEngine(config)
    
    # 3. Running Simulation
    print(f"\n[INFO] Starting Simulation for {config.NUM_ROUNDS} rounds...")
    metrics_data = engine.run_simulation()
    radar_data = engine.extract_radar_metrics()
    
    # 3.1 Sensitivity Analysis (Reviewer 1)
    engine.run_sensitivity_analysis()
    
    # Save Raw Data
    metrics_path = os.path.join(config.OUTPUT_DIR, config.METRICS_FILE)
    radar_path = os.path.join(config.OUTPUT_DIR, config.RADAR_FILE)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    with open(radar_path, 'w') as f:
        json.dump(radar_data, f, indent=4)
        
    print(f"[SUCCESS] Simulation complete. Logs saved to {metrics_path}")
    
    # 4. Statistical Analysis
    print("\n[INFO] Performing Statistical Analysis (on Mean Accuracy)...")
    
    # Shim to make metrics_data compatible with old analyzer expectation
    flat_metrics = {}
    for algo, vals in metrics_data.items():
        flat_metrics[algo] = {'accuracy': vals['accuracy']['mean']}
        
    analyzer = StatisticalAnalyzer(flat_metrics)
    report = analyzer.generate_report([
        # Proposed vs Baselines
        ("HE-FL-IoT (Proposed)", "Vanilla FedAvg"),
        ("HE-FL-IoT (Proposed)", "FedProx"),
        ("HE-FL-IoT (Proposed)", "SCAFFOLD"),
        ("HE-FL-IoT (Proposed)", "DP-FedAvg"),
        ("HE-FL-IoT (Proposed)", "FedBN"),
        ("HE-FL-IoT (Proposed)", "FedNova"),
        ("HE-FL-IoT (Proposed)", "FLASHE"),
        ("HE-FL-IoT (Proposed)", "BatchCrypt"),
        # Ablation Comparisons
        ("HE-FL-IoT (Proposed)", "FL + CKKS HE Only"),
        ("HE-FL-IoT (Proposed)", "FL + Gradient Compression Only"),
        ("HE-FL-IoT (Proposed)", "FL + Adaptive DP Only"),
    ])
    
    report_path = os.path.join(config.OUTPUT_DIR, config.STATS_REPORT)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"[SUCCESS] Statistical Report generated at {report_path}")
    print(report)
    
    # 5. Visualization
    print("\n[INFO] Generating Visualizations...")
    viz = Visualizer(config.OUTPUT_DIR)
    
    viz.plot_all_metrics(metrics_data)
    viz.plot_heatmap(radar_data)
    viz.plot_pareto_frontier(metrics_data, radar_data)
    
    print("\n[COMPLETE] All tasks finished successfully.")
    print("Generated outputs:")
    print(f"  - {metrics_path}")
    print(f"  - {radar_path}")
    print(f"  - {report_path}")
    print(f"  - results/comparison_accuracy.png")
    print(f"  - results/comparison_f1.png")
    print(f"  - results/comparison_auc.png")
    print(f"  - results/comparison_mae.png")
    print(f"  - results/comparison_heatmap.png")
    print(f"  - results/comparison_pareto.png")

if __name__ == "__main__":
    main()
