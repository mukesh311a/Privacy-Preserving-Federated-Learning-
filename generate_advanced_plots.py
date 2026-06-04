
# Generate Advanced Visualizations for our proposed study
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.visualization.advanced_plots import AdvancedVisualizer

def main():
    print("==========================================")
    print("   Advanced Visualizations Generator")
    print("==========================================\n")
    
    metrics_path = os.path.join("results", "comprehensive_metrics.json")
    radar_path = os.path.join("results", "radar_metrics.json")
    output_dir = os.path.join("results", "advanced_plots")
    
    if not os.path.exists(metrics_path) or not os.path.exists(radar_path):
        print(f"[ERROR] Required JSON data files not found in results directory.")
        print("Please run `python main.py` first to generate data.")
        sys.exit(1)
        
    print(f"[INFO] Loading data from {metrics_path}...")
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
        
    with open(radar_path, 'r') as f:
        radar_data = json.load(f)

    print(f"[INFO] Data loaded. Initializing AdvancedVisualizer...")
    viz = AdvancedVisualizer(output_dir)
    
    print("\n[INFO] Generating Box Plot...")
    viz.plot_grouped_box(metrics_data)
    
    print("\n[INFO] Generating Stacked Bar Overhead Chart...")
    viz.plot_overhead_stacked_bar()
    
    print("\n[INFO] Generating Gradient Joyplot...")
    viz.plot_gradient_joyplot()
    
    print("\n[INFO] Generating Ablation Waterfall...")
    viz.plot_ablation_waterfall(metrics_data)
    
    print("\n[INFO] Generating Parallel Coordinates Plot...")
    viz.plot_parallel_coordinates(metrics_data, radar_data)
    
    print("\n[INFO] Generating Heterogeneity Analysis Plot...")
    viz.plot_heterogeneity_analysis(metrics_data)
    
    print("\n[COMPLETE] All advanced plots successfully generated in results/advanced_plots/")

if __name__ == "__main__":
    main()
