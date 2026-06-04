
# Visualization Module
# Generates publication-quality plots with Error Bars.
# HE-FL-IoT Framework — 12 algorithms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from typing import Dict

class Visualizer:
    """Handles all plotting logic for HE-FL-IoT comparative analysis."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        # Set professional style
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
        
        self.styles = {
            # --- Proposed (highlighted) ---
            'HE-FL-IoT (Proposed)': {'color': '#d62728', 'linewidth': 2.5, 'linestyle': '-', 'marker': 'o', 'markersize': 8},
            # --- Baselines ---
            'Vanilla FedAvg':  {'color': '#1f77b4', 'linewidth': 1.5, 'linestyle': '-', 'marker': 's', 'markersize': 6},
            'FedProx':         {'color': '#2ca02c', 'linewidth': 1.5, 'linestyle': '-', 'marker': '^', 'markersize': 6},
            'SCAFFOLD':        {'color': '#9467bd', 'linewidth': 1.5, 'linestyle': '-',  'marker': 'v', 'markersize': 6},
            'DP-FedAvg':       {'color': '#ff7f0e', 'linewidth': 1.5, 'linestyle': '-',  'marker': 'D', 'markersize': 5},
            'FedBN':           {'color': '#17becf', 'linewidth': 1.5, 'linestyle': '-', 'marker': 'p', 'markersize': 6},
            'FedNova':         {'color': '#bcbd22', 'linewidth': 1.5, 'linestyle': '-', 'marker': 'h', 'markersize': 6},
            'FLASHE':          {'color': '#e377c2', 'linewidth': 1.5, 'linestyle': '-',  'marker': 'X', 'markersize': 6},
            'BatchCrypt':      {'color': '#8c564b', 'linewidth': 1.5, 'linestyle': '-', 'marker': '*', 'markersize': 7},
            # --- Ablation ---
            'FL + CKKS HE Only':              {'color': '#7f7f7f', 'linewidth': 1.5, 'linestyle': '--',  'marker': 'd', 'markersize': 5},
            'FL + Gradient Compression Only':  {'color': '#aec7e8', 'linewidth': 1.5, 'linestyle': '--',  'marker': '<', 'markersize': 6},
            'FL + Adaptive DP Only':           {'color': '#ffbb78', 'linewidth': 1.5, 'linestyle': '--', 'marker': '>', 'markersize': 6},
        }
    
    def _plot_metric_with_error_bars(self, metrics_data: Dict, metric_key: str, 
                                     title: str, ylabel: str, filename: str):
        """Generic helper to plot any metric with discrete markers and vertical error bars."""
        
        plt.figure(figsize=(12, 7))
        
        for algo, data in metrics_data.items():
            rounds = np.array(data['rounds'])
            mean_vals = np.array(data[metric_key]['mean'])
            std_vals = np.array(data[metric_key]['std'])
            
            style = self.styles.get(algo, {'color': 'gray', 'marker': 'o', 'markersize': 6})
            c = style.get('color', 'black')
            ls = style.get('linestyle', '-')
            lw = style.get('linewidth', 1.5)
            marker = style.get('marker', 'o')
            ms = style.get('markersize', 6)
            
            # Plot Mean Line with discrete markers every 20 rounds
            plt.plot(rounds, mean_vals, label=algo, color=c, linestyle=ls, linewidth=lw, 
                     marker=marker, markersize=ms, markevery=20)
            
            # Discrete error bars every 20 rounds
            plt.errorbar(rounds[::20], mean_vals[::20], yerr=std_vals[::20], 
                         fmt='none', ecolor=c, alpha=0.7, capsize=3, elinewidth=1.5)
            
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Communication Rounds', fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=9, frameon=True, fancybox=True, shadow=True, ncol=2, loc='best')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_path}")

    def plot_all_metrics(self, metrics: Dict):
        """Generates line charts for Accuracy, F1, AUC, and MAE."""
        
        self._plot_metric_with_error_bars(
            metrics, 'accuracy', 
            'Convergence Analysis: Test Accuracy vs Communication Rounds', 
            'Test Accuracy (%)', 
            'comparison_accuracy.png'
        )
        
        self._plot_metric_with_error_bars(
            metrics, 'f1', 
            'F1-Score Analysis', 'F1 Score', 
            'comparison_f1.png'
        )
        
        self._plot_metric_with_error_bars(
            metrics, 'auc', 
            'Area Under Curve (AUC) vs Rounds', 'AUC Score', 
            'comparison_auc.png'
        )
        
        self._plot_metric_with_error_bars(
            metrics, 'mae', 
            'Mean Absolute Error (MAE) vs Rounds', 'Mean Absolute Error', 
            'comparison_mae.png'
        )

    def plot_heatmap(self, radar_data: Dict):
        """Generates an annotated Heatmap for Holistic Performance."""
        labels = radar_data['labels']
        algos = list(radar_data['data'].keys())
        data_matrix = np.array([radar_data['data'][algo] for algo in algos])
        
        plt.figure(figsize=(10, 8))
        cmap = sns.color_palette("YlGnBu", as_cmap=True)
        
        ax = sns.heatmap(data_matrix, annot=True, fmt=".1f", cmap=cmap, 
                         xticklabels=labels, yticklabels=algos,
                         linewidths=.5, cbar_kws={"label": "Score (0-10)"})
        
        plt.title('Holistic Performance Assessment (Heatmap)', fontsize=16, fontweight='bold', pad=20)
        
        # Highlight the proposed method row in bold red
        for i, algo in enumerate(algos):
            if "HE-FL-IoT" in algo:
                ax.get_yticklabels()[i].set_color("red")
                ax.get_yticklabels()[i].set_weight("bold")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'comparison_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_path}")

    def plot_pareto_frontier(self, metrics: Dict, radar_data: Dict):
        """Generates a Pareto Frontier Scatter Plot for Privacy vs. Accuracy."""
        plt.figure(figsize=(12, 7))
        
        points = []
        radar_map = radar_data['data']
        
        for algo, data in metrics.items():
            if algo not in radar_map: continue
            
            final_acc = data['accuracy']['mean'][-1]
            priv_score = radar_map[algo][1]  # Index 1 is Privacy in standard HE-FL-IoT labels
            points.append((priv_score, final_acc, algo))
            
        # Draw all points
        for priv_score, final_acc, algo in points:
            style = self.styles.get(algo, {'color': 'black', 'marker': 'o', 'markersize': 6})
            c = style.get('color', 'black')
            marker = style.get('marker', 'o')
            ms = style.get('markersize', 6) * 1.5  # Make them slightly larger for scatter
            
            plt.scatter(priv_score, final_acc, color=c, marker=marker, s=ms**2, 
                        edgecolors="black", linewidth=1.0, zorder=3)
            
            # Optimal Zone Highlighting & Text
            offset_x = 0
            offset_y = 10
            
            kwargs = dict(xytext=(offset_x, offset_y), textcoords='offset points', 
                          ha='center', fontsize=9, zorder=4)
            if "Proposed" in algo:
                kwargs['fontweight'] = 'bold'
                kwargs['bbox'] = dict(boxstyle="round,pad=0.3", fc="#ffdddd", ec="red", lw=1)
            else:
                kwargs['bbox'] = dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8, lw=0.5)
                
            plt.annotate(algo, (priv_score, final_acc), **kwargs)

        # Calculate Pareto Frontier (Maximums)
        # Sort by x first
        points.sort(key=lambda x: x[0])
        pareto_front = []
        max_y = -np.inf
        for p in reversed(points):
            if p[1] > max_y:
                pareto_front.append(p)
                max_y = p[1]
        pareto_front.reverse()
        
        if pareto_front:
            p_x = [p[0] for p in pareto_front]
            p_y = [p[1] for p in pareto_front]
            # Draw step line for Pareto front
            plt.step(p_x, p_y, where='post', color='darkgreen', linestyle='--', linewidth=2, zorder=2, label="Pareto Frontier")
            
        plt.fill_between([8, 11], [85, 85], [100, 100], color='lightgreen', alpha=0.15, zorder=1)
        plt.text(9.5, 89, 'OPTIMAL ZONE', fontsize=12, fontweight='bold', color='darkgreen', ha='center', alpha=0.5)

        plt.title('Pareto Frontier: Privacy vs. Accuracy', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Privacy Security Score (Theoretical, 0-10)', fontsize=14, fontweight='bold')
        plt.ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
        
        plt.xlim(-0.5, 10.5)
        # Dynamic Y limits
        min_acc = min(p[1] for p in points)
        max_acc = max(p[1] for p in points)
        plt.ylim(max(0, min_acc - 5), min(100, max_acc + 5))
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'comparison_pareto.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_path}")
