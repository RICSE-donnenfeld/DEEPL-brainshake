"""
Visualization module for EEG signals and metrics.

Generates comprehensive visualizations of:
- Raw EEG signals
- Spectral analysis
- Computed metrics
- Feature comparisons
- Statistical summaries

All outputs saved to output folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import warnings

from brainshake.features import (
    compute_hjorth_parameters,
    compute_band_powers,
    compute_band_power_ratios,
    compute_sample_entropy,
    compute_permutation_entropy,
    compute_line_length,
    compute_turning_points_ratio,
    compute_channel_features,
    compute_mean_coherence,
    compute_phase_locking_value,
    compute_spectral_entropy,
    compute_all_advanced_features,
)


plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'ggplot')
warnings.filterwarnings('ignore')


class EEGVisualizer:
    """
    Comprehensive EEG visualization and analysis tool.
    
    Generates visualizations and saves them to output directory.
    """
    
    def __init__(self, output_dir: str = "outputs", figsize: Tuple[int, int] = (12, 8)):
        """
        Args:
            output_dir: Directory to save visualizations
            figsize: Default figure size
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.figsize = figsize
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subdirectories
        (self.output_dir / "signals").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "spectra").mkdir(exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        print(f"Timestamp: {self.timestamp}")
    
    def plot_eeg_signal(
        self,
        eeg_window: np.ndarray,
        title: str = "EEG Signal",
        save_name: Optional[str] = None,
        sfreq: float = 256,
        show_channels: int = 6,
    ) -> plt.Figure:
        """
        Plot EEG channels as multi-line plot.
        
        Args:
            eeg_window: EEG data [n_channels, n_samples]
            title: Plot title
            save_name: Filename to save (without extension)
            sfreq: Sampling frequency
            show_channels: Number of channels to display
        """
        n_channels, n_samples = eeg_window.shape
        time = np.arange(n_samples) / sfreq
        
        n_show = min(show_channels, n_channels)
        fig, axes = plt.subplots(n_show, 1, figsize=(14, 2 * n_show), sharex=True)
        
        if n_show == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.plot(time, eeg_window[i], linewidth=0.8, color=f'C{i}')
            ax.set_ylabel(f'Ch {i+1}')
            ax.set_title(f'Channel {i+1}', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'{title} - First {n_show} Channels', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name, folder="signals")
        
        return fig
    
    def plot_spectrogram(
        self,
        eeg_window: np.ndarray,
        channel_idx: int = 0,
        title: str = "Spectrogram",
        save_name: Optional[str] = None,
        sfreq: float = 256,
    ) -> plt.Figure:
        """
        Plot spectrogram for a single channel.
        """
        from scipy.signal import spectrogram as scipy_spectrogram
        
        signal = eeg_window[channel_idx]
        f, t, Sxx = scipy_spectrogram(signal, fs=sfreq, nperseg=32, noverlap=16)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylim([0, sfreq/2])
        ax.set_title(f'{title} - Channel {channel_idx + 1}', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, label='Power (dB)')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name, folder="spectra")
        
        return fig
    
    def plot_band_powers(
        self,
        eeg_window: np.ndarray,
        title: str = "Band Powers",
        save_name: Optional[str] = None,
        sfreq: float = 256,
    ) -> plt.Figure:
        """
        Plot power in different frequency bands as bar chart.
        """
        band_powers = compute_band_powers(eeg_window, sfreq)
        
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Average across channels
        powers = [np.mean(band_powers[band]) for band in bands]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(bands, powers, color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Frequency Band', fontsize=12)
        ax.set_ylabel('Power (μV²)', fontsize=12)
        ax.set_title(f'{title} - Average Power by Band', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, power in zip(bars, powers):
            height = bar.get_height()
            ax.annotate(f'{power:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name, folder="spectra")
        
        return fig
    
    def plot_metric_comparison(
        self,
        metrics_dict: Dict[str, float],
        title: str = "EEG Metrics",
        save_name: Optional[str] = None,
        top_n: int = 10,
    ) -> plt.Figure:
        """
        Plot comparison of computed metrics as horizontal bar chart.
        """
        # Sort by value
        sorted_metrics = sorted(metrics_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        labels = [m[0] for m in sorted_metrics[:top_n]]
        values = [m[1] for m in sorted_metrics[:top_n]]
        
        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4)))
        y_pos = np.arange(len(labels))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Value', fontsize=12)
        ax.set_title(f'{title} - Top {top_n} Metrics', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name, folder="metrics")
        
        return fig
    
    def plot_channel_correlation_matrix(
        self,
        eeg_window: np.ndarray,
        title: str = "Channel Correlations",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot channel correlation matrix as heatmap.
        """
        from brainshake.features import compute_channel_correlation
        
        corr_matrix = compute_channel_correlation(eeg_window)
        n_channels = corr_matrix.shape[0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xlabel('Channel', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        ax.set_title(f'{title} - Correlation Matrix ({n_channels} channels)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Correlation')
        
        # Add channel labels
        if n_channels <= 21:
            ax.set_xticks(np.arange(n_channels))
            ax.set_yticks(np.arange(n_channels))
            ax.set_xticklabels([f'Ch{i+1}' for i in range(n_channels)], fontsize=8)
            ax.set_yticklabels([f'Ch{i+1}' for i in range(n_channels)], fontsize=8)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name, folder="metrics")
        
        return fig
    
    def plot_metrics_radar_chart(
        self,
        metrics: Dict[str, float],
        title: str = "EEG Metrics Radar",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot metrics as radar/spider chart.
        """
        # Select key metrics for radar
        key_metrics = [
            'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
            'spectral_entropy', 'mean', 'std', 'skewness', 'kurtosis',
            'turning_points', 'waveform_complexity'
        ]
        
        available = [m for m in key_metrics if m in metrics]
        if len(available) < 3:
            return None
        
        values = [metrics.get(m, 0) for m in available]
        
        # Normalize values to 0-1 for radar
        max_val = max(abs(v) for v in values) if max(abs(v) for v in values) != 0 else 1
        values_normalized = [v / max_val for v in values]
        
        n_metrics = len(available)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        values_plot = values_normalized + [values_normalized[0]]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(angles, values_plot, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values_plot, alpha=0.25, color='steelblue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available, fontsize=10)
        ax.set_title(f'{title}', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name, folder="metrics")
        
        return fig
    
    def plot_feature_distributions(
        self,
        eeg_windows: List[np.ndarray],
        labels: Optional[List[int]] = None,
        title: str = "Feature Distributions",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot distribution of features across multiple EEG windows.
        
        Args:
            eeg_windows: List of EEG windows
            labels: Optional labels (0=normal, 1=seizure) for color coding
            title: Plot title
            save_name: Filename to save
        """
        n_windows = len(eeg_windows)
        n_features = 6  # Number of features to display
        
        # Compute features for each window
        features_list = []
        for window in eeg_windows:
            feats = compute_all_advanced_features(window)
            features_list.append(feats)
        
        # Select features to plot
        feature_names = ['hjorth_mobility', 'hjorth_complexity', 'spectral_entropy',
                        'seizure_index', 'turning_points', 'waveform_complexity']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['#2ca02c', '#d62728']  # Green for normal, red for seizure
        
        for i, feat_name in enumerate(feature_names):
            ax = axes[i]
            values = [[] for _ in range(2)]
            
            for j, feats in enumerate(features_list):
                if labels and j < len(labels):
                    values[labels[j]].append(feats.get(feat_name, 0))
                else:
                    values[0].append(feats.get(feat_name, 0))
            
            for c, vals in enumerate(values):
                if vals:
                    ax.hist(vals, bins=15, alpha=0.6, color=colors[c], 
                           label=['Normal', 'Seizure'][c] if labels else 'All',
                           edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel(feat_name.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.legend(fontsize=8)
            ax.set_title(f'{feat_name.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{title} - Feature Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name, folder="comparisons")
        
        return fig
    
    def plot_time_series_metrics(
        self,
        eeg_windows: List[np.ndarray],
        window_duration: float = 0.5,
        sfreq: float = 256,
        title: str = "Metrics Over Time",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot how metrics change over multiple consecutive windows.
        
        Args:
            eeg_windows: List of consecutive EEG windows
            window_duration: Duration of each window in seconds
            sfreq: Sampling frequency
            title: Plot title
            save_name: Filename to save
        """
        n_windows = len(eeg_windows)
        time_points = np.arange(n_windows) * window_duration
        
        # Compute metrics for each window
        metrics_over_time = {
            'activity': [],
            'mobility': [],
            'complexity': [],
            'seizure_index': [],
            'line_length': [],
            'spectral_entropy': [],
        }
        
        for window in eeg_windows:
            all_feats = compute_all_advanced_features(window)
            metrics_over_time['activity'].append(all_feats.get('hjorth_activity', 0))
            metrics_over_time['mobility'].append(all_feats.get('hjorth_mobility', 0))
            metrics_over_time['complexity'].append(all_feats.get('hjorth_complexity', 0))
            metrics_over_time['seizure_index'].append(all_feats.get('seizure_index', 0))
            metrics_over_time['line_length'].append(all_feats.get('line_length', 0))
            metrics_over_time['spectral_entropy'].append(all_feats.get('spectral_entropy', 0))
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (metric_name, values) in enumerate(metrics_over_time.items()):
            ax = axes[i]
            ax.plot(time_points, values, color=colors[i], linewidth=1.5, marker='o', markersize=3)
            ax.fill_between(time_points, values, alpha=0.3, color=colors[i])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{title}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name, folder="metrics")
        
        return fig
    
    def create_comprehensive_report(
        self,
        eeg_window: np.ndarray,
        label: Optional[int] = None,
        sfreq: float = 256,
        sample_name: str = "sample",
    ) -> Dict[str, any]:
        """
        Generate comprehensive analysis report for an EEG window.
        
        Args:
            eeg_window: EEG data [n_channels, n_samples]
            label: Optional label (0=normal, 1=seizure)
            sfreq: Sampling frequency
            sample_name: Name for this sample
            
        Returns:
            Dictionary with all computed features and paths to saved plots
        """
        print(f"\n{'='*60}")
        print(f"Generating comprehensive report for: {sample_name}")
        print(f"{'='*60}")
        
        saved_plots = {}
        
        # 1. Raw signal plot
        print("  [1/7] Plotting raw signal...")
        fig1 = self.plot_eeg_signal(
            eeg_window, 
            title=f"EEG Signal - {sample_name}",
            save_name=f"{sample_name}_signal",
            sfreq=sfreq
        )
        saved_plots['signal'] = str(self.output_dir / "signals" / f"{sample_name}_signal.png")
        plt.close(fig1)
        
        # 2. Spectrogram
        print("  [2/7] Computing spectrogram...")
        fig2 = self.plot_spectrogram(
            eeg_window,
            channel_idx=0,
            title=f"Spectrogram - {sample_name}",
            save_name=f"{sample_name}_spectrogram",
            sfreq=sfreq
        )
        saved_plots['spectrogram'] = str(self.output_dir / "spectra" / f"{sample_name}_spectrogram.png")
        plt.close(fig2)
        
        # 3. Band powers
        print("  [3/7] Computing band powers...")
        fig3 = self.plot_band_powers(
            eeg_window,
            title=f"Band Powers - {sample_name}",
            save_name=f"{sample_name}_band_powers",
            sfreq=sfreq
        )
        saved_plots['band_powers'] = str(self.output_dir / "spectra" / f"{sample_name}_band_powers.png")
        plt.close(fig3)
        
        # 4. Channel correlations
        print("  [4/7] Computing channel correlations...")
        fig4 = self.plot_channel_correlation_matrix(
            eeg_window,
            title=f"Channel Correlations - {sample_name}",
            save_name=f"{sample_name}_correlations"
        )
        saved_plots['correlations'] = str(self.output_dir / "metrics" / f"{sample_name}_correlations.png")
        plt.close(fig4)
        
        # 5. Metrics radar chart
        print("  [5/7] Computing all features...")
        all_features = compute_all_advanced_features(eeg_window, sfreq)
        
        fig5 = self.plot_metrics_radar_chart(
            all_features,
            title=f"Metrics Radar - {sample_name}",
            save_name=f"{sample_name}_radar"
        )
        if fig5:
            saved_plots['radar'] = str(self.output_dir / "metrics" / f"{sample_name}_radar.png")
            plt.close(fig5)
        
        # 6. Top metrics bar chart
        print("  [6/7] Plotting top metrics...")
        fig6 = self.plot_metric_comparison(
            all_features,
            title=f"Top Metrics - {sample_name}",
            save_name=f"{sample_name}_metrics",
            top_n=15
        )
        saved_plots['metrics'] = str(self.output_dir / "metrics" / f"{sample_name}_metrics.png")
        plt.close(fig6)
        
        # 7. Save JSON report
        print("  [7/7] Saving JSON report...")
        report = {
            'sample_name': sample_name,
            'timestamp': self.timestamp,
            'label': label if label is not None else 'unknown',
            'eeg_shape': list(eeg_window.shape),
            'sfreq': sfreq,
            'features': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in all_features.items()},
            'saved_plots': saved_plots,
        }
        
        report_path = self.output_dir / "reports" / f"{sample_name}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        saved_plots['report_json'] = str(report_path)
        
        print(f"\n[OK] Report saved to: {self.output_dir / 'reports'}")
        print(f"  Total features computed: {len(all_features)}")
        
        return report
    
    def create_comparison_report(
        self,
        normal_windows: List[np.ndarray],
        seizure_windows: List[np.ndarray],
        sfreq: float = 256,
    ) -> Dict[str, any]:
        """
        Create comparison report between normal and seizure windows.
        """
        print(f"\n{'='*60}")
        print(f"Generating comparison report")
        print(f"{'='*60}")
        
        # Combine labels
        all_windows = normal_windows + seizure_windows
        labels = [0] * len(normal_windows) + [1] * len(seizure_windows)
        
        # 1. Feature distributions
        print("  [1/4] Plotting feature distributions...")
        fig1 = self.plot_feature_distributions(
            all_windows, labels,
            title="Normal vs Seizure Feature Distributions",
            save_name="comparison_distributions"
        )
        saved_plots = {'distributions': str(self.output_dir / "comparisons" / "comparison_distributions.png")}
        plt.close(fig1)
        
        # 2. Compute and compare metrics
        print("  [2/4] Computing statistics...")
        normal_stats = []
        seizure_stats = []
        
        for window in normal_windows:
            normal_stats.append(compute_all_advanced_features(window, sfreq))
        for window in seizure_windows:
            seizure_stats.append(compute_all_advanced_features(window, sfreq))
        
        # 3. Statistical summary
        print("  [3/4] Computing statistical summary...")
        summary = self._compute_statistical_summary(normal_stats, seizure_stats)
        
        # 4. Save comparison report
        print("  [4/4] Saving comparison report...")
        comparison_report = {
            'timestamp': self.timestamp,
            'n_normal': len(normal_windows),
            'n_seizure': len(seizure_windows),
            'statistical_summary': summary,
            'saved_plots': saved_plots,
        }
        
        report_path = self.output_dir / "reports" / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)
        
        saved_plots['report_json'] = str(report_path)
        
        print(f"\n[OK] Comparison report saved")
        
        return comparison_report
    
    def _compute_statistical_summary(
        self,
        normal_stats: List[Dict],
        seizure_stats: List[Dict],
    ) -> Dict:
        """
        Compute statistical summary comparing normal vs seizure.
        """
        summary = {}
        
        # Get all feature names
        all_features = set()
        for stats in normal_stats + seizure_stats:
            all_features.update(stats.keys())
        
        for feature in sorted(all_features):
            normal_values = [s.get(feature, np.nan) for s in normal_stats]
            seizure_values = [s.get(feature, np.nan) for s in seizure_stats]
            
            normal_values = [v for v in normal_values if not np.isnan(v)]
            seizure_values = [v for v in seizure_values if not np.isnan(v)]
            
            if normal_values and seizure_values:
                summary[feature] = {
                    'normal': {
                        'mean': float(np.mean(normal_values)),
                        'std': float(np.std(normal_values)),
                        'min': float(np.min(normal_values)),
                        'max': float(np.max(normal_values)),
                    },
                    'seizure': {
                        'mean': float(np.mean(seizure_values)),
                        'std': float(np.std(seizure_values)),
                        'min': float(np.min(seizure_values)),
                        'max': float(np.max(seizure_values)),
                    },
                    'difference': {
                        'mean_diff': float(np.mean(seizure_values) - np.mean(normal_values)),
                        'ratio': float(np.mean(seizure_values) / (np.mean(normal_values) + 1e-10)),
                    }
                }
        
        return summary
    
    def _save_figure(self, fig: plt.Figure, save_name: str, folder: str):
        """Save figure to output directory."""
        path = self.output_dir / folder / f"{self.timestamp}_{save_name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {path.name}")
    
    def list_outputs(self) -> Dict[str, List[str]]:
        """List all generated output files."""
        outputs = {}
        for subdir in ['signals', 'metrics', 'spectra', 'comparisons', 'reports']:
            dir_path = self.output_dir / subdir
            outputs[subdir] = [f.name for f in dir_path.glob('*') if f.is_file()]
        return outputs


def demo_visualization():
    """Demonstrate visualization capabilities with sample data."""
    import sys
    sys.path.insert(0, 'src')
    
    np.random.seed(42)
    
    print("="*60)
    print("EEG Visualization Demo")
    print("="*60)
    
    # Create sample EEG data
    n_channels, n_samples = 21, 128
    sfreq = 256
    
    # Simulate normal EEG window
    t = np.arange(n_samples) / sfreq
    normal_window = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        # Simulate alpha waves with some noise
        normal_window[i] = (np.sin(2 * np.pi * 10 * t) + 
                           0.5 * np.sin(2 * np.pi * 20 * t) +
                           np.random.randn(n_samples) * 0.3) * 20
    
    # Create visualizer
    viz = EEGVisualizer(output_dir="outputs/demo")
    
    # Generate comprehensive report
    report = viz.create_comprehensive_report(
        eeg_window=normal_window,
        label=0,
        sfreq=sfreq,
        sample_name="normal_sample"
    )
    
    print("\n" + "="*60)
    print("Generated outputs:")
    outputs = viz.list_outputs()
    for folder, files in outputs.items():
        print(f"\n  {folder}/ ({len(files)} files)")
        for f in files[:3]:
            print(f"    - {f}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")
    
    print("\n" + "="*60)
    print("Demo complete! Check outputs/demo folder.")
    print("="*60)
    
    return viz


if __name__ == "__main__":
    demo_visualization()
