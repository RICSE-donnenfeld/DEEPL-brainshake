"""
Features module
---
Implements different metrics computation functions from a single EEG signal input.

Features computed:
- Basic statistics: mean, std, min, max, median, range
- Higher-order statistics: kurtosis, skewness
- Information-theoretic: entropy, sample entropy, permutation entropy
- Frequency-domain: power, energy, spectral features, band powers
- Hjorth parameters: activity, mobility, complexity
- Nonlinear dynamics: Hurst exponent, fractal dimension, Lyapunov
- Seizure-specific: spike rate, line length, turning points

EEG Signal Dimensions:
- Input shape: [21 channels, 128 samples] per window
- Features can be computed per-channel or aggregated across channels
"""

import numpy as np
import math
from scipy import stats
from scipy.signal import welch, hilbert, find_peaks
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Union, Tuple, Optional
import warnings


# =============================================================================
# HJORTH PARAMETERS (Classic EEG metrics)
# =============================================================================

def compute_hjorth_activity(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Hjorth Activity - variance of the signal.
    
    Represents the surface under the power spectrum.
    Higher during seizure activity due to increased amplitude.
    """
    return compute_variance(eeg_signal)


def compute_hjorth_mobility(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Hjorth Mobility - normalized first derivative.
    
    Mobility = sqrt(var(dx/dt) / var(x))
    Measures average frequency of the signal.
    Can indicate frequency changes during seizures.
    """
    if eeg_signal.ndim == 1:
        dx = np.diff(eeg_signal)
        return np.sqrt(compute_variance(dx) / (compute_variance(eeg_signal) + 1e-10))
    else:
        results = []
        for ch in eeg_signal:
            dx = np.diff(ch)
            results.append(np.sqrt(compute_variance(dx) / (compute_variance(ch) + 1e-10)))
        return np.array(results)


def compute_hjorth_complexity(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Hjorth Complexity - mobility of the derivative.
    
    Complexity = Mobility(dx/dt) / Mobility(x)
    Approaches 1 for signals consisting of many frequencies.
    Can increase during seizure activity.
    """
    if eeg_signal.ndim == 1:
        dx = np.diff(eeg_signal)
        mob_x = np.sqrt(compute_variance(dx) / (compute_variance(eeg_signal) + 1e-10))
        d2x = np.diff(dx)
        mob_dx = np.sqrt(compute_variance(d2x) / (compute_variance(dx) + 1e-10))
        return mob_dx / (mob_x + 1e-10)
    else:
        results = []
        for ch in eeg_signal:
            dx = np.diff(ch)
            mob_x = np.sqrt(compute_variance(dx) / (compute_variance(ch) + 1e-10))
            d2x = np.diff(dx)
            mob_dx = np.sqrt(compute_variance(d2x) / (compute_variance(dx) + 1e-10))
            results.append(mob_dx / (mob_x + 1e-10))
        return np.array(results)


def compute_hjorth_parameters(eeg_signal: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute all Hjorth parameters.
    
    Returns:
        Dict with 'activity', 'mobility', 'complexity'
    """
    return {
        'activity': compute_hjorth_activity(eeg_signal),
        'mobility': compute_hjorth_mobility(eeg_signal),
        'complexity': compute_hjorth_complexity(eeg_signal),
    }


# =============================================================================
# NONLINEAR DYNAMICS METRICS
# =============================================================================

def compute_sample_entropy(eeg_signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Sample Entropy (SampEn) - measures signal complexity.
    
    Lower values indicate more self-similarity (regularity).
    Seizures typically show LOWER sample entropy (more regular).
    
    Args:
        eeg_signal: 1D signal
        m: Embedding dimension
        r: Tolerance (usually 0.1-0.2 times std)
    """
    N = len(eeg_signal)
    if N < m + 1:
        return np.nan
    
    r = r * np.std(eeg_signal)
    
    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi(m):
        patterns = np.array([eeg_signal[i:i + m] for i in range(N - m)])
        count = 0
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i != j and _maxdist(patterns[i], patterns[j]) < r:
                    count += 1
        return count / (N - m)
    
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    if phi_m == 0 or phi_m1 == 0:
        return np.inf
    
    return -np.log(phi_m1 / phi_m)


def compute_approximate_entropy(eeg_signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Approximate Entropy (ApEn) - measures regularity.
    
    Similar to sample entropy but includes self-matches.
    Lower ApEn = more regular signal.
    """
    N = len(eeg_signal)
    r = r * np.std(eeg_signal)
    
    def _phi(m):
        patterns = np.array([eeg_signal[i:i + m] for i in range(N - m + 1)])
        C = np.zeros(N - m + 1)
        for i in range(N - m + 1):
            for j in range(N - m + 1):
                if max(abs(patterns[j] - patterns[i])) < r:
                    C[i] += 1
        C /= (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1)
    
    return _phi(m) - _phi(m + 1)


def compute_permutation_entropy(eeg_signal: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Permutation Entropy - measures complexity via ordinal patterns.
    
    Fast to compute, robust to noise.
    Lower values indicate more deterministic/regular signals.
    
    Args:
        eeg_signal: 1D signal
        order: Pattern length (3-7 typical)
        delay: Time delay between elements
    """
    n = len(eeg_signal)
    if n < order * delay:
        return np.nan
    
    # Create ordinal patterns
    patterns = []
    for i in range(0, n - order * delay + 1, delay):
        pattern = tuple(np.argsort(eeg_signal[i:i + order * delay:delay]))
        patterns.append(pattern)
    
    # Count pattern frequencies
    from collections import Counter
    counts = Counter(patterns)
    total = len(patterns)
    
    # Shannon entropy
    entropy = 0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log(p)
    
    # Normalize
    max_entropy = np.log(math.factorial(order))
    return entropy / max_entropy if max_entropy > 0 else 0


def compute_hurst_exponent(eeg_signal: np.ndarray) -> float:
    """
    Hurst Exponent (H) - measures long-term memory.
    
    H < 0.5: anti-persistent (mean-reverting)
    H = 0.5: random walk (white noise)
    H > 0.5: persistent (trending)
    
    EEG typically has H > 0.5 (persistent).
    Changes in H can indicate seizure onset.
    """
    lags = range(2, min(20, len(eeg_signal) // 2))
    
    tau = []
    for lag in lags:
        diffs = eeg_signal[lag:] - eeg_signal[:-lag]
        tau.append(np.std(diffs))
    
    tau = np.array(tau)
    lags = np.array(list(lags))
    
    if np.any(tau == 0) or len(tau) < 3:
        return 0.5
    
    # Linear fit in log-log space
    log_lags = np.log(lags)
    log_tau = np.log(tau)
    
    slope, _ = np.polyfit(log_lags, log_tau, 1)
    return slope / 2 + 0.5


def compute_lyapunov_exponent(eeg_signal: np.ndarray, delay: int = 1, dim: int = 3) -> float:
    """
    Lyapunov Exponent - measures chaos/divergence.
    
    Positive LE indicates chaotic system (sensitive to initial conditions).
    Seizures may show increased Lyapunov exponent.
    
    This is a simplified estimation using Wolf method approximation.
    """
    n = len(eeg_signal) - (dim - 1) * delay
    if n < 100:
        return np.nan
    
    # Build trajectory matrix
    trajectory = np.zeros((n, dim))
    for i in range(dim):
        trajectory[:, i] = eeg_signal[i * delay:i * delay + n]
    
    # Simple approximation: mean divergence of nearby points
    divergences = []
    for i in range(10, n - 10):
        nearby = np.where(np.abs(np.arange(n) - i) <= 5)[0]
        nearby = nearby[nearby != i]
        if len(nearby) > 0:
            for j in nearby[:3]:
                div = np.log(np.abs(eeg_signal[i + 1] - eeg_signal[j + 1]) + 1e-10)
                divergences.append(div)
    
    if divergences:
        return np.mean(divergences)
    return 0.0


def compute_fractal_dimension(eeg_signal: np.ndarray, method: str = 'petrosian') -> float:
    """
    Fractal Dimension - measures signal complexity.
    
    Higher FD = more complex signal.
    Seizures often show changes in fractal dimension.
    
    Methods:
    - 'petrosian': Fast, uses zero-crossings
    - 'katz': Uses waveform length
    """
    if method == 'petrosian':
        n = len(eeg_signal)
        nc = compute_zero_crossings(eeg_signal)[0] if eeg_signal.ndim == 1 else np.mean(compute_zero_crossings(eeg_signal))
        return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * nc) + 1e-10))
    
    elif method == 'katz':
        n = len(eeg_signal)
        if n < 2:
            return 0.0
        
        # Compute distances
        d = np.abs(eeg_signal - eeg_signal[0])
        L = np.sum(np.abs(np.diff(eeg_signal)))
        
        if L == 0:
            return 0.0
        
        k = np.log10(d / L + 1e-10)
        return np.log10(n - 1) / (np.log10(d / L + 1e-10) + np.log10(n - 1))
    
    return 0.0


# =============================================================================
# SEIZURE-SPECIFIC METRICS
# =============================================================================

def compute_line_length(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Line Length - sum of absolute differences between consecutive samples.
    
    Robust to outliers, less sensitive to artifacts.
    Seizure periods show increased line length.
    """
    if eeg_signal.ndim == 1:
        return np.sum(np.abs(np.diff(eeg_signal)))
    else:
        return np.array([np.sum(np.abs(np.diff(ch))) for ch in eeg_signal])


def compute_turning_points_ratio(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Turning Points Ratio - proportion of points that are local extrema.
    
    Higher ratio = more oscillatory signal.
    Seizures show characteristic oscillatory patterns.
    """
    if eeg_signal.ndim == 1:
        if len(eeg_signal) < 3:
            return 0.0
        diff1 = np.diff(eeg_signal)
        diff2 = np.diff(np.sign(diff1))
        return np.sum(np.abs(diff2) > 0) / (len(eeg_signal) - 2)
    else:
        return np.array([compute_turning_points_ratio(ch) for ch in eeg_signal])


def compute_spike_count(eeg_signal: np.ndarray, threshold_sigma: float = 3.0) -> int:
    """
    Count spikes exceeding threshold (simplified seizure spike detection).
    
    Args:
        eeg_signal: 1D signal
        threshold_sigma: Number of standard deviations for threshold
    """
    threshold = threshold_sigma * np.std(eeg_signal)
    mean_val = np.mean(eeg_signal)
    
    # Find peaks above threshold
    peaks, _ = find_peaks(np.abs(eeg_signal - mean_val), height=threshold)
    return len(peaks)


def compute_burst_suppression_ratio(eeg_signal: np.ndarray, threshold: float = 20.0) -> float:
    """
    Burst Suppression Ratio - proportion of time in burst state.
    
    Used in seizure detection and anesthesia monitoring.
    Burst: high amplitude activity
    Suppression: low amplitude, flat
    """
    abs_signal = np.abs(eeg_signal)
    suppressed = np.sum(abs_signal < threshold) / len(eeg_signal)
    return 1.0 - suppressed


def compute_waveform_complexity(eeg_signal: np.ndarray) -> float:
    """
    Waveform Complexity - based on curve length and amplitude.
    
    Combines multiple measures into a single complexity metric.
    """
    n = len(eeg_signal)
    
    # Curve length
    curve_length = np.sum(np.abs(np.diff(eeg_signal)))
    
    # Normalize by amplitude range
    amplitude_range = np.max(eeg_signal) - np.min(eeg_signal) + 1e-10
    
    # Normalize by duration
    complexity = curve_length / (n * amplitude_range)
    
    return complexity


def compute_spectral_edge_frequency(eeg_signal: np.ndarray, sfreq: float = 256, 
                                   percentile: float = 95) -> float:
    """
    Spectral Edge Frequency (SEF) - frequency below which X% of power exists.
    
    SEF95 is commonly used in anesthesia monitoring.
    Shifts to lower frequencies during seizures.
    """
    freqs, psd = welch(eeg_signal, fs=sfreq, nperseg=min(256, len(eeg_signal)))
    
    cumsum = np.cumsum(psd)
    total_power = cumsum[-1]
    
    target_power = percentile / 100 * total_power
    idx = np.where(cumsum >= target_power)[0]
    
    if len(idx) > 0:
        return freqs[idx[0]]
    return freqs[-1]


def compute_spectral_centroid(eeg_signal: np.ndarray, sfreq: float = 256) -> float:
    """
    Spectral Centroid - center of mass of the spectrum.
    
    Measures the "brightness" of the sound.
    Higher centroid = more high-frequency content.
    """
    freqs, psd = welch(eeg_signal, fs=sfreq, nperseg=min(256, len(eeg_signal)))
    
    centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
    return centroid


def compute_spectral_rolloff(eeg_signal: np.ndarray, sfreq: float = 256,
                             threshold: float = 0.85) -> float:
    """
    Spectral Rolloff - frequency below which X% of energy is contained.
    """
    freqs, psd = welch(eeg_signal, fs=sfreq, nperseg=min(256, len(eeg_signal)))
    
    cumsum = np.cumsum(psd)
    total_energy = cumsum[-1]
    
    target_energy = threshold * total_energy
    idx = np.where(cumsum >= target_energy)[0]
    
    if len(idx) > 0:
        return freqs[idx[0]]
    return freqs[-1]


def compute_band_power_ratios(eeg_signal: np.ndarray, sfreq: float = 256) -> Dict[str, float]:
    """
    Compute band power ratios - useful seizure biomarkers.
    
    Common ratios:
    - Theta/Beta: Increases during seizures
    - Delta/Alpha: Shifts during seizures
    - (Delta+Theta)/(Alpha+Beta): Slow/fast ratio
    """
    band_powers = compute_band_powers(eeg_signal, sfreq)
    
    # Handle array results
    def safe_mean(arr):
        if isinstance(arr, np.ndarray):
            return np.mean(arr)
        return arr
    
    delta = safe_mean(band_powers.get('delta', 0))
    theta = safe_mean(band_powers.get('theta', 0))
    alpha = safe_mean(band_powers.get('alpha', 0))
    beta = safe_mean(band_powers.get('beta', 0))
    gamma = safe_mean(band_powers.get('gamma', 0))
    
    return {
        'theta_beta_ratio': theta / (beta + 1e-10),
        'delta_alpha_ratio': delta / (alpha + 1e-10),
        'slow_fast_ratio': (delta + theta) / (alpha + beta + 1e-10),
        'theta_alpha_ratio': theta / (alpha + 1e-10),
        'delta_beta_ratio': delta / (beta + 1e-10),
        'seizure_index': (delta + theta) / (alpha + beta + gamma + 1e-10),
    }


# =============================================================================
# CONNECTIVITY METRICS (between channels)
# =============================================================================

def compute_channel_correlation(eeg_window: np.ndarray) -> np.ndarray:
    """
    Compute pairwise correlation between all channels.
    
    Returns:
        Correlation matrix of shape [n_channels, n_channels]
    """
    n_channels = eeg_window.shape[0]
    corr_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            corr, _ = pearsonr(eeg_window[i], eeg_window[j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    return corr_matrix


def compute_coherence(eeg_window: np.ndarray, sfreq: float = 256) -> np.ndarray:
    """
    Compute coherence between channel pairs.
    
    Coherence measures frequency-domain correlation.
    """
    n_channels = eeg_window.shape[0]
    n_freqs = 64
    
    coherence_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            # Compute coherence using Welch's method
            f, Cxy = plt_coherence(eeg_window[i], eeg_window[j], fs=sfreq, nperseg=64)
            avg_coherence = np.mean(Cxy[:n_freqs])
            coherence_matrix[i, j] = avg_coherence
            coherence_matrix[j, i] = avg_coherence
    
    return coherence_matrix


def plt_coherence(x, y, fs=1.0, nperseg=256):
    """Compute coherence function (simplified version)."""
    from scipy.signal import csd
    
    freqs, Pxx = welch(x, fs=fs, nperseg=nperseg)
    _, Pyy = welch(y, fs=fs, nperseg=nperseg)
    _, Pxy = csd(x, y, fs=fs, nperseg=nperseg)
    
    coherence = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-10)
    
    return freqs, coherence


def compute_mean_coherence(eeg_window: np.ndarray, sfreq: float = 256) -> float:
    """
    Compute mean coherence across all channel pairs.
    """
    corr_matrix = compute_channel_correlation(eeg_window)
    n = corr_matrix.shape[0]
    
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    return np.mean(corr_matrix[mask])


# =============================================================================
# AMPLITUDE AND INSTANTANEOUS FEATURES
# =============================================================================

def compute_envelope_statistics(eeg_signal: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics of the signal envelope (via Hilbert transform).
    """
    analytic_signal = hilbert(eeg_signal)
    envelope = np.abs(analytic_signal)
    
    return {
        'envelope_mean': np.mean(envelope),
        'envelope_std': np.std(envelope),
        'envelope_max': np.max(envelope),
        'envelope_var': np.var(envelope),
    }


def compute_instantaneous_phase(eeg_signal: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous phase via Hilbert transform.
    """
    analytic_signal = hilbert(eeg_signal)
    phase = np.angle(analytic_signal)
    return phase


def compute_phase_locking_value(eeg_window: np.ndarray) -> float:
    """
    Phase Locking Value (PLV) - measures phase synchronization.
    
    PLV = |mean(exp(i * (phi_i - phi_j)))|
    Values close to 1 indicate strong phase synchronization.
    """
    n_channels = eeg_window.shape[0]
    
    if n_channels < 2:
        return 1.0
    
    # Compute phases for all channels
    phases = np.array([compute_instantaneous_phase(ch) for ch in eeg_window])
    
    # Compute PLV across channel pairs
    plvs = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phases[i] - phases[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plvs.append(plv)
    
    return np.mean(plvs)


# =============================================================================
# COMPREHENSIVE FEATURE EXTRACTION
# =============================================================================

def compute_all_advanced_features(eeg_window: np.ndarray, sfreq: float = 256) -> Dict[str, float]:
    """
    Compute comprehensive set of EEG features.
    
    Returns dictionary with all features.
    """
    features = {}
    
    # Basic statistics
    features['mean'] = float(np.mean(eeg_window))
    features['std'] = float(np.std(eeg_window))
    features['min'] = float(np.min(eeg_window))
    features['max'] = float(np.max(eeg_window))
    features['range'] = features['max'] - features['min']
    
    # Hjorth parameters (average across channels)
    features['hjorth_activity'] = float(np.mean(compute_hjorth_activity(eeg_window)))
    features['hjorth_mobility'] = float(np.mean(compute_hjorth_mobility(eeg_window)))
    features['hjorth_complexity'] = float(np.mean(compute_hjorth_complexity(eeg_window)))
    
    # Higher order statistics
    features['skewness'] = float(stats.skew(eeg_window.flatten()))
    features['kurtosis'] = float(stats.kurtosis(eeg_window.flatten()))
    
    # Spectral features
    mean_signal = np.mean(eeg_window, axis=0)
    features['spectral_entropy'] = float(np.mean(compute_spectral_entropy(eeg_window, sfreq)))
    features['spectral_centroid'] = float(compute_spectral_centroid(mean_signal, sfreq))
    features['spectral_rolloff'] = float(compute_spectral_rolloff(mean_signal, sfreq))
    features['sef95'] = float(compute_spectral_edge_frequency(mean_signal, sfreq, 95))
    
    # Band powers and ratios
    band_powers = compute_band_powers(eeg_window, sfreq)
    for band, power in band_powers.items():
        features[f'{band}_power'] = float(np.mean(power))
    
    band_ratios = compute_band_power_ratios(eeg_window, sfreq)
    features.update({f'{k}_ratio': v for k, v in band_ratios.items()})
    
    # Nonlinear dynamics
    features['line_length'] = float(np.mean(compute_line_length(eeg_window)))
    features['turning_points'] = float(np.mean(compute_turning_points_ratio(eeg_window)))
    features['waveform_complexity'] = float(compute_waveform_complexity(mean_signal))
    
    # Connectivity
    features['mean_correlation'] = float(compute_mean_coherence(eeg_window, sfreq))
    
    # Envelope features
    env_stats = compute_envelope_statistics(mean_signal)
    features.update(env_stats)
    
    # Phase locking
    features['phase_locking_value'] = float(compute_phase_locking_value(eeg_window))
    
    return features


# =============================================================================
# HELPER FUNCTIONS FOR COMPATIBILITY
# =============================================================================

def compute_mean(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute mean of EEG signal.
    
    Args:
        eeg_signal: EEG data of shape [n_channels, n_samples] or [n_samples]
        
    Returns:
        Mean value(s). If 2D input, returns array of means per channel.
    """
    return np.mean(eeg_signal, axis=-1)


def compute_std(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute standard deviation of EEG signal.
    
    Measures signal variability around the mean.
    Higher std indicates more variable/bursty neural activity.
    """
    return np.std(eeg_signal, axis=-1)


def compute_min(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """Minimum value in the signal."""
    return np.min(eeg_signal, axis=-1)


def compute_max(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """Maximum value in the signal."""
    return np.max(eeg_signal, axis=-1)


def compute_median(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Median of the signal.
    
    More robust to outliers than mean - good for EEG which can have
    artifacts (eye blinks, muscle movements) causing extreme values.
    """
    return np.median(eeg_signal, axis=-1)


def compute_range(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """Signal range (max - min)."""
    return compute_max(eeg_signal) - compute_min(eeg_signal)


def compute_variance(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """Variance of the signal."""
    return np.var(eeg_signal, axis=-1)


def compute_skewness(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Skewness of the signal distribution.
    
    Measures asymmetry:
    - skewness > 0: right-skewed (long tail to right) - typical for EEG
    - skewness = 0: symmetric distribution
    - skewness < 0: left-skewed
    
    EEG signals often show positive skewness due to occasional
    high-amplitude spikes during seizures.
    """
    return stats.skew(eeg_signal, axis=-1)


def compute_kurtosis(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Kurtosis (excess kurtosis) of the signal distribution.
    
    Measures "tailedness" - how much probability mass is in tails:
    - kurtosis > 0: heavy tails (more outliers) - characteristic of seizures
    - kurtosis = 0: normal distribution tails
    - kurtosis < 0: light tails (fewer outliers)
    
    Seizure periods often show high kurtosis due to sharp spikes.
    """
    return stats.kurtosis(eeg_signal, axis=-1)


def compute_entropy(eeg_signal: np.ndarray, n_bins: int = 32) -> Union[float, np.ndarray]:
    """
    Shannon entropy of the signal.
    
    Measures signal complexity/information content:
    - High entropy: complex, unpredictable signal
    - Low entropy: simple, predictable signal
    
    Entropy typically decreases during seizures as activity
    becomes more regular/periodic.
    
    Args:
        eeg_signal: EEG data
        n_bins: Number of bins for histogram estimation
    """
    def entropy_1d(signal):
        hist, _ = np.histogram(signal, bins=n_bins, density=True)
        hist = hist[hist > 0]  # Remove zeros to avoid log(0)
        return -np.sum(hist * np.log(hist + 1e-10))
    
    if eeg_signal.ndim == 1:
        return entropy_1d(eeg_signal)
    else:
        return np.array([entropy_1d(ch) for ch in eeg_signal])


def compute_rms(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Root Mean Square (RMS) of the signal.
    
    RMS = sqrt(mean(signal²))
    
    Measures signal magnitude - useful for comparing overall
    signal power. Often used in seizure detection as seizure
    periods have higher RMS.
    """
    return np.sqrt(np.mean(eeg_signal ** 2, axis=-1))


def compute_energy(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Total energy of the signal.
    
    Energy = sum(signal²)
    
    Seizure activity typically shows increased energy.
    """
    return np.sum(eeg_signal ** 2, axis=-1)


def compute_mean_absolute_value(eeg_signal: np.ndarray) -> Union[float, np.ndarray]:
    """
    Mean Absolute Value (MAV).
    
    MAV = mean(|signal|)
    
    Similar to RMS but less sensitive to large spikes.
    Commonly used in EMG/EEG signal processing.
    """
    return np.mean(np.abs(eeg_signal), axis=-1)


def compute_zero_crossings(eeg_signal: np.ndarray, threshold: float = 0.0) -> Union[int, np.ndarray]:
    """
    Number of times the signal crosses zero.
    
    Higher zero-crossing rate indicates more high-frequency
    activity. Can distinguish seizure (high ZCR) from normal (low ZCR).
    
    Args:
        eeg_signal: EEG data
        threshold: Small threshold to avoid noise-induced crossings
    """
    if eeg_signal.ndim == 1:
        return np.sum(np.abs(np.diff(np.sign(eeg_signal - threshold))) > 0)
    else:
        return np.array([
            np.sum(np.abs(np.diff(np.sign(ch - threshold))) > 0)
            for ch in eeg_signal
        ])


def compute_spectral_entropy(eeg_signal: np.ndarray, sfreq: float = 256) -> Union[float, np.ndarray]:
    """
    Spectral entropy in frequency domain.
    
    Measures frequency content complexity. Lower spectral entropy
    indicates more concentrated power in few frequencies (typical of seizures).
    
    Args:
        eeg_signal: EEG data
        sfreq: Sampling frequency (default 256 Hz for CHB-MIT)
    """
    def spectral_entropy_1d(signal):
        # Compute power spectral density
        freqs, psd = welch(signal, fs=sfreq, nperseg=min(256, len(signal)))
        
        # Normalize PSD to get probability distribution
        psd_norm = psd / (np.sum(psd) + 1e-10)
        
        # Shannon entropy of the power distribution
        return -np.sum(psd_norm * np.log(psd_norm + 1e-10))
    
    if eeg_signal.ndim == 1:
        return spectral_entropy_1d(eeg_signal)
    else:
        return np.array([spectral_entropy_1d(ch) for ch in eeg_signal])


def compute_band_powers(eeg_signal: np.ndarray, sfreq: float = 256) -> Dict[str, np.ndarray]:
    """
    Compute power in standard EEG frequency bands.
    
    Bands:
    - Delta: 0.5-4 Hz (deep sleep, slow waves)
    - Theta: 4-8 Hz (drowsiness, meditation)
    - Alpha: 8-13 Hz (relaxation, eyes closed)
    - Beta: 13-30 Hz (active thinking, alertness)
    - Gamma: 30-100 Hz (cognitive processing)
    
    Seizures often show increased delta/theta and decreased alpha.
    
    Args:
        eeg_signal: EEG data of shape [n_channels, n_samples]
        sfreq: Sampling frequency
        
    Returns:
        Dict with band names as keys and power arrays as values
    """
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100),
    }
    
    result = {}
    
    if eeg_signal.ndim == 2:
        n_channels = eeg_signal.shape[0]
    else:
        n_channels = 1
        eeg_signal = eeg_signal.reshape(1, -1)
    
    for band_name, (low, high) in bands.items():
        powers = []
        for ch in eeg_signal:
            freqs, psd = welch(ch, fs=sfreq, nperseg=min(256, len(ch)))
            idx = np.logical_and(freqs >= low, freqs <= high)
            powers.append(np.sum(psd[idx]))
        result[band_name] = np.array(powers)
    
    return result


def compute_all_statistical_features(eeg_signal: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute all basic statistical features for a 1D signal.
    
    Args:
        eeg_signal: 1D EEG signal array
        
    Returns:
        Dictionary of feature_name: value
    """
    return {
        'mean': compute_mean(eeg_signal),
        'std': compute_std(eeg_signal),
        'min': compute_min(eeg_signal),
        'max': compute_max(eeg_signal),
        'median': compute_median(eeg_signal),
        'range': compute_range(eeg_signal),
        'variance': compute_variance(eeg_signal),
        'skewness': compute_skewness(eeg_signal),
        'kurtosis': compute_kurtosis(eeg_signal),
        'rms': compute_rms(eeg_signal),
        'energy': compute_energy(eeg_signal),
        'mav': compute_mean_absolute_value(eeg_signal),
        'zero_crossings': compute_zero_crossings(eeg_signal),
        'entropy': compute_entropy(eeg_signal),
    }


def compute_channel_features(eeg_window: np.ndarray) -> np.ndarray:
    """
    Compute features for each channel in an EEG window.
    
    Args:
        eeg_window: EEG data of shape [n_channels=21, n_samples=128]
        
    Returns:
        Feature array of shape [n_channels, n_features]
    """
    n_channels = eeg_window.shape[0]
    features_list = []
    
    for ch in range(n_channels):
        ch_features = [
            compute_mean(eeg_window[ch]),
            compute_std(eeg_window[ch]),
            compute_min(eeg_window[ch]),
            compute_max(eeg_window[ch]),
            compute_median(eeg_window[ch]),
            compute_skewness(eeg_window[ch]),
            compute_kurtosis(eeg_window[ch]),
            compute_rms(eeg_window[ch]),
            compute_energy(eeg_window[ch]),
            compute_entropy(eeg_window[ch]),
            compute_zero_crossings(eeg_window[ch]),
        ]
        features_list.append(ch_features)
    
    return np.array(features_list)


def compute_window_features(eeg_window: np.ndarray) -> np.ndarray:
    """
    Compute aggregated features for entire EEG window.
    
    Computes statistics across all channels, then aggregates
    with mean, std, min, max across channels.
    
    Args:
        eeg_window: EEG data of shape [n_channels=21, n_samples=128]
        
    Returns:
        Feature vector of fixed size (e.g., 44 features)
    """
    ch_features = compute_channel_features(eeg_window)
    
    features = []
    
    for feat_idx in range(ch_features.shape[1]):
        ch_vals = ch_features[:, feat_idx]
        features.extend([
            np.mean(ch_vals),
            np.std(ch_vals),
            np.min(ch_vals),
            np.max(ch_vals),
        ])
    
    return np.array(features)


def extract_features(eeg_input: np.ndarray, mode: str = 'window') -> np.ndarray:
    """
    Main feature extraction function.
    
    Args:
        eeg_input: EEG data
            - mode='channel': shape [n_channels, n_samples]
            - mode='window': shape [n_channels, n_samples]
        mode: Extraction mode - 'channel' or 'window'
        
    Returns:
        Feature array
    """
    if mode == 'window':
        return compute_window_features(eeg_input)
    elif mode == 'channel':
        return compute_channel_features(eeg_input)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'channel' or 'window'.")


class EEGFeatureExtractor:
    """
    Feature extractor class for EEG signals.
    
    Provides a callable interface for feature extraction with
    optional caching and batch processing.
    """
    
    def __init__(self, mode: str = 'window', device: str = 'cpu'):
        """
        Args:
            mode: 'channel' or 'window' feature extraction
            device: 'cpu' or 'cuda' for torch conversion
        """
        self.mode = mode
        self.device = device
    
    def __call__(self, eeg_input: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract features from EEG input."""
        if isinstance(eeg_input, torch.Tensor):
            eeg_input = eeg_input.numpy()
        
        features = extract_features(eeg_input, mode=self.mode)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_batch(self, eeg_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a batch of EEG windows.
        
        Args:
            eeg_batch: Tensor of shape [batch, n_channels, n_samples]
            
        Returns:
            Feature tensor of shape [batch, n_features]
        """
        batch_size = eeg_batch.shape[0]
        features = []
        
        for i in range(batch_size):
            feat = self(eeg_batch[i].numpy())
            features.append(feat)
        
        return torch.stack(features)


def test_features():
    """Test feature extraction on sample data."""
    np.random.seed(42)
    
    eeg_window = np.random.randn(21, 128).astype(np.float32)
    
    print("Testing feature extraction...")
    print(f"Input shape: {eeg_window.shape}")
    
    window_features = compute_window_features(eeg_window)
    print(f"Window features shape: {window_features.shape}")
    
    ch_features = compute_channel_features(eeg_window)
    print(f"Channel features shape: {ch_features.shape}")
    
    band_powers = compute_band_powers(eeg_window)
    print(f"Band powers: {list(band_powers.keys())}")
    
    extractor = EEGFeatureExtractor(mode='window')
    torch_features = extractor(torch.tensor(eeg_window))
    print(f"PyTorch output shape: {torch_features.shape}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_features()
