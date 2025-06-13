# src/features.py
import numpy as np
import pandas as pd
from scipy import signal
from typing import List, Dict
import antropy

from utils import AppConfig, TrialData

def get_band_power(data: np.ndarray, sfreq: float, band: tuple) -> float:
    win = min(int(4 * sfreq), len(data))
    if win == 0: return 0.0
    freqs, psd = signal.welch(data, sfreq, nperseg=win)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    if not np.any(idx_band): return 0.0
    return np.sum(psd[idx_band])

def compute_features_for_epoch(epoch: np.ndarray, config: AppConfig) -> Dict[str, np.ndarray]:
    sfreq = config.filter.sfreq
    features = {}
    for band_name, band_freqs in config.freq_bands.model_dump().items():
        powers = np.array([get_band_power(epoch[i], sfreq, band_freqs) for i in range(epoch.shape[0])])
        features[f'{band_name}_power'] = powers
    
    features['differential_entropy'] = epoch.var(axis=1)
    features['permutation_entropy'] = np.array([antropy.perm_entropy(epoch[i], normalize=True) for i in range(epoch.shape[0])])
    features['spectral_entropy'] = np.array([antropy.spectral_entropy(epoch[i], sf=sfreq, method='welch', normalize=True) for i in range(epoch.shape[0])])
    return features

def extract_all_features(processed_trials: List[TrialData], config: AppConfig) -> pd.DataFrame:
    feature_list = []
    for trial in processed_trials:
        if not trial.is_valid or trial.clean_baseline_data is None or trial.clean_stim_data is None: continue
        
        baseline_feats = compute_features_for_epoch(trial.clean_baseline_data, config)
        stim_feats = compute_features_for_epoch(trial.clean_stim_data, config)
        
        trial_features = {'subject_id': trial.subject_id, 'trial_id': trial.trial_id, 'preference': trial.preference.value}
        
        for feat_name in baseline_feats:
            base_val = baseline_feats[feat_name] + 1e-12
            stim_val = stim_feats[feat_name]
            ratio = (stim_val - base_val) / base_val
            if len(ratio) == 2:
                trial_features[f'FP1_{feat_name}_ratio'] = ratio[0]
                trial_features[f'FP2_{feat_name}_ratio'] = ratio[1]
                trial_features[f'{feat_name}_asymmetry'] = ratio[1] - ratio[0]
        feature_list.append(trial_features)

    if not feature_list: return pd.DataFrame()
    df = pd.DataFrame(feature_list)
    df['dummy_valence'] = np.random.uniform(1, 9, len(df))
    return df
