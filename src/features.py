# src/features.py
import numpy as np
import pandas as pd
from scipy import signa
from typing import List, Dict
import antropy

from utils import AppConfig, TrialData

def get_band_power(data: np.ndarray, sfreq: float, band: tuple) -> float:
    """単一チャネルのバンドパワーを計算"""
    win = int(4 * sfreq) # Welch法の窓長
    freqs, psd = scipy.signal.welch(data, sfreq, nperseg=win)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd[idx_band])

def compute_features_for_epoch(epoch: np.ndarray, config: AppConfig) -> Dict[str, np.ndarray]:
    """1エポック(FP1, FP2)分の全特徴量を計算"""
    sfreq = config.filter.sfreq
    features = {}

    # パワー
    for band_name, band_freqs in config.freq_bands:
        powers = np.array([get_band_power(epoch[i], sfreq, band_freqs) for i in range(2)])
        features[f'{band_name}_power'] = powers

    # エントロピー
    features['differential_entropy'] = antropy.detrend(epoch).var(axis=1) # DEは分散と等価
    features['permutation_entropy'] = np.array([antropy.perm_entropy(epoch[i], normalize=True) for i in range(2)])
    features['spectral_entropy'] = np.array([antropy.spectral_entropy(epoch[i], sf=sfreq, normalize=True) for i in range(2)])
    
    return features

def extract_all_features(processed_trials: List[TrialData], config: AppConfig) -> pd.DataFrame:
    """全有効試行から特徴量を抽出し、正規化してDataFrameを作成"""
    feature_list = []

    for trial in processed_trials:
        if not trial.is_valid: continue

        baseline_feats = compute_features_for_epoch(trial.clean_baseline_data, config)
        stim_feats = compute_features_for_epoch(trial.clean_stim_data, config)
        
        trial_features = {
            'subject_id': trial.subject_id,
            'trial_id': trial.trial_id,
            'preference': trial.preference.value
        }

        # ベースライン比と非対称性を計算
        for feat_name in baseline_feats:
            # ゼロ除算を避ける
            base_val = baseline_feats[feat_name] + 1e-9
            stim_val = stim_feats[feat_name]
            
            # ベースライン比
            ratio = (stim_val - base_val) / base_val
            trial_features[f'FP1_{feat_name}_ratio'] = ratio[0]
            trial_features[f'FP2_{feat_name}_ratio'] = ratio[1]
            
            # 非対称性 (FP2 - FP1)
            trial_features[f'{feat_name}_asymmetry'] = ratio[1] - ratio[0]
        
        feature_list.append(trial_features)

    # ダミーの連続値を追加
    df = pd.DataFrame(feature_list)
    if not df.empty:
      df['dummy_valence'] = np.random.uniform(1, 9, len(df))
    
    return df
