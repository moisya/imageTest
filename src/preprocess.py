# src/preprocess.py
import mne
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

from utils import AppConfig, TrialData, QCResult, WindowConfig

def filter_data(data: Optional[np.ndarray], config: AppConfig) -> Optional[np.ndarray]:
    if data is None or data.shape[1] < 10: return data
    return mne.filter.filter_data(data, sfreq=config.filter.sfreq, l_freq=config.filter.l_freq, h_freq=config.filter.h_freq, fir_design='firwin', verbose=False)

def notch_filter_data(data: Optional[np.ndarray], config: AppConfig) -> Optional[np.ndarray]:
    if data is None or data.shape[1] < 10: return data
    return mne.filter.notch_filter(data, Fs=config.filter.sfreq, freqs=config.filter.notch_freq, fir_design='firwin', verbose=False)

def check_window_quality(window: Optional[np.ndarray], config: AppConfig) -> bool:
    if window is None: return False
    if np.any(np.abs(window) > config.qc.amp_uV): return False
    if np.any(np.abs(np.diff(window, axis=1)) > config.qc.diff_uV): return False
    return True

def get_clean_windows(data: Optional[np.ndarray], config: AppConfig, num_samples_to_take: int) -> Tuple[Optional[np.ndarray], QCResult]:
    win_len_samples = int(config.win.win_len * config.filter.sfreq)
    if data is None or data.shape[1] < win_len_samples:
        return None, QCResult(is_valid=False, n_clean_windows=0, total_windows=0, quality_score=0.0)
    
    total_windows = data.shape[1] // win_len_samples
    clean_windows = [data[:, i*win_len_samples:(i+1)*win_len_samples] for i in range(total_windows) if check_window_quality(data[:, i*win_len_samples:(i+1)*win_len_samples], config)]
    
    n_clean = len(clean_windows)
    quality_score = n_clean / total_windows if total_windows > 0 else 0.0
    is_valid = n_clean >= num_samples_to_take
    
    avg_clean_data = None
    if is_valid:
        np.random.seed(42)
        indices = np.random.choice(n_clean, num_samples_to_take, replace=False)
        avg_clean_data = np.mean([clean_windows[i] for i in indices], axis=0)
    
    return avg_clean_data, QCResult(is_valid=is_valid, n_clean_windows=n_clean, total_windows=total_windows, quality_score=quality_score)

def run_preprocessing_pipeline(all_trials: List[TrialData], config: AppConfig) -> Tuple[List[TrialData], pd.DataFrame]:
    processed_trials, qc_summary_rows = [], []
    for trial in all_trials:
        trial.filtered_baseline_data = notch_filter_data(filter_data(trial.raw_baseline_data, config), config)
        stim_offset = int(config.win.stim_start * config.filter.sfreq)
        stim_data_to_filter = trial.raw_stim_data[:, stim_offset:] if trial.raw_stim_data.shape[1] > stim_offset else None
        trial.filtered_stim_data = notch_filter_data(filter_data(stim_data_to_filter, config), config)
        
        trial.clean_baseline_data, trial.qc_info['baseline'] = get_clean_windows(trial.filtered_baseline_data, config, config.win.baseline_samples)
        trial.clean_stim_data, trial.qc_info['stim'] = get_clean_windows(trial.filtered_stim_data, config, config.win.stim_samples)
        
        trial.is_valid = trial.qc_info['baseline'].is_valid and trial.qc_info['stim'].is_valid
        processed_trials.append(trial)
        
        qc_summary_rows.append({
            "subject_id": trial.subject_id, "trial_id": trial.trial_id, "preference": trial.preference.value,
            "is_valid": trial.is_valid,
            "baseline_clean_windows": trial.qc_info['baseline'].n_clean_windows,
            "baseline_total_windows": trial.qc_info['baseline'].total_windows,
            "stim_clean_windows": trial.qc_info['stim'].n_clean_windows,
            "stim_total_windows": trial.qc_info['stim'].total_windows
        })
    return processed_trials, pd.DataFrame(qc_summary_rows)
