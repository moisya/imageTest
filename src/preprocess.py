# src/preprocess.py

import mne
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from utils import AppConfig, TrialData, QCResult, WindowConfig

def filter_data(data: np.ndarray, config: AppConfig) -> np.ndarray:
    """MNEのフィルタ関数をラップしてEEGデータにフィルタを適用する"""
    # データが空、または短すぎる場合は何もしない
    if data is None or data.shape[1] < 10:
        return data
        
    return mne.filter.filter_data(
        data, 
        sfreq=config.filter.sfreq, 
        l_freq=config.filter.l_freq, 
        h_freq=config.filter.h_freq, 
        fir_design='firwin',
        verbose=False
    )

def notch_filter_data(data: np.ndarray, config: AppConfig) -> np.ndarray:
    """MNEのノッチフィルタ関数をラップして適用する"""
    if data is None or data.shape[1] < 10:
        return data

    return mne.filter.notch_filter(
        data,
        Fs=config.filter.sfreq,
        freqs=config.filter.notch_freq,
        fir_design='firwin',
        verbose=False
    )

def check_window_quality(window: np.ndarray, config: AppConfig) -> bool:
    """
    単一の窓の品質をチェックする。
    データの単位がVかµVかに依存しないよう、単位変換 (* 1e6) を削除。
    ユーザーがデータのスケールに合わせて閾値を設定することを想定。
    """
    if window is None:
        return False

    # 振幅閾値チェック
    if np.any(np.abs(window) > config.qc.amp_uV):
        return False
    
    # 隣接サンプル差（急峻な変化）チェック
    if np.any(np.abs(np.diff(window, axis=1)) > config.qc.diff_uV):
        return False
        
    return True

def get_clean_windows(data: np.ndarray, config: AppConfig, num_samples_to_take: int) -> Tuple[Optional[np.ndarray], QCResult]:
    """データからクリーンな窓を抽出し、指定された数だけランダムにサンプリングして平均化する"""
    win_len_samples = int(config.win.win_len * config.filter.sfreq)
    
    if data is None or data.shape[1] < win_len_samples:
        return None, QCResult(is_valid=False, n_clean_windows=0, total_windows=0, quality_score=0.0)

    total_windows = data.shape[1] // win_len_samples
    
    clean_windows = []
    for i in range(total_windows):
        window = data[:, i*win_len_samples:(i+1)*win_len_samples]
        if check_window_quality(window, config):
            clean_windows.append(window)

    n_clean = len(clean_windows)
    quality_score = n_clean / total_windows if total_windows > 0 else 0.0
    
    # 有効な窓が、要求されたサンプル数以上あるか？
    if n_clean >= num_samples_to_take:
        # ランダムにサンプリング（再現性のためにシードを固定）
        np.random.seed(42)
        indices = np.random.choice(n_clean, num_samples_to_take, replace=False)
        selected_windows = [clean_windows[i] for i in indices]
        
        avg_clean_data = np.mean(selected_windows, axis=0)
        is_valid = True
    else:
        avg_clean_data = None
        is_valid = False

    qc_result = QCResult(
        is_valid=is_valid, 
        n_clean_windows=n_clean,
        total_windows=total_windows,
        quality_score=quality_score
    )
    return avg_clean_data, qc_result

def run_preprocessing_pipeline(all_trials: List[TrialData], config: AppConfig) -> Tuple[List[TrialData], pd.DataFrame]:
    """前処理パイプライン全体を実行する"""
    processed_trials = []
    qc_summary_rows = []

    for trial in all_trials:
        # 1. フィルタリング
        trial.filtered_baseline_data = notch_filter_data(filter_data(trial.raw_baseline_data, config), config)
        
        # 刺激区間は開始0.5秒後からを対象とする
        stim_offset_samples = int(config.win.stim_start * config.filter.sfreq)
        stim_data_for_filtering = trial.raw_stim_data[:, stim_offset_samples:]
        trial.filtered_stim_data = notch_filter_data(filter_data(stim_data_for_filtering, config), config)

        # 2. 品質管理と窓抽出
        trial.clean_baseline_data, trial.qc_info['baseline'] = get_clean_windows(
            trial.filtered_baseline_data, config, config.win.baseline_samples
        )
        
        trial.clean_stim_data, trial.qc_info['stim'] = get_clean_windows(
            trial.filtered_stim_data, config, config.win.stim_samples
        )

        # 3. 試行の有効性を最終判断
        trial.is_valid = trial.qc_info['baseline'].is_valid and trial.qc_info['stim'].is_valid
        processed_trials.append(trial)
        
        # 4. サマリー情報を作成
        qc_summary_rows.append({
            "subject_id": trial.subject_id,
            "trial_id": trial.trial_id,
            "preference": trial.preference.value,
            "is_valid": trial.is_valid,
            "baseline_clean_windows": trial.qc_info['baseline'].n_clean_windows,
            "baseline_total_windows": trial.qc_info['baseline'].total_windows,
            "stim_clean_windows": trial.qc_info['stim'].n_clean_windows,
            "stim_total_windows": trial.qc_info['stim'].total_windows,
        })

    return processed_trials, pd.DataFrame(qc_summary_rows)
