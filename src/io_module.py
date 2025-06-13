# src/io_module.py
import mne
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import tempfile
from pathlib import Path
import os
import warnings
import streamlit as st # エラー表示のためにインポート

# pyxdfをインポート
try:
    import pyxdf
except ImportError:
    st.error("XDFファイルを処理するために `pyxdf` ライブラリが必要です。`pip install pyxdf` を実行してください。")
    pyxdf = None

from utils import AppConfig, TrialData, PreferenceLabel

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')

def load_xdf_file(file_path: str, config: AppConfig) -> Optional[mne.io.Raw]:
    """XDFファイルを読み込み、MNE Rawオブジェクトに変換する"""
    if pyxdf is None: return None
    
    try:
        streams, header = pyxdf.load_xdf(file_path)
    except Exception as e:
        st.error(f"XDFファイルの読み込みに失敗しました: {e}")
        return None

    eeg_stream, marker_stream = None, None
    for stream in streams:
        if stream['info']['type'][0].lower() == 'eeg':
            eeg_stream = stream
        if stream['info']['type'][0].lower() == 'markers':
            marker_stream = stream

    if eeg_stream is None:
        st.warning("XDFファイル内にEEGストリームが見つかりませんでした。")
        return None

    # EEGデータと情報を抽出
    eeg_data = eeg_stream['time_series'].T  # (n_channels, n_samples)
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    config.filter.sfreq = sfreq # 設定オブジェクトを更新

    # チャンネル名を取得
    ch_names_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']
    ch_names = [ch['label'][0] for ch in ch_names_info]

    # MNE Rawオブジェクトを作成
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    # マーカー情報からアノテーションを作成
    if marker_stream is not None:
        onsets = marker_stream['time_stamps'] - eeg_stream['time_stamps'][0] # EEG開始からの相対時間
        descriptions = [m[0] for m in marker_stream['time_series']]
        annotations = mne.Annotations(onset=onsets, duration=0, description=descriptions)
        raw.set_annotations(annotations)
    
    return raw

def load_single_file(uploaded_file, config: AppConfig) -> Optional[mne.io.Raw]:
    """アップロードされた単一ファイルをMNE Rawオブジェクトとして読み込む"""
    try:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        if suffix == '.xdf':
            raw = load_xdf_file(tmp_path, config)
        elif suffix in ['.edf', '.bdf']:
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        elif suffix == '.fif':
            raw = mne.io.read_raw_fif(tmp_path, preload=True, verbose=False)
        else:
            st.warning(f"サポートされていないファイル形式: {suffix}")
            raw = None
        
        os.unlink(tmp_path)
        return raw
    except Exception as e:
        st.error(f"ファイル {uploaded_file.name} の読み込み中にエラーが発生しました: {e}")
        return None

# ... (prepare_raw, generate_dummy_events, extract_trials は前回とほぼ同じ) ...

def extract_trials(raw: mne.io.Raw, events: np.ndarray, config: AppConfig, subject_id: str) -> List[TrialData]:
    # (前回と同じコード)
    trials = []
    sfreq = raw.info['sfreq']
    
    for i, event in enumerate(events):
        event_sample = event[0]
        
        baseline_start = event_sample - int(config.win.baseline_len * sfreq)
        stim_end = event_sample + int(config.win.stim_end * sfreq)
        
        if baseline_start < 0 or stim_end > len(raw.times):
            continue

        baseline_data = raw.get_data(start=baseline_start, stop=event_sample)
        stim_data = raw.get_data(start=event_sample, stop=stim_end)
        
        # 1 or '1' -> 好き, 2 or '2' -> そうでもない
        event_id = str(event[2])
        preference = PreferenceLabel.LIKE if event_id.startswith('1') else PreferenceLabel.NEUTRAL

        trials.append(TrialData(
            subject_id=subject_id,
            trial_id=i,
            preference=preference,
            raw_baseline_data=baseline_data,
            raw_stim_data=stim_data
        ))
    return trials

def load_all_trial_data(uploaded_files: List, config: AppConfig) -> Tuple[List[TrialData], dict]:
    """メインの読み込み関数"""
    all_trials = []
    meta_info = {} # 省略

    for i, file in enumerate(uploaded_files):
        subject_id = f"S{i+1:02d}"
        raw = load_single_file(file, config)
        if raw is None: continue
        
        raw = prepare_raw(raw, config)
        
        try:
            # アノテーションからイベントを優先的に取得
            if raw.annotations and len(raw.annotations) > 0:
                events, event_id = mne.events_from_annotations(raw)
                # event_id は {'1': 1, '2': 2} のような辞書になる
                # event_id のキーを整数に変換しようとするエラーを防ぐため、文字列のまま扱う
                # MNEが返すイベントIDは、アノテーションの説明を辞書順にソートしたものになる
                # ここでは単純化し、説明が '1' or '2' であることを期待
            else:
                 # 従来の方法
                events = mne.find_events(raw, stim_channel='STI 014', min_duration=0.002, verbose=False)

            # 画像1, 2に対応するイベントを除外
            if events.shape[1] > 2:
                events = events[events[:, 2] > 2] if np.any(events[:, 2] > 2) else events
                events[:, 2] -= 2
        except (ValueError, RuntimeError):
            events = generate_dummy_events(raw, config)

        subject_trials = extract_trials(raw, events, config, subject_id)
        all_trials.extend(subject_trials)
    
    return all_trials, meta_info

def prepare_raw(raw: mne.io.Raw, config: AppConfig) -> mne.io.Raw:
    target_ch = ['FP1', 'FP2']
    # 大文字小文字の違いを吸収
    ch_names_upper = [ch.upper() for ch in raw.ch_names]
    ch_mapping = {orig: upper for orig, upper in zip(raw.ch_names, ch_names_upper)}
    raw.rename_channels(ch_mapping)
    
    missing_ch = [ch for ch in target_ch if ch not in raw.ch_names]
    if missing_ch:
        st.warning(f"{missing_ch} が見つかりません。利用可能な最初の2チャネルを使用します。")
        raw.pick_channels(raw.ch_names[:2])
        raw.rename_channels({raw.ch_names[0]: 'FP1', raw.ch_names[1]: 'FP2'})
    else:
        raw.pick_channels(target_ch)

    config.filter.sfreq = raw.info['sfreq']
    if raw.info['sfreq'] > 300:
        raw.resample(250, npad='auto', verbose=False)
        config.filter.sfreq = 250.0
    
    return raw

def generate_dummy_events(raw: mne.io.Raw, config: AppConfig) -> np.ndarray:
    # (前回と同じコード)
    trial_duration = config.win.baseline_len + config.win.stim_end
    n_trials = int(raw.times[-1] // trial_duration)
    
    events = []
    for i in range(2, n_trials): # 画像1, 2は使用しない
        event_sample = int(i * trial_duration * raw.info['sfreq'])
        event_id = 1 if (i-2) % 2 == 0 else 2 # 交互に好き・そうでもない
        events.append([event_sample, 0, event_id])
    
    return np.array(events)
