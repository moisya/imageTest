# src/io_module.py
import mne
import numpy as np
import pandas as pd
import json
import re
from typing import List, Dict, Tuple, Optional
import tempfile
from pathlib import Path
import os
import streamlit as st

try:
    import pyxdf
except ImportError:
    pyxdf = None

from utils import AppConfig, TrialData, PreferenceLabel

def load_xdf_as_raw(file_path: str, config: AppConfig) -> Optional[Tuple[mne.io.Raw, pd.DataFrame]]:
    """XDFをMNE RawオブジェクトとマーカーDataFrameに変換。ラベルがない場合も考慮。"""
    if pyxdf is None: return None, pd.DataFrame()
    try:
        streams, _ = pyxdf.load_xdf(file_path)
    except Exception as e:
        st.error(f"XDFファイルの読み込みでエラーが発生しました: {e}")
        return None, pd.DataFrame()

    eeg_stream, marker_stream_raw = None, None
    for s in streams:
        stype = s['info']['type'][0].lower()
        if 'eeg' in stype and int(s['info']['channel_count'][0]) >= 2:
            eeg_stream = s
        elif 'markers' in stype:
            marker_stream_raw = s
    
    if eeg_stream is None:
        st.warning("XDFファイル内に2チャネル以上のEEGストリームが見つかりませんでした。")
        return None, pd.DataFrame()
    
    eeg_data = eeg_stream['time_series'][:, :2].T # 先頭2チャネルを強制的に使用
    sfreq = float(eeg_stream['info']['nominal_srate'][0])

    # ★★★ ラベルの有無を安全にチェック ★★★
    ch_names = ['FP1', 'FP2'] # デフォルトのチャンネル名
    try:
        # ラベルが存在するか試す
        labels_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']
        if labels_info and isinstance(labels_info, list) and len(labels_info) >= 2:
            # ラベルが存在し、かつ2つ以上あればそれを使用
            ch_names = [ch['label'][0] for ch in labels_info][:2]
    except (TypeError, KeyError, IndexError):
        # ラベル情報が欠損している場合はデフォルト名を使用
        st.info("EEGストリームにチャンネルラベル情報が見つかりませんでした。先頭2chを 'FP1', 'FP2' と仮定します。")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    markers_df = pd.DataFrame()
    if marker_stream_raw:
        markers_df = parse_xdf_markers(marker_stream_raw)
        if not markers_df.empty:
            onsets = markers_df['marker_time'].values - eeg_stream['time_stamps'][0]
            descriptions = markers_df['marker_value'].astype(str).values
            raw.set_annotations(mne.Annotations(onset=onsets, duration=0, description=descriptions))
            
    return raw, markers_df

# ... (以降の関数は前回提案から変更なし) ...
# normalize_subject_id, load_survey_data, load_all_trial_data, etc.
