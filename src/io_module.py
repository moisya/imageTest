# src/io_module.py
import mne
import numpy as np
import pandas as pd
import json
import re # 正規表現モジュールをインポート
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

# --- ID正規化ヘルパー ---
def normalize_subject_id(raw_id: str) -> str:
    """様々な形式のIDを 'Sub1', 'Sub2' の形式に正規化する"""
    # 数値のみを抽出
    nums = re.findall(r'\d+', str(raw_id))
    if not nums:
        return str(raw_id) # 数値が見つからない場合はそのまま
    # 最初の数値を使い、'Sub'プレフィックスを付ける
    return f"Sub{int(nums[0])}"

# --- XDF読み込み関数 (NameError修正) ---
def load_xdf_as_raw(file_path: str, config: AppConfig) -> Optional[Tuple[mne.io.Raw, pd.DataFrame]]:
    # (この関数は前回提案したものを流用し、エラーがないことを確認)
    if pyxdf is None: return None
    try:
        streams, _ = pyxdf.load_xdf(file_path)
    except Exception:
        return None, pd.DataFrame() # エラー時は空を返す

    eeg_stream, marker_stream_raw = None, None
    for s in streams:
        stype = s['info']['type'][0].lower()
        if 'eeg' in stype: eeg_stream = s
        elif 'markers' in stype: marker_stream_raw = s
    
    if eeg_stream is None: return None, pd.DataFrame()
    
    eeg_data = eeg_stream['time_series'][:, :2].T
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    ch_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']][:2]
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

# --- アンケート読み込み関数 (ID正規化を追加) ---
def load_survey_data(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        fname = uploaded_file.name
        if fname.endswith('.csv'): df = pd.read_csv(uploaded_file)
        elif fname.endswith(('.xlsx', '.xls')): df = pd.read_excel(uploaded_file)
        else: return None
        
        # 列名を小文字化
        df.columns = [col.lower() for col in df.columns]

        # 'sid' 列を 'subject_id' として認識
        if 'sid' in df.columns:
            df = df.rename(columns={'sid': 'subject_id'})
        elif 'subjectid' not in df.columns: # subjectidも探す
             st.error("評価データに 'sid' または 'subject_id' 列が見つかりません。")
             return None

        # trial_id 列の正規化
        if 'img_id' in df.columns:
            df = df.rename(columns={'img_id': 'trial_id'})
        if 'trial_id' not in df.columns:
            st.error("評価データに 'trial_id' または 'img_id' 列が見つかりません。")
            return None
            
        # ★★★ subject_id列を正規化 ★★★
        df['subject_id'] = df['subject_id'].apply(normalize_subject_id)
        df['trial_id'] = pd.to_numeric(df['trial_id'], errors='coerce').dropna().astype(int)
        
        st.success(f"評価データ読み込み完了 ({len(df)}件)")
        return df
    except Exception as e:
        st.error(f"評価データの読み込みに失敗しました: {e}")
        return None

# --- メインのパイプライン関数 (ID正規化を追加) ---
def load_all_trial_data(uploaded_eeg_files: List, uploaded_survey_file, config: AppConfig) -> Tuple[List[TrialData], dict]:
    survey_df = load_survey_data(uploaded_survey_file) if uploaded_survey_file else None

    all_trials = []
    for file in uploaded_eeg_files:
        # ★★★ ファイル名から被験者IDを抽出し、正規化 ★★★
        try:
            # 例: "sub-P001..." -> "P001" -> "Sub1"
            subject_id_from_filename = normalize_subject_id(Path(file.name).stem)
        except Exception:
            continue # ファイル名からIDが取れなければスキップ

        subject_survey_df = None
        if survey_df is not None:
            # ★★★ 正規化されたIDでマッチング ★★★
            subject_survey_df = survey_df[survey_df['subject_id'] == subject_id_from_filename]
            if subject_survey_df.empty:
                st.warning(f"アンケートデータに被験者 '{subject_id_from_filename}' のデータが見つかりませんでした。")
        
        raw, _ = None, None # 初期化
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.getvalue())
            file_path = tmp_file.name
        try:
            if file.name.lower().endswith('.xdf'):
                raw, _ = load_xdf_as_raw(file_path, config)
        finally:
            os.unlink(file_path)

        if raw is None: continue
        
        events = get_events_from_raw(raw)
        subject_trials = extract_trials(raw, events, config, subject_id_from_filename, subject_survey_df)
        all_trials.extend(subject_trials)
        
    return all_trials, {}

# ... (extract_trials, get_events_from_raw, generate_dummy_events, parse_xdf_markers などは前回提案から流用可能) ...
