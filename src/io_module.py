# src/io_module.py
import mne
import numpy as np
import pandas as pd
import json
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

# --- データ読み込みヘルパー ---
def load_survey_data(uploaded_file) -> Optional[pd.DataFrame]:
    """CSVまたはExcel形式のアンケートデータを読み込む"""
    try:
        fname = uploaded_file.name
        if fname.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif fname.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("評価データはCSVまたはExcel形式である必要があります。")
            return None
        
        # trial_id/img_id列の正規化
        if 'img_id' in df.columns:
            df = df.rename(columns={'img_id': 'trial_id'})
        if 'trial_id' not in df.columns:
            st.error("評価データに必須列 'trial_id' (または 'img_id') が見つかりません。")
            return None
        
        df['trial_id'] = pd.to_numeric(df['trial_id'], errors='coerce').dropna().astype(int)
        st.success(f"評価データ読み込み完了 ({len(df)}件)")
        return df
    except Exception as e:
        st.error(f"評価データの読み込みに失敗しました: {e}")
        return None

def parse_xdf_markers(marker_stream: Dict) -> pd.DataFrame:
    """XDFのマーカーストリームをパースしてDataFrameを作成する"""
    rows = []
    for ts, val_list in zip(marker_stream['time_stamps'], marker_stream['time_series']):
        if not val_list or not val_list[0]: continue
        val_str = val_list[0]
        marker_value = None
        try:
            # JSON形式のマーカーを試す
            obj = json.loads(val_str)
            if isinstance(obj, dict) and 'img_id' in obj:
                marker_value = obj.get('img_id')
        except (json.JSONDecodeError, TypeError):
            # JSONでない場合、単純な数値としてパースを試す
            marker_value = val_str
        
        if marker_value is not None:
            try:
                rows.append({'marker_time': ts, 'marker_value': int(marker_value)})
            except (ValueError, TypeError):
                continue # intに変換できないマーカーは無視
    return pd.DataFrame(rows)

def load_xdf_as_raw(file_path: str, config: AppConfig) -> Optional[Tuple[mne.io.Raw, pd.DataFrame]]:
    """XDFをMNE RawオブジェクトとマーカーDataFrameに変換"""
    if pyxdf is None: return None
    streams, _ = pyxdf.load_xdf(file_path)

    eeg_stream, marker_stream_raw = None, None
    for s in streams:
        stype = s['info']['type'][0].lower()
        if 'eeg' in stype and s['time_series'].shape[1] >= 2:
            eeg_stream = s
        elif 'markers' in stype:
            marker_stream_raw = s

    if eeg_stream is None: return None

    eeg_data = eeg_stream['time_series'][:, :2].T # FP1, FP2のみと仮定
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    ch_names = ['FP1', 'FP2']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info, first_samp=int(eeg_stream['time_stamps'][0] * sfreq), verbose=False)
    
    markers_df = pd.DataFrame()
    if marker_stream_raw:
        markers_df = parse_xdf_markers(marker_stream_raw)
        if not markers_df.empty:
            # マーカーの時間をサンプル数に変換
            onsets_sec = markers_df['marker_time'].values - raw.first_samp / sfreq
            descriptions = markers_df['marker_value'].astype(str).values
            raw.set_annotations(mne.Annotations(onset=onsets_sec, duration=0, description=descriptions))

    return raw, markers_df

# --- メインのパイプライン関数 ---
def load_all_trial_data(uploaded_eeg_files: List, uploaded_survey_file, config: AppConfig) -> Tuple[List[TrialData], dict]:
    survey_df = load_survey_data(uploaded_survey_file) if uploaded_survey_file else None

    all_trials = []
    for i, file in enumerate(uploaded_eeg_files):
        subject_id = Path(file.name).stem.split('_')[0]
        subject_survey_df = survey_df[survey_df['subject_id'] == subject_id] if survey_df is not None else None
        
        raw, markers_df = None, None
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.getvalue())
            file_path = tmp_file.name

        try:
            if file.name.lower().endswith('.xdf'):
                raw, markers_df = load_xdf_as_raw(file_path, config)
            # (ここにEDF, BDFなどの読み込みロジックを追加可能)
        finally:
            os.unlink(file_path)

        if raw is None: continue

        events = get_events_from_raw(raw)
        subject_trials = extract_trials(raw, events, config, subject_id, subject_survey_df)
        all_trials.extend(subject_trials)
        
    return all_trials, {}

def get_events_from_raw(raw: mne.io.Raw) -> np.ndarray:
    try:
        return mne.events_from_annotations(raw, verbose=False)[0]
    except (ValueError, IndexError):
        # アノテーションがない、または無効な場合、ダミーイベントを生成
        return generate_dummy_events(raw)

def extract_trials(raw: mne.io.Raw, events: np.ndarray, config: AppConfig, subject_id: str, survey_df: Optional[pd.DataFrame]) -> List[TrialData]:
    trials = []
    sfreq = raw.info['sfreq']
    
    # 画像1, 2はテストなので除外 (イベントID > 2)
    valid_events = events[events[:, 2] > 2]

    for event in valid_events:
        trial_id = int(event[2]) # イベントIDを試行IDとする
        event_sample = event[0]
        
        baseline_start = event_sample - int(config.win.baseline_len * sfreq)
        stim_end = event_sample + int(config.win.stim_end * sfreq)
        if baseline_start < raw.first_samp or stim_end > raw.last_samp: continue
        
        baseline_data = raw.get_data(start=baseline_start, stop=event_sample)
        stim_data = raw.get_data(start=event_sample, stop=stim_end)
        
        preference = PreferenceLabel.NEUTRAL # デフォルト
        # アンケートデータから嗜好ラベルを取得
        if survey_df is not None:
            trial_survey = survey_df[survey_df['trial_id'] == trial_id]
            if not trial_survey.empty and 'Dislike_Like' in trial_survey.columns:
                score = trial_survey['Dislike_Like'].iloc[0]
                if pd.notna(score):
                    preference = PreferenceLabel.LIKE if score >= 5 else PreferenceLabel.NEUTRAL # 7段階評価の5以上を「好き」と仮定
        
        trials.append(TrialData(
            subject_id=subject_id, trial_id=trial_id, preference=preference,
            raw_baseline_data=baseline_data, raw_stim_data=stim_data
        ))
    return trials

def generate_dummy_events(raw: mne.io.Raw) -> np.ndarray:
    trial_duration = 13.0
    n_trials = int((raw.last_samp - raw.first_samp) / raw.info['sfreq'] // trial_duration)
    events = []
    for i in range(n_trials):
        event_sample = raw.first_samp + int((i * trial_duration + 3.0) * raw.info['sfreq'])
        event_id = i + 1
        if event_id > 2: # 1,2を除外
            events.append([event_sample, 0, event_id])
    return np.array(events)
