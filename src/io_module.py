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


def extract_subject_id_from_filename(fname: str) -> Optional[str]:
    """ファイル名から 'Sub<N>' 形式の被験者IDを堅牢に抽出する"""
    stem = Path(fname).stem
    # 様々な命名規則に対応する正規表現パターン
    patterns = [
        r'sub[-_]?P?0*(\d+)',        # 例: sub-P001, sub_01, sub-1
        r'experiment_data_(\d+)',  # 例: experiment_data_1
        r'\bsid_?0*(\d+)\b',          # 例: sid_01, sid1
        r'\b(\d+)\b'                 # 例: 001, 1 (最終手段)
    ]
    for pat in patterns:
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            return f'Sub{int(m.group(1))}'
    return None

def parse_xdf_markers(marker_stream: Dict) -> pd.DataFrame:
    # (この関数は変更なし)
    rows = []
    if 'time_stamps' not in marker_stream or 'time_series' not in marker_stream:
        return pd.DataFrame()
    for ts, val_list in zip(marker_stream['time_stamps'], marker_stream['time_series']):
        if not val_list or not val_list[0]: continue
        val_str = val_list[0]
        marker_value = None
        try:
            obj = json.loads(val_str)
            if isinstance(obj, dict) and 'img_id' in obj: marker_value = obj.get('img_id')
        except (json.JSONDecodeError, TypeError): marker_value = val_str
        if marker_value is not None:
            try: rows.append({'marker_time': ts, 'marker_value': int(marker_value)})
            except (ValueError, TypeError): continue
    return pd.DataFrame(rows)

def load_survey_data(uploaded_file) -> Optional[pd.DataFrame]:
    # (この関数は変更なし)
    try:
        fname = uploaded_file.name
        if fname.endswith('.csv'): df = pd.read_csv(uploaded_file)
        elif fname.endswith(('.xlsx', '.xls')): df = pd.read_excel(uploaded_file, engine='openpyxl')
        else: return None
        
        df.columns = [col.lower().replace(' ', '').replace('_', '') for col in df.columns]
        
        id_variants = ['subjectid', 'subject', 'id', 'sid']
        subject_id_col = next((v for v in id_variants if v in df.columns), None)
        if subject_id_col is not None: df = df.rename(columns={subject_id_col: 'subject_id'})
        
        trial_variants = ['trialid', 'imgid', 'imageid']
        trial_id_col = next((v for v in trial_variants if v in df.columns), None)
        if trial_id_col is not None: df = df.rename(columns={trial_id_col: 'trial_id'})
        
        rename_map = {'dislikelike': 'like_score', 'samval': 'valence', 'samaro': 'arousal'}
        for original, new_name in rename_map.items():
            col_found = next((c for c in df.columns if original in c), None)
            if col_found: df = df.rename(columns={col_found: new_name})
        
        return df
    except Exception as e:
        st.error(f'評価データ {uploaded_file.name} の読み込みに失敗: {e}')
        return None

def load_xdf_as_raw(file_path: str, config: AppConfig) -> Optional[mne.io.Raw]:
    # (戻り値をシンプルに変更)
    if pyxdf is None: return None
    try:
        streams, _ = pyxdf.load_xdf(file_path)
    except Exception as e:
        st.error(f'XDFファイル {Path(file_path).name} の読み込みでエラー: {e}')
        return None

    eeg_stream, marker_stream_raw = None, None
    for s in streams:
        stype = s['info']['type'][0].lower()
        if 'eeg' in stype and int(s['info']['channel_count'][0]) >= 2: eeg_stream = s
        elif 'markers' in stype: marker_stream_raw = s
    
    if eeg_stream is None: return None

    eeg_data = eeg_stream['time_series'][:, :2].T
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    config.filter.sfreq = sfreq
    ch_names = ['FP1', 'FP2']
    try:
        labels_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']
        if labels_info and isinstance(labels_info, list) and len(labels_info) >= 2:
            ch_names = [ch['label'][0] for ch in labels_info][:2]
    except (TypeError, KeyError, IndexError, AttributeError):
        st.info(f'{Path(file_path).name} にチャンネルラベル情報がありません。先頭2chを FP1, FP2 と仮定します。')
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    
    if marker_stream_raw:
        markers_df = parse_xdf_markers(marker_stream_raw)
        if not markers_df.empty:
            onsets = markers_df['marker_time'].values - eeg_stream['time_stamps'][0]
            descriptions = markers_df['marker_value'].astype(str).values
            raw.set_annotations(mne.Annotations(onset=onsets, duration=0, description=descriptions))
    return raw

def get_events_from_raw(raw: mne.io.Raw) -> np.ndarray:
    # (この関数は変更なし)
    try:
        events, _ = mne.events_from_annotations(raw, verbose=False)
        return events
    except (ValueError, IndexError):
        return generate_dummy_events(raw)

def generate_dummy_events(raw: mne.io.Raw) -> np.ndarray:
    # (この関数は変更なし)
    st.warning('イベントマーカーが見つかりませんでした。13秒間隔でダミーイベントを生成します。')
    trial_duration = 13.0
    n_trials = int(raw.n_times / raw.info['sfreq'] // trial_duration)
    events = []
    for i in range(n_trials):
        event_sample = int((i * trial_duration + 3.0) * raw.info['sfreq'])
        event_id = i + 1
        if event_id > 2: events.append([event_sample, 0, event_id])
    return np.array(events)

def extract_trials(raw: mne.io.Raw, events: np.ndarray, config: AppConfig, subject_id: str, survey_df: Optional[pd.DataFrame]) -> List[TrialData]:
    # (この関数は変更なし)
    trials: List[TrialData] = []
    sfreq = raw.info['sfreq']
    valid_events = events[events[:, 2] > 2] if np.any(events[:, 2] > 2) else events
    for event in valid_events:
        trial_id = int(event[2])
        event_sample = event[0]
        baseline_start = event_sample - int(config.win.baseline_len * sfreq)
        stim_end = event_sample + int(config.win.stim_end * sfreq)
        if baseline_start < 0 or stim_end > raw.n_times: continue
        
        baseline_data = raw.get_data(start=baseline_start, stop=event_sample)
        stim_data = raw.get_data(start=event_sample, stop=stim_end)
        
        preference, valence, arousal = PreferenceLabel.NEUTRAL, None, None
        if survey_df is not None and not survey_df.empty:
            trial_survey = survey_df[survey_df['trial_id'] == trial_id]
            if not trial_survey.empty:
                row = trial_survey.iloc[0]
                if 'like_score' in row and pd.notna(row['like_score']):
                    score = row['like_score']
                    if score >= 6: preference = PreferenceLabel.LIKE
                    elif score <= 2: preference = PreferenceLabel.DISLIKE
                if 'valence' in row and pd.notna(row['valence']): valence = row['valence']
                if 'arousal' in row and pd.notna(row['arousal']): arousal = row['arousal']
        
        trials.append(TrialData(
            subject_id=subject_id, trial_id=trial_id, preference=preference,
            raw_baseline_data=baseline_data, raw_stim_data=stim_data,
            valence=valence, arousal=arousal
        ))
    return trials

def pair_subject_files(eeg_files: List, survey_files: List) -> Dict[str, Dict[str, List]]:
    """アップロードされたファイルを被験者IDごとにグルーピングする"""
    subjects: Dict[str, Dict[str, List]] = {}
    
    all_files = [('eeg', f) for f in eeg_files] + [('survey', f) for f in survey_files]
    
    for file_type, f in all_files:
        sid = extract_subject_id_from_filename(f.name)
        if sid is None:
            st.warning(f'ファイル名 {f.name} から被験者IDを抽出できませんでした。スキップします。')
            continue
        
        subjects.setdefault(sid, {'eeg': [], 'survey': []})[file_type].append(f)
        
    return subjects

def load_all_trial_data(uploaded_eeg_files: List, uploaded_survey_files: List, config: AppConfig) -> Tuple[List[TrialData], dict]:
    """被験者ごとにグループ化されたファイルから全試行データを読み込む"""
    subjects = pair_subject_files(uploaded_eeg_files, uploaded_survey_files)
    all_trials: List[TrialData] = []
    
    for sid, files in subjects.items():
        if not files['eeg']:
            st.warning(f"被験者 {sid}: EEG (.xdf) ファイルが見つかりません。スキップします。")
            continue
        
        # 1被験者につき1つのEEGファイルと、対応する評価ファイル(あれば)を処理
        eeg_file = files['eeg'][0]
        
        survey_df = None
        if files['survey']:
            all_subject_surveys = [df for f in files['survey'] if (df := load_survey_data(f)) is not None]
            if all_subject_surveys:
                survey_df = pd.concat(all_subject_surveys, ignore_index=True)
                
        raw = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(eeg_file.name).suffix) as tmp:
            tmp.write(eeg_file.getvalue())
            file_path = tmp.name
        try:
            if eeg_file.name.lower().endswith('.xdf'):
                raw = load_xdf_as_raw(file_path, config)
        finally:
            os.unlink(file_path)
            
        if raw is None:
            st.error(f"被験者 {sid} のEEGファイル {eeg_file.name} の読み込みに失敗しました。")
            continue
            
        events = get_events_from_raw(raw)
        if len(events) == 0:
            st.warning(f"被験者 {sid} のイベントが見つかりませんでした。")
            continue
            
        events_df = pd.DataFrame(events, columns=['sample', 'zero', 'event_id'])
        unique_events_df = events_df.drop_duplicates(subset='event_id', keep='first')
        unique_events = unique_events_df.to_numpy()
        
        if len(events) > len(unique_events):
            st.info(f"被験者 {sid}: {len(events)}個のイベントを検出し、重複を除去して {len(unique_events)}個のユニークな試行を処理します。")
            
        subject_trials = extract_trials(raw, unique_events, config, sid, survey_df)
        all_trials.extend(subject_trials)
        
    return all_trials, {}
