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

def normalize_subject_id(raw_id: str) -> str:
    nums = re.findall(r'\d+', str(raw_id))
    if not nums: return str(raw_id)
    return f"Sub{int(nums[0])}"

def parse_xdf_markers(marker_stream: Dict) -> pd.DataFrame:
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
    try:
        fname = uploaded_file.name
        if fname.endswith('.csv'): df = pd.read_csv(uploaded_file)
        elif fname.endswith(('.xlsx', '.xls')): df = pd.read_excel(uploaded_file, engine='openpyxl')
        else: return None
        
        df.columns = [col.lower().replace(" ", "").replace("_", "") for col in df.columns]
        
        id_variants = ["subjectid", "subject", "id", "sid"]
        subject_id_col = next((v for v in id_variants if v in df.columns), None)
        if subject_id_col is None:
            st.error(f"評価データ '{fname}' に必須列 ({'/'.join(id_variants)}) が見つかりません。")
            return None
        df = df.rename(columns={subject_id_col: 'subject_id'})
        
        trial_variants = ["trialid", "imgid", "imageid"]
        trial_id_col = next((v for v in trial_variants if v in df.columns), None)
        if trial_id_col is None:
            st.error(f"評価データ '{fname}' に必須列 ({'/'.join(trial_variants)}) が見つかりません。")
            return None
        df = df.rename(columns={trial_id_col: 'trial_id'})
        
        rename_map = {"dislikelike": "like_score", "samval": "valence", "samaro": "arousal"}
        for original, new_name in rename_map.items():
            col_found = next((c for c in df.columns if original in c), None)
            if col_found: df = df.rename(columns={col_found: new_name})

        df['subject_id'] = df['subject_id'].apply(normalize_subject_id)
        df['trial_id'] = pd.to_numeric(df['trial_id'], errors='coerce').dropna().astype(int)
        
        st.success(f"評価データ '{fname}' 読み込み完了")
        return df
    except Exception as e:
        st.error(f"評価データ '{uploaded_file.name}' の読み込みに失敗: {e}")
        return None

def load_xdf_as_raw(file_path: str, config: AppConfig) -> Optional[Tuple[mne.io.Raw, pd.DataFrame]]:
    if pyxdf is None: return None, pd.DataFrame()
    try:
        streams, _ = pyxdf.load_xdf(file_path)
    except Exception as e:
        st.error(f"XDFファイル '{Path(file_path).name}' の読み込みでエラー: {e}")
        return None, pd.DataFrame()
    
    eeg_stream, marker_stream_raw = None, None
    for s in streams:
        stype = s['info']['type'][0].lower()
        if 'eeg' in stype and int(s['info']['channel_count'][0]) >= 2: eeg_stream = s
        elif 'markers' in stype: marker_stream_raw = s
    
    if eeg_stream is None: return None, pd.DataFrame()
    
    eeg_data = eeg_stream['time_series'][:, :2].T
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    config.filter.sfreq = sfreq
    ch_names = ['FP1', 'FP2']
    try:
        labels_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']
        if labels_info and isinstance(labels_info, list) and len(labels_info) >= 2:
            ch_names = [ch['label'][0] for ch in labels_info][:2]
    except (TypeError, KeyError, IndexError, AttributeError):
        st.info(f"'{Path(file_path).name}' にチャンネルラベル情報がありません。先頭2chを 'FP1', 'FP2' と仮定します。")
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    
    if marker_stream_raw:
        markers_df = parse_xdf_markers(marker_stream_raw)
        if not markers_df.empty:
            onsets = markers_df['marker_time'].values - eeg_stream['time_stamps'][0]
            descriptions = markers_df['marker_value'].astype(str).values
            raw.set_annotations(mne.Annotations(onset=onsets, duration=0, description=descriptions))
    return raw, markers_df

def get_events_from_raw(raw: mne.io.Raw) -> np.ndarray:
    try:
        events, _ = mne.events_from_annotations(raw, verbose=False)
        return events
    except (ValueError, IndexError):
        return generate_dummy_events(raw)

def generate_dummy_events(raw: mne.io.Raw) -> np.ndarray:
    st.warning("イベントマーカーが見つかりませんでした。13秒間隔でダミーイベントを生成します。")
    trial_duration = 13.0
    n_trials = int(raw.n_times / raw.info['sfreq'] // trial_duration)
    events = []
    for i in range(n_trials):
        event_sample = int((i * trial_duration + 3.0) * raw.info['sfreq'])
        event_id = i + 1
        if event_id > 2: events.append([event_sample, 0, event_id])
    return np.array(events)

def extract_trials(raw: mne.io.Raw, events: np.ndarray, config: AppConfig, subject_id: str, survey_df: Optional[pd.DataFrame]) -> List[TrialData]:
    trials = []
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
                    else: preference = PreferenceLabel.NEUTRAL
                if 'valence' in row and pd.notna(row['valence']): valence = row['valence']
                if 'arousal' in row and pd.notna(row['arousal']): arousal = row['arousal']
        
        trials.append(TrialData(
            subject_id=subject_id, trial_id=trial_id, preference=preference,
            raw_baseline_data=baseline_data, raw_stim_data=stim_data,
            valence=valence, arousal=arousal
        ))
    return trials

def load_all_trial_data(uploaded_eeg_files: List, uploaded_survey_files: List, config: AppConfig) -> Tuple[List[TrialData], dict]:
    survey_df = None
    if uploaded_survey_files:
        all_survey_dfs = [df for file in uploaded_survey_files if (df := load_survey_data(file)) is not None]
        if all_survey_dfs:
            survey_df = pd.concat(all_survey_dfs, ignore_index=True)
            st.success(f"合計 {len(uploaded_survey_files)} 件の評価ファイルをマージしました。")

    all_trials = []
    for file in uploaded_eeg_files:
        try:
            subject_id = normalize_subject_id(Path(file.name).stem)
        except Exception:
            st.warning(f"ファイル名 {file.name} から被験者IDを抽出できませんでした。スキップします。")
            continue
        
        subject_survey_df = survey_df[survey_df['subject_id'] == subject_id] if survey_df is not None else None
        
        raw, _ = None, pd.DataFrame()
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
        if len(events) == 0: continue
            
        events_df = pd.DataFrame(events, columns=['sample', 'zero', 'event_id'])
        unique_events_df = events_df.drop_duplicates(subset='event_id', keep='first')
        unique_events = unique_events_df.to_numpy()
        
        if len(events) > len(unique_events):
            st.info(f"被験者 {subject_id}: {len(events)}個のイベントを検出し、重複を除去して {len(unique_events)}個のユニークな試行を処理します。")
        
        subject_trials = extract_trials(raw, unique_events, config, subject_id, subject_survey_df)
        all_trials.extend(subject_trials)
        
    return all_trials, {}
