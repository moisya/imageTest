# src/io_module.py
import mne
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import tempfile
from pathlib import Path
import os
import warnings
import streamlit as st

try:
    import pyxdf
except ImportError:
    pyxdf = None

from utils import AppConfig, TrialData, PreferenceLabel

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')

def find_streams(streams: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """XDFストリームの中からEEGとマーカーのストリームを柔軟に探索する"""
    eeg_stream, marker_stream = None, None
    for stream in streams:
        stype = stream['info']['type'][0].lower()
        if 'eeg' in stype:
            eeg_stream = stream
        elif 'markers' in stype:
            marker_stream = stream
    
    if eeg_stream is None:
        st.warning("XDFファイル内にEEGタイプのストリームが見つかりませんでした。")
    if marker_stream is None:
        st.info("XDFファイル内にマーカータイプのストリームが見つかりませんでした。ダミーイベントを生成します。")
        
    return eeg_stream, marker_stream

def load_xdf_file(file_path: str, config: AppConfig) -> Optional[mne.io.Raw]:
    if pyxdf is None: return None
    
    try:
        streams, header = pyxdf.load_xdf(file_path)
    except Exception as e:
        st.error(f"XDFファイルの読み込みに失敗しました: {e}")
        return None

    eeg_stream, marker_stream = find_streams(streams)

    if eeg_stream is None: return None

    eeg_data = eeg_stream['time_series'].T
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    config.filter.sfreq = sfreq

    ch_names_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']
    ch_names = [ch['label'][0] for ch in ch_names_info]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    # ★★★ エラー修正：marker_streamがNoneでない場合のみアノテーションを設定 ★★★
    if marker_stream and 'time_stamps' in marker_stream and 'time_series' in marker_stream:
        # マーカーのタイムスタンプをEEGストリームの開始時間に合わせる
        onsets = marker_stream['time_stamps'] - eeg_stream['time_stamps'][0]
        descriptions = [str(m[0]) for m in marker_stream['time_series']]
        try:
            raw.set_annotations(mne.Annotations(onset=onsets, duration=0, description=descriptions))
        except Exception as e:
            st.warning(f"アノテーションの設定中にエラーが発生しました: {e}. マーカーは無視されます。")

    return raw

def load_all_trial_data(uploaded_eeg_files: List, uploaded_survey_file, config: AppConfig) -> Tuple[List[TrialData], dict]:
    """メインの読み込み関数。アンケートファイルも受け取るように変更"""
    survey_df = None
    if uploaded_survey_file:
        try:
            survey_df = pd.read_csv(uploaded_survey_file)
            # 必要な列があるかチェック
            if 'subject_id' not in survey_df.columns or 'trial_id' not in survey_df.columns:
                st.error("アンケートCSVには 'subject_id' と 'trial_id' 列が必要です。")
                return [], {}
        except Exception as e:
            st.error(f"アンケートCSVの読み込みに失敗しました: {e}")
            return [], {}

    all_trials = []
    # ... (load_single_file, prepare_rawは前回と同じでOK) ...
    for i, file in enumerate(uploaded_eeg_files):
        # ファイル名から被験者IDを推測 (例: sub-P001_... -> P001)
        try:
            subject_id_raw = Path(file.name).stem.split('_')[0].split('-')[1]
        except IndexError:
            subject_id_raw = f"S{i+1:02d}"
        
        # アンケートデータからこの被験者のデータをフィルタリング
        subject_survey_df = survey_df[survey_df['subject_id'] == subject_id_raw] if survey_df is not None else None

        raw = load_single_file(file, config)
        if raw is None: continue
        
        raw = prepare_raw(raw, config)
        
        events = get_events_from_raw(raw)
        
        subject_trials = extract_trials(raw, events, config, subject_id_raw, subject_survey_df)
        all_trials.extend(subject_trials)
    
    return all_trials, {}

def get_events_from_raw(raw: mne.io.Raw) -> np.ndarray:
    """Rawオブジェクトからイベント配列を取得するヘルパー"""
    try:
        # アノテーションを優先
        if raw.annotations and len(raw.annotations) > 0:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            return events
        # 次にスティミュラスチャネルを探す
        return mne.find_events(raw, stim_channel='STI 014', min_duration=0.002, verbose=False)
    except (ValueError, RuntimeError):
        # どちらも失敗したらダミーを生成
        return generate_dummy_events(raw)


def extract_trials(raw: mne.io.Raw, events: np.ndarray, config: AppConfig, subject_id: str, survey_df: Optional[pd.DataFrame]) -> List[TrialData]:
    """アンケートデータを考慮してTrialDataを作成"""
    trials = []
    sfreq = raw.info['sfreq']
    
    # 画像1, 2はテストなので除外 (イベントIDが3から始まると仮定)
    valid_events = events[events[:, 2] >= 3] if np.any(events[:, 2] >= 3) else events

    for i, event in enumerate(valid_events):
        trial_id = i + 1 # 1-based index
        event_sample = event[0]
        
        # データ切り出し
        baseline_start = event_sample - int(config.win.baseline_len * sfreq)
        stim_end = event_sample + int(config.win.stim_end * sfreq)
        if baseline_start < 0 or stim_end > len(raw.times): continue
        baseline_data = raw.get_data(start=baseline_start, stop=event_sample)
        stim_data = raw.get_data(start=event_sample, stop=stim_end)
        
        # 嗜好ラベルを決定 (アンケートデータ優先)
        preference = PreferenceLabel.NEUTRAL # デフォルト
        if survey_df is not None and not survey_df.empty:
            trial_survey = survey_df[survey_df['trial_id'] == trial_id]
            if not trial_survey.empty and 'sd_score' in trial_survey.columns:
                # 7段階SD法スコアを「好き」「そうでもない」に変換 (例)
                sd_score = trial_survey['sd_score'].iloc[0]
                if sd_score >= 6:
                    preference = PreferenceLabel.LIKE
        else:
            # アンケートがない場合、イベントIDで仮決定
            preference = PreferenceLabel.LIKE if event[2] % 2 != 0 else PreferenceLabel.NEUTRAL

        trials.append(TrialData(
            subject_id=subject_id,
            trial_id=trial_id,
            preference=preference,
            raw_baseline_data=baseline_data,
            raw_stim_data=stim_data
        ))
    return trials

# ... (load_single_file, prepare_raw, generate_dummy_eventsは前回とほぼ同じでOK) ...

def load_single_file(uploaded_file, config: AppConfig) -> Optional[mne.io.Raw]:
    # (前回と同じコード)
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

def prepare_raw(raw: mne.io.Raw, config: AppConfig) -> mne.io.Raw:
    # (前回と同じコード)
    target_ch = ['FP1', 'FP2']
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

def generate_dummy_events(raw: mne.io.Raw) -> np.ndarray:
    # (前回と同じコード、ただしconfigは不要に)
    trial_duration = 13.0 # 仮定 (ベースライン3秒 + 刺激10秒)
    n_trials = int(raw.times[-1] // trial_duration)
    
    events = []
    for i in range(n_trials):
        event_sample = int((i * trial_duration + 3.0) * raw.info['sfreq'])
        event_id = i + 1 # 1-based index
        events.append([event_sample, 0, event_id])
    
    return np.array(events)
