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

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """データフレームの列名を正規化する（小文字化、スペース・アンダースコア除去）"""
    normalized_cols = {col: col.lower().replace(" ", "").replace("_", "") for col in df.columns}
    df = df.rename(columns=normalized_cols)
    return df

def find_column_by_variants(df: pd.DataFrame, variants: List[str]) -> Optional[str]:
    """列名のバリアントの中から、存在する列名を探す"""
    for variant in variants:
        if variant in df.columns:
            return variant
    return None

def load_survey_data(uploaded_file) -> Optional[pd.DataFrame]:
    """CSVまたはExcel形式のアンケートデータを読み込み、列名を正規化する"""
    try:
        fname = uploaded_file.name
        if fname.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif fname.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("評価データはCSVまたはExcel形式である必要があります。")
            return None
        
        # ★★★ 列名正規化と必須列チェックを追加 ★★★
        df = normalize_column_names(df)

        # subject_id列を探す
        subject_id_col = find_column_by_variants(df, ["subjectid", "subject", "id"])
        if subject_id_col is None:
            st.error("評価データに必須列 'subject_id' (または 'subject', 'id') が見つかりません。")
            return None
        df = df.rename(columns={subject_id_col: 'subject_id'})
        
        # trial_id列を探す
        trial_id_col = find_column_by_variants(df, ["trialid", "imgid", "imageid"])
        if trial_id_col is None:
            st.error("評価データに必須列 'trial_id' (または 'img_id') が見つかりません。")
            return None
        df = df.rename(columns={trial_id_col: 'trial_id'})

        # Dislike_Like列を探す
        like_col = find_column_by_variants(df, ["dislikelike", "sdscore", "preference", "like"])
        if like_col:
            df = df.rename(columns={like_col: 'like_score'})

        df['trial_id'] = pd.to_numeric(df['trial_id'], errors='coerce').dropna().astype(int)
        df['subject_id'] = df['subject_id'].astype(str) # 文字列として扱う

        st.success(f"評価データ読み込み完了 ({len(df)}件)")
        return df
    except Exception as e:
        st.error(f"評価データの読み込みに失敗しました: {e}")
        return None

def load_all_trial_data(uploaded_eeg_files: List, uploaded_survey_file, config: AppConfig) -> Tuple[List[TrialData], dict]:
    survey_df = load_survey_data(uploaded_survey_file) if uploaded_survey_file else None

    all_trials = []
    for i, file in enumerate(uploaded_eeg_files):
        # ファイル名から被験者IDを推測 (例: sub-P001_... -> P001)
        try:
            subject_id = Path(file.name).stem.split('_')[0].split('-')[1]
        except IndexError:
            subject_id = f"S{i+1:02d}"
        
        # ★★★ アンケートデータからこの被験者のデータを安全にフィルタリング ★★★
        subject_survey_df = None
        if survey_df is not None:
            # subject_id列が存在することを確信できる
            subject_survey_df = survey_df[survey_df['subject_id'] == subject_id]
            if subject_survey_df.empty:
                st.warning(f"アンケートデータに被験者 '{subject_id}' のデータが見つかりませんでした。")
        
        # ... (以降のファイル読み込みロジックは同じ) ...
        raw, markers_df = None, None
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.getvalue())
            file_path = tmp_file.name
        try:
            if file.name.lower().endswith('.xdf'):
                raw, markers_df = load_xdf_as_raw(file_path, config)
        finally:
            os.unlink(file_path)
        if raw is None: continue
        events = get_events_from_raw(raw)
        subject_trials = extract_trials(raw, events, config, subject_id, subject_survey_df)
        all_trials.extend(subject_trials)
        
    return all_trials, {}

def extract_trials(raw: mne.io.Raw, events: np.ndarray, config: AppConfig, subject_id: str, survey_df: Optional[pd.DataFrame]) -> List[TrialData]:
    # ... (前回のコードから少し修正) ...
    preference = PreferenceLabel.NEUTRAL
    if survey_df is not None and not survey_df.empty:
        trial_survey = survey_df[survey_df['trial_id'] == trial_id]
        if not trial_survey.empty and 'like_score' in trial_survey.columns:
            score = trial_survey['like_score'].iloc[0]
            if pd.notna(score):
                preference = PreferenceLabel.LIKE if score >= 5 else PreferenceLabel.NEUTRAL
    # ...
    return trials

# ... (その他の関数は前回と同じ) ...
