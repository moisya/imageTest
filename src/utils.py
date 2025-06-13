# src/utils.py
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum

# --- Enum定義 ---
class PreferenceLabel(str, Enum):
    LIKE = "好き"
    NEUTRAL = "そうでもない"

# --- Pydantic設定クラス ---
class FilterConfig(BaseModel):
    l_freq: float = Field(1.0, description="下限周波数 (Hz)")
    h_freq: float = Field(50.0, description="上限周波数 (Hz)")
    notch_freq: float = Field(50.0, description="ノッチフィルタ周波数 (Hz)")
    sfreq: float = Field(250.0, description="サンプリング周波数 (Hz)", mutable=True)

class QCThresholds(BaseModel):
    amp_uV: float = Field(80.0, description="振幅閾値 (µV)")
    diff_uV: float = Field(35.0, description="隣接サンプル差閾値 (µV)")

class WindowConfig(BaseModel):
    baseline_len: float = Field(3.0, description="利用可能なベースライン長 (秒)")
    baseline_samples: int = Field(2, description="ベースラインから抽出する1秒窓の数")
    stim_start: float = Field(0.5, description="刺激提示後の無視区間 (秒)")
    stim_end: float = Field(10.0, description="刺激提示の終了時点 (秒)")
    win_len: float = Field(1.0, description="解析窓長 (秒)")
    stim_samples: int = Field(5, description="刺激区間からランダム抽出する窓の数")

class FrequencyBands(BaseModel):
    theta: Tuple[float, float] = (4.0, 7.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta: Tuple[float, float] = (13.0, 30.0)

class AppConfig(BaseModel):
    filter: FilterConfig = Field(default_factory=FilterConfig)
    qc: QCThresholds = Field(default_factory=QCThresholds)
    win: WindowConfig = Field(default_factory=WindowConfig)
    freq_bands: FrequencyBands = Field(default_factory=FrequencyBands)

# --- Dataclass定義 ---
@dataclass
class QCResult:
    is_valid: bool
    n_clean_windows: int
    total_windows: int
    quality_score: float

@dataclass
class TrialData:
    subject_id: str
    trial_id: int
    preference: PreferenceLabel
    raw_baseline_data: np.ndarray
    raw_stim_data: np.ndarray
    filtered_baseline_data: Optional[np.ndarray] = None
    filtered_stim_data: Optional[np.ndarray] = None
    clean_baseline_data: Optional[np.ndarray] = None
    clean_stim_data: Optional[np.ndarray] = None
    is_valid: bool = False
    qc_info: Dict[str, QCResult] = field(default_factory=dict)
