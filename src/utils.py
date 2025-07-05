"""
CatBond 모델 유틸리티 함수들
로깅, 시간측정, seed 고정 등
"""

import time
import logging
import numpy as np
import pandas as pd
from functools import wraps
from typing import Any, Dict, List, Optional
import yaml
import os

def setup_logging(level: str = "INFO") -> logging.Logger:
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('catbond.log')
        ]
    )
    return logging.getLogger(__name__)

def timer(func):
    """함수 실행 시간 측정 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[TIMER] {func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        return result
    return wrapper

def set_seed(seed: int = 42):
    """랜덤 시드 고정"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    print(f"[INFO] 랜덤 시드 설정: {seed}")

def load_config(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"[INFO] 설정 파일 로드: {config_path}")
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """설정을 YAML 파일로 저장"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"[INFO] 설정 파일 저장: {config_path}")

def ensure_dir(directory: str):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 디렉토리 생성: {directory}")

def find_chungju_region(region_list: List[str]) -> str:
    """충주 지역명 찾기"""
    for region in region_list:
        if '충주' in region:
            return region
    raise ValueError("충주가 포함된 지역명이 없습니다.")

def calculate_region_weights(regions: pd.Series, 
                           chungju_region_name: str,
                           chungju_weight: float = 3.0,
                           major_city_weight: float = 1.5) -> Dict[str, float]:
    """지역별 가중치 계산"""
    major_cities = ['서울', '부산', '대구', '인천', '광주', '대전', '울산']
    
    region_weights = {}
    for region in regions.unique():
        if region == chungju_region_name:
            region_weights[region] = chungju_weight
        elif region in major_cities:
            region_weights[region] = major_city_weight
        else:
            region_weights[region] = 1.0
    
    return region_weights

def print_data_info(df: pd.DataFrame, name: str = "데이터"):
    """데이터 기본 정보 출력"""
    print(f"[INFO] {name} 정보:")
    print(f"  • 형태: {df.shape}")
    print(f"  • 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  • 결측값: {df.isnull().sum().sum()}개")
    if 'date' in df.columns:
        print(f"  • 날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
    if 'region' in df.columns:
        print(f"  • 지역 수: {df['region'].nunique()}개")

def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """필수 컬럼 존재 여부 검증"""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"[ERROR] 필수 컬럼 누락: {missing_columns}")
        return False
    return True

def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                fill_value: float = 0.0) -> np.ndarray:
    """안전한 나눗셈 (0으로 나누기 방지)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result[~np.isfinite(result)] = fill_value
    return result 