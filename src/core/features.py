"""
CatBond 모델 피처 엔지니어링
임계값 계산, 계절성, 지역 임베딩 등
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .utils import timer, safe_divide

class FeatureEngineer:
    """CatBond 모델 피처 엔지니어"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trigger_scaler = None
        self.severity_scaler = None
        self.region_embedding_df = None
        self.region_stats = None
        
    @timer
    def calculate_exposure(self, freq_df: pd.DataFrame) -> pd.DataFrame:
        """지역별 exposure 계산 (지역별 기준연도 적용)"""
        print("[INFO] 지역별 exposure 계산...")
        
        # 지역별 최소연도 계산
        region_min_years = freq_df.groupby('region')['year'].min().to_dict()
        
        # 지역별 exposure 계산
        exposure_growth_rate = self.config['frequency']['exposure_growth_rate']
        exposure_freq = np.ones(len(freq_df))
        
        for i, (_, row) in enumerate(freq_df.iterrows()):
            region = row['region']
            year = row['year']
            min_year = region_min_years[region]
            growth_factor = 1 + (year - min_year) * exposure_growth_rate
            exposure_freq[i] = growth_factor
        
        freq_df['exposure'] = exposure_freq
        
        # 통계 출력
        exposure_stats = freq_df.groupby('region')['exposure'].agg(['min', 'max', 'mean']).round(3)
        print(f"  • 전체 exposure 범위: {exposure_freq.min():.3f} ~ {exposure_freq.max():.3f}")
        
        return freq_df
    
    @timer
    def create_region_embeddings(self, df_triggered: pd.DataFrame) -> pd.DataFrame:
        """지역 임베딩 생성"""
        print("[INFO] 지역 임베딩 생성...")
        
        # 지역별 통계 계산
        self.region_stats = df_triggered.groupby('region').agg({
            'severity_log': ['mean', 'std', 'count'],
            'rain_mm': ['mean', 'std'],
            'pop_density_km2': ['mean'],
            'grdp': ['mean']
        }).round(4)
        
        if self.region_stats.isna().any().any():
            self.region_stats = self.region_stats.fillna(0)
        
        self.region_stats.columns = ['_'.join(col).strip() for col in self.region_stats.columns]
        self.region_stats = self.region_stats.reset_index()
        
        # PCA로 지역 임베딩 생성
        region_features_matrix = self.region_stats[self.region_stats.columns[1:]].values
        
        if np.isnan(region_features_matrix).any():
            region_features_matrix = np.nan_to_num(region_features_matrix, nan=0.0)
        
        n_regions = region_features_matrix.shape[0]
        n_components = min(3, n_regions - 1)
        
        if n_components < 1:
            self.region_embedding_df = pd.DataFrame(
                {'region_embedding_1': self.region_stats['region']},
                index=self.region_stats['region']
            )
        else:
            pca = PCA(n_components=n_components, random_state=42)
            region_embeddings = pca.fit_transform(region_features_matrix)
            
            embedding_cols = [f'region_embedding_{i+1}' for i in range(n_components)]
            self.region_embedding_df = pd.DataFrame(
                region_embeddings, 
                columns=embedding_cols,
                index=self.region_stats['region']
            )
        
        # 데이터프레임에 임베딩 추가
        df_triggered = df_triggered.merge(
            self.region_stats, on='region', how='left', suffixes=('', '_region_stats')
        )
        df_triggered = df_triggered.merge(
            self.region_embedding_df.reset_index(), 
            left_on='region', 
            right_on='region', 
            how='left',
            suffixes=('', '_embedding')
        )
        
        print(f"  • 지역 임베딩 생성 완료: {n_components}개 차원")
        return df_triggered
    
    @timer
    def scale_features(self, X: pd.DataFrame, feature_type: str) -> pd.DataFrame:
        """피처 스케일링"""
        print(f"[INFO] {feature_type} 모델 피처 스케일링...")
        
        if feature_type == 'trigger':
            features = self.config['features']['trigger_features']
            scaler = StandardScaler()
            self.trigger_scaler = scaler
        elif feature_type == 'severity':
            features = self.config['features']['severity_features']
            scaler = StandardScaler()
            self.severity_scaler = scaler
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # 수치형 피처만 선택 (region 제외)
        numeric_features = [col for col in features if col not in ['region']]
        
        X_scaled = X.copy()
        X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
        
        print(f"  • 스케일링 완료: {len(numeric_features)}개 피처")
        return X_scaled
    
    def get_region_weights(self, df_triggered: pd.DataFrame, 
                          chungju_region_name: str) -> pd.Series:
        """지역별 가중치 계산 - 모든 지역에 동일한 가중치 적용"""
        # 모든 지역에 동일한 가중치 적용 (가중치 편향 제거)
        region_weights = {region: 1.0 for region in df_triggered['region'].unique()}
        
        sample_weight = df_triggered['region'].map(region_weights)
        
        print(f"[INFO] 지역별 가중치 적용 (균등 가중치):")
        print(f"  • 모든 지역 가중치: 1.0 (편향 제거)")
        print(f"  • 평균 가중치: {sample_weight.mean():.2f}")
        print(f"  • 가중치 범위: {sample_weight.min():.1f} ~ {sample_weight.max():.1f}")
        
        return sample_weight
    
    def prepare_frequency_features(self, freq_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """빈도 모델용 피처 준비"""
        climate_features = self.config['features']['climate_features']
        
        # Region 더미 생성
        region_dummies_cols = pd.get_dummies(freq_df['region']).columns.tolist()
        X_multi = pd.get_dummies(freq_df['region']).join(freq_df[climate_features])
        
        # 타겟 변환: y = N_y / exposure (rate 모델)
        y_rate = safe_divide(freq_df['N_y'], freq_df['exposure'])
        sample_weights = freq_df['exposure']
        
        print(f"[INFO] 빈도 모델 피처:")
        print(f"  • 기후 특성: {len(climate_features)}개")
        print(f"  • Region 더미: {len(region_dummies_cols)}개")
        print(f"  • 총 특성 수: {X_multi.shape[1]}개")
        
        return X_multi, y_rate, sample_weights
    
    def prepare_trigger_features(self, df_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """트리거 모델용 피처 준비"""
        trigger_features = self.config['features']['trigger_features']
        
        X_trigger = df_full[trigger_features + ["region"]]
        y_trigger = df_full["trigger_flag"]
        
        print(f"[INFO] 트리거 모델 피처:")
        print(f"  • 피처 수: {len(trigger_features)}개")
        print(f"  • 전체 샘플: {len(X_trigger):,}개")
        print(f"  • 트리거 발생: {y_trigger.sum():,}개 ({y_trigger.mean():.2%})")
        
        return X_trigger, y_trigger
    
    def prepare_severity_features(self, df_triggered: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """손실 모델용 피처 준비"""
        severity_features = self.config['features']['severity_features']
        
        # 기본 피처 + 지역 통계 + 임베딩
        cat_cols = ["region"]
        region_feats = [col for col in self.region_stats.columns if col != 'region']
        embedding_feats = list(self.region_embedding_df.columns)
        num_feats_enhanced = severity_features + region_feats + embedding_feats
        
        X_severity = df_triggered[cat_cols + num_feats_enhanced]
        y_severity = df_triggered["severity_log"]
        
        print(f"[INFO] 손실 모델 피처:")
        print(f"  • 기본 피처: {len(severity_features)}개")
        print(f"  • 지역 통계: {len(region_feats)}개")
        print(f"  • 지역 임베딩: {len(embedding_feats)}개")
        print(f"  • 총 피처 수: {len(num_feats_enhanced)}개")
        
        return X_severity, y_severity
    
    def get_feature_names(self, feature_type: str) -> List[str]:
        """피처명 반환"""
        if feature_type == 'trigger':
            return self.config['features']['trigger_features']
        elif feature_type == 'severity':
            return self.config['features']['severity_features']
        elif feature_type == 'climate':
            return self.config['features']['climate_features']
        else:
            raise ValueError(f"Unknown feature type: {feature_type}") 