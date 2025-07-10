#!/usr/bin/env python3
"""
CatBond 임계값 분석 Streamlit 앱
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import joblib
import lightgbm as lgb
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="CatBond 임계값 분석 대시보드",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThresholdAnalysisApp:
    """임계값 분석 Streamlit 앱"""
    
    def __init__(self):
        """초기화"""
        self.config = self._load_config()
        self.models = self._load_models()
        self.data = self._load_data()
        
    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"설정 파일 로드 실패: {e}")
            return {}
    
    def _load_models(self) -> Dict:
        """학습된 모델들 로드"""
        try:
            models = {
                'trigger_model': joblib.load('models/catbond_trigger_model.pkl'),
                'severity_model': lgb.Booster(model_file='models/catbond_severity_model.txt'),
                'region_info': joblib.load('models/catbond_region_info.pkl')
            }
            return models
        except Exception as e:
            st.error(f"모델 로드 실패: {e}")
            return None
    
    def _load_data(self) -> Dict:
        """데이터 로드"""
        try:
            from src.core.data import CatBondDataLoader
            from src.core.features import FeatureEngineer
            
            # 데이터 로더 초기화
            data_loader = CatBondDataLoader(self.config)
            
            # 데이터 로드
            df_rain = data_loader.load_rain_data()
            if df_rain is None or df_rain.empty:
                st.error("강수 데이터 로드에 실패했습니다.")
                return None
                
            df_loss = data_loader.load_loss_data()
            if df_loss is None or df_loss.empty:
                st.error("손실 데이터 로드에 실패했습니다.")
                return None
            
            # 지역별 임계값 계산
            region_thresholds = data_loader.calculate_thresholds(df_rain)
            if not region_thresholds:
                st.error("지역별 임계값 계산에 실패했습니다.")
                return None
            
            # 트리거 데이터 준비 (df_full 생성)
            df_full = data_loader.prepare_trigger_data(df_loss, region_thresholds)
            if df_full is None or df_full.empty:
                st.error("트리거 데이터 준비에 실패했습니다.")
                return None
            
            # 피처 엔지니어 초기화 (config만 전달)
            feature_engineer = FeatureEngineer(self.config)

            # 트리거된 데이터만 필터링
            df_triggered = df_full[df_full['trigger_flag'] == 1].copy()
            if not df_triggered.empty:
                # 반드시 severity, severity_log 생성
                df_triggered = data_loader.prepare_severity_data(df_triggered)
                # 임베딩/통계 생성
                df_triggered = feature_engineer.create_region_embeddings(df_triggered)
            
            # 스케일링된 데이터 준비 (안전하게 처리)
            try:
                X_trigger, y_trigger = feature_engineer.prepare_trigger_features(df_triggered)
                X_trigger_scaled = feature_engineer.scale_features(X_trigger, 'trigger')
            except Exception as e:
                st.warning(f"트리거 피처 준비 중 오류: {e}")
                X_trigger_scaled = pd.DataFrame()
                y_trigger = pd.Series()

            try:
                X_severity, y_severity = feature_engineer.prepare_severity_features(df_triggered)
                X_severity_scaled = feature_engineer.scale_features(X_severity, 'severity')
            except Exception as e:
                st.warning(f"손실 피처 준비 중 오류: {e}")
                X_severity_scaled = pd.DataFrame()
                y_severity = pd.Series()
            
            # 충주 지역 정보
            chungju_region_name = "충청북도 충주시"
            chungju_threshold = 205.0
            
            # 인제군 정보 추가
            inje_region_name = "강원도 인제군"
            inje_threshold = region_thresholds.get(inje_region_name, None)
            if inje_threshold is None:
                st.warning(f"⚠️ 인제군 임계값 정보를 찾을 수 없습니다.")
            else:
                st.info(f"📊 인제군 임계값: {inje_threshold:.1f}mm")
            
            data = {
                'df_rain': df_rain,
                'df_full': df_full,
                'df_triggered': df_triggered,
                'X_trigger_scaled': X_trigger_scaled,
                'X_severity_scaled': X_severity_scaled,
                'feature_engineer': feature_engineer,
                'chungju_region_name': chungju_region_name,
                'chungju_threshold': chungju_threshold,
                'inje_region_name': inje_region_name,
                'inje_threshold': inje_threshold,
                'region_thresholds': region_thresholds,
                'data_loader': data_loader
            }
            
            return data
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            import traceback
            st.error(f"상세 오류: {traceback.format_exc()}")
            return None
    
    def calculate_lambda_by_threshold(self, region_name: str, threshold_range: List[float]) -> pd.DataFrame:
        """임계값별 연간 빈도 계산"""
        try:
            region_rain = self.data['df_rain'][self.data['df_rain']['region'] == region_name].copy()
            
            results = []
            for threshold in threshold_range:
                # 해당 임계값으로 트리거 플래그 계산
                region_rain['trigger_flag'] = (region_rain['rain_mm'] >= threshold).astype(int)
                
                # 연간 사건 수 계산
                yearly_counts = region_rain.groupby('year').apply(
                    lambda x: x['trigger_flag'].sum()
                ).reset_index()
                yearly_counts.columns = ['year', 'event_count']
                
                # 연간 평균 빈도 계산
                lambda_annual = yearly_counts['event_count'].mean()
                
                results.append({
                    'threshold': threshold,
                    'lambda_annual': lambda_annual
                })
            
            return pd.DataFrame(results)
        except Exception as e:
            st.error(f"λ 계산 실패: {e}")
            return pd.DataFrame()
    
    def calculate_loss_rate_by_threshold(self, region_name: str, threshold_range: List[float]) -> pd.DataFrame:
        """임계값별 손실률 계산"""
        try:
            results = []
            
            for threshold in threshold_range:
                # 해당 임계값으로 트리거 플래그 재계산
                df_full_temp = self.data['df_full'].copy()
                df_full_temp['trigger_flag'] = (df_full_temp['rain_mm'] >= threshold).astype(int)
                
                # 트리거된 데이터 필터링
                df_triggered_temp = df_full_temp[df_full_temp['trigger_flag'] == 1].copy()
                
                # 반드시 prepare_severity_data를 거쳐야 severity 컬럼이 생성됨
                df_triggered_temp = self.data['data_loader'].prepare_severity_data(df_triggered_temp)
                
                # 해당 지역의 트리거된 데이터만 필터링
                region_triggered = df_triggered_temp[df_triggered_temp['region'] == region_name]
                
                if len(region_triggered) > 0:
                    # 손실률 계산
                    loss_rate = region_triggered['severity'].mean()
                    
                    # 신뢰도 계산 (표본 크기 기반)
                    sample_size = len(region_triggered)
                    if sample_size >= 50:
                        confidence = 'high'
                    elif sample_size >= 20:
                        confidence = 'medium'
                    else:
                        confidence = 'low'
                else:
                    loss_rate = 0.0
                    confidence = 'low'
                
                results.append({
                    'threshold': threshold,
                    'loss_rate': loss_rate,
                    'confidence': confidence,
                    'sample_size': len(region_triggered)
                })
            
            return pd.DataFrame(results)
        except Exception as e:
            st.error(f"손실률 계산 실패: {e}")
            return pd.DataFrame()
    
    def calculate_expected_loss_by_threshold(self, region_name: str, threshold_range: List[float]) -> pd.DataFrame:
        """임계값별 기대손실 계산"""
        try:
            # λ 계산
            lambda_df = self.calculate_lambda_by_threshold(region_name, threshold_range)
            
            # L 계산
            loss_df = self.calculate_loss_rate_by_threshold(region_name, threshold_range)
            
            if lambda_df.empty or loss_df.empty:
                return pd.DataFrame()
            
            # 결과 병합
            results_df = pd.merge(lambda_df, loss_df, on='threshold')
            
            # 기대손실 계산: EL = F × λ × n × L
            face_value = self.config['model']['face_value']
            years = self.config['expected_loss']['years']
            
            results_df['expected_loss'] = (
                face_value * 
                results_df['lambda_annual'] * 
                years * 
                results_df['loss_rate']
            )
            
            # 발생 확률 계산 (3년간 최소 1회)
            results_df['event_probability'] = 1 - np.exp(-results_df['lambda_annual'] * years)
            
            return results_df
        except Exception as e:
            st.error(f"기대손실 계산 실패: {e}")
            return pd.DataFrame()
    
    def analyze_issuer_perspective(self, region_name: str, threshold_range: List[float], coupon_rate: float) -> pd.DataFrame:
        """발행자 관점 분석 (쿠폰율 기반)"""
        try:
            # 기대손실 계산
            results_df = self.calculate_expected_loss_by_threshold(region_name, threshold_range)
            
            if results_df.empty:
                return pd.DataFrame()
            
            # 발행자 관점 지표 계산
            face_value = self.config['model']['face_value']
            years = self.config['expected_loss']['years']
            
            # 무위험 이자율 설정 (2.5%)
            risk_free_rate = 0.025
            
            # 쿠폰 수입 (모든 임계값에서 동일)
            coupon_income = face_value * coupon_rate * years
            
            # 발행자 순이익
            results_df['coupon_income'] = coupon_income
            results_df['net_profit'] = coupon_income - results_df['expected_loss']
            results_df['profit_margin'] = results_df['net_profit'] / coupon_income * 100
            
            # 리스크 프리미엄 (쿠폰율 - 무위험 이자율)
            results_df['risk_premium'] = (coupon_rate - risk_free_rate) * 100
            
            # 위험 대비 수익률
            results_df['risk_adjusted_return'] = results_df['net_profit'] / (results_df['expected_loss'] + 1e-8)
            
            # Break-Even Coupon Rate 계산
            # 기대손실을 보전하려면 필요한 최소 쿠폰율
            results_df['break_even_coupon_rate'] = (results_df['expected_loss'] / face_value / years) * 100
            
            # 발행자 관점 최적화 지표 계산
            # 1. 수익성 지표 (쿠폰율 대비 순이익 비율)
            results_df['profit_to_coupon_ratio'] = results_df['net_profit'] / (coupon_income + 1e-8) * 100
            
            # 2. 위험 지표 (쿠폰율 대비 기대손실 비율)
            results_df['risk_to_coupon_ratio'] = results_df['expected_loss'] / (coupon_income + 1e-8) * 100
            
            # 3. 쿠폰율 기반 트리거 빈도 선호도
            # 쿠폰율이 높을수록 더 자주 트리거되어도 수익성이 좋으므로 낮은 임계값 선호
            # 쿠폰율이 낮을수록 트리거 빈도를 줄여서 위험을 최소화하려고 함
            
            # 쿠폰율 기반 트리거 빈도 가중치
            # 높은 쿠폰율: 낮은 임계값 선호 (더 자주 트리거)
            # 낮은 쿠폰율: 높은 임계값 선호 (트리거 빈도 감소)
            coupon_frequency_weight = coupon_rate * 5  # 쿠폰율에 따른 빈도 선호도
            
            # 임계값에 대한 빈도 기반 조정
            # 낮은 임계값 = 높은 트리거 빈도
            # 높은 임계값 = 낮은 트리거 빈도
            frequency_adjustment = (results_df['lambda_annual'] * coupon_frequency_weight) * 1e8
            
            # 4. 발행자 관점 종합 점수 (수익성 + 빈도 선호도)
            # 수익성: 높을수록 좋음
            # 빈도 선호도: 쿠폰율에 따라 조정
            results_df['issuer_score'] = (
                results_df['profit_to_coupon_ratio'] +  # 수익성 (높을수록 좋음)
                frequency_adjustment  # 빈도 선호도 (쿠폰율에 따라 조정)
            )
            
            # 5. 쿠폰율 대비 위험 조정 수익률
            # 위험 대비 수익률에 쿠폰율 가중치 적용
            results_df['risk_adjusted_profit'] = (
                results_df['net_profit'] * (1 + coupon_rate) / (results_df['expected_loss'] + 1e-8)
            )
            
            return results_df
        except Exception as e:
            st.error(f"발행자 관점 분석 실패: {e}")
            return pd.DataFrame()
    
    def _find_closest_threshold(self, df: pd.DataFrame, target_threshold: float) -> pd.Series:
        """데이터프레임에서 가장 가까운 임계값의 행을 찾는 헬퍼 함수"""
        if df.empty:
            return pd.Series()
        
        # 절댓값 차이 계산
        df['threshold_diff'] = np.abs(df['threshold'] - target_threshold)
        
        # 가장 가까운 임계값의 인덱스 찾기
        closest_idx = df['threshold_diff'].idxmin()
        
        # 임시 컬럼 제거
        result = df.loc[closest_idx].drop('threshold_diff')
        
        return result
    
    def calculate_break_even_coupon_rate(self, region_name: str, target_threshold: float) -> Dict:
        """특정 임계값에서의 Break-Even 쿠폰율 계산 (Loading Factor 방식)"""
        try:
            # 해당 임계값에서의 기대손실 계산
            loss_result = self.calculate_expected_loss_by_threshold(region_name, [target_threshold])
            
            if loss_result.empty:
                return {}
            
            expected_loss = loss_result.iloc[0]['expected_loss']
            lambda_annual = loss_result.iloc[0]['lambda_annual']
            loss_rate = loss_result.iloc[0]['loss_rate']
            face_value = self.config['model']['face_value']
            years = self.config['expected_loss']['years']
            
            # 무위험 이자율 (2.5%)
            risk_free_rate = 0.025 * 100
            
            # Loading Factor (α) - 2.0으로 고정
            alpha = 2.0
            
            # 기대손실률 계산 (연간)
            expected_loss_rate = (expected_loss / face_value / years) * 100
            
            # 새로운 쿠폰율 계산: 무위험 이자율 + α × EL
            required_coupon_rate = risk_free_rate + (alpha * expected_loss_rate)
            
            # 기본 Break-Even 쿠폰율 (기존 방식 - 참고용)
            break_even_rate = expected_loss_rate
            
            # 리스크 프리미엄 (새로운 방식)
            risk_premium = alpha * expected_loss_rate
            
            # 위험 수준 평가 (Loading Factor 기준)
            if required_coupon_rate <= risk_free_rate + 1:
                risk_level = "매우 낮음"
                risk_description = "무위험 이자율 + 최소 리스크 프리미엄"
            elif required_coupon_rate <= risk_free_rate + 3:
                risk_level = "낮음"
                risk_description = "낮은 리스크 프리미엄 필요"
            elif required_coupon_rate <= risk_free_rate + 7:
                risk_level = "보통"
                risk_description = "적정 수준의 리스크 프리미엄"
            elif required_coupon_rate <= risk_free_rate + 12:
                risk_level = "높음"
                risk_description = "높은 리스크 프리미엄 필요"
            else:
                risk_level = "매우 높음"
                risk_description = "매우 높은 리스크 프리미엄 필요"
            
            return {
                'target_threshold': target_threshold,
                'expected_loss': expected_loss,
                'lambda_annual': lambda_annual,
                'loss_rate': loss_rate,
                'break_even_rate': break_even_rate,
                'alpha': alpha,
                'expected_loss_rate': expected_loss_rate,
                'risk_premium': risk_premium,
                'risk_free_rate': risk_free_rate,
                'required_coupon_rate': required_coupon_rate,
                'risk_level': risk_level,
                'risk_description': risk_description
            }
        except Exception as e:
            st.error(f"Break-Even 쿠폰율 계산 실패: {e}")
            return {}
    
    def _get_threshold_percentile(self, threshold: float, region_name: str) -> float:
        """임계값이 해당 지역 강수 분포에서 차지하는 백분위 계산"""
        try:
            region_rain = self.data['df_rain'][self.data['df_rain']['region'] == region_name]
            
            if region_rain.empty:
                return 50.0  # 기본값
            
            # 임계값이 해당 지역 강수 분포에서 어느 위치에 있는지 계산
            rain_percentile = (region_rain['rain_mm'] >= threshold).mean() * 100
            return rain_percentile
        except:
            return 50.0  # 기본값
    
    def _calculate_threshold_risk_factor(self, threshold: float, region_name: str) -> float:
        """임계값 기반 리스크 팩터 계산"""
        try:
            # 해당 지역의 강수 데이터에서 임계값 분포 분석
            region_rain = self.data['df_rain'][self.data['df_rain']['region'] == region_name]
            
            if region_rain.empty:
                return 0.0
            
            # 임계값이 해당 지역 강수 분포에서 어느 위치에 있는지 계산
            rain_percentile = (region_rain['rain_mm'] >= threshold).mean() * 100
            
            # 낮은 임계값 (높은 백분위) = 높은 위험 = 높은 리스크 팩터
            # 높은 임계값 (낮은 백분위) = 낮은 위험 = 낮은 리스크 팩터
            if rain_percentile >= 80:  # 매우 낮은 임계값
                return 2.0
            elif rain_percentile >= 60:  # 낮은 임계값
                return 1.5
            elif rain_percentile >= 40:  # 보통 임계값
                return 1.0
            elif rain_percentile >= 20:  # 높은 임계값
                return 0.5
            else:  # 매우 높은 임계값
                return 0.2
        except:
            return 0.0
    
    def _calculate_frequency_risk_factor(self, lambda_annual: float) -> float:
        """빈도 기반 리스크 팩터 계산"""
        # 높은 빈도 = 높은 위험 = 높은 리스크 팩터
        if lambda_annual >= 2.0:  # 연간 2회 이상
            return 1.5
        elif lambda_annual >= 1.0:  # 연간 1회 이상
            return 1.0
        elif lambda_annual >= 0.5:  # 2년에 1회
            return 0.7
        elif lambda_annual >= 0.2:  # 5년에 1회
            return 0.4
        else:  # 10년에 1회 미만
            return 0.2
    
    def _calculate_severity_risk_factor(self, loss_rate: float) -> float:
        """손실률 기반 리스크 팩터 계산"""
        # 높은 손실률 = 높은 위험 = 높은 리스크 팩터
        if loss_rate >= 0.1:  # 10% 이상
            return 2.0
        elif loss_rate >= 0.05:  # 5% 이상
            return 1.5
        elif loss_rate >= 0.02:  # 2% 이상
            return 1.0
        elif loss_rate >= 0.01:  # 1% 이상
            return 0.7
        else:  # 1% 미만
            return 0.4
    
    def analyze_threshold_coupon_relationship(self, region_name: str, threshold_range: List[float], coupon_range: List[float]) -> pd.DataFrame:
        """임계값과 쿠폰율 간의 관계 분석"""
        try:
            results = []
            
            for threshold in threshold_range:
                for coupon_rate in coupon_range:
                    # 해당 조합에서의 발행자 관점 분석
                    issuer_result = self.analyze_issuer_perspective(region_name, [threshold], coupon_rate)
                    
                    if not issuer_result.empty:
                        row = issuer_result.iloc[0].copy()
                        row['coupon_rate_pct'] = coupon_rate * 100
                        results.append(row)
            
            return pd.DataFrame(results)
        except Exception as e:
            st.error(f"임계값-쿠폰율 관계 분석 실패: {e}")
            return pd.DataFrame()
    
    def find_optimal_threshold_coupon_combinations(self, region_name: str, threshold_range: List[float], coupon_range: List[float]) -> pd.DataFrame:
        """임계값과 쿠폰율의 최적 조합 찾기"""
        try:
            # 모든 조합 분석
            relationship_df = self.analyze_threshold_coupon_relationship(region_name, threshold_range, coupon_range)
            
            if relationship_df.empty:
                return pd.DataFrame()
            
            # 투자자 관점 최적화 (기대손실 최소화)
            min_loss_idx = relationship_df['expected_loss'].idxmin()
            investor_optimal = relationship_df.loc[min_loss_idx]
            
            # 발행자 관점 최적화 (발행자 점수 최대화)
            max_score_idx = relationship_df['issuer_score'].idxmax()
            issuer_optimal = relationship_df.loc[max_score_idx]
            
            # 위험 대비 수익률 최적화
            max_risk_adj_idx = relationship_df['risk_adjusted_profit'].idxmax()
            risk_optimal = relationship_df.loc[max_risk_adj_idx]
            
            # 결과 정리
            optimal_combinations = pd.DataFrame({
                '최적화 관점': ['투자자 관점', '발행자 관점', '위험대비수익률'],
                '임계값(mm)': [investor_optimal['threshold'], issuer_optimal['threshold'], risk_optimal['threshold']],
                '쿠폰율(%)': [investor_optimal['coupon_rate_pct'], issuer_optimal['coupon_rate_pct'], risk_optimal['coupon_rate_pct']],
                '기대손실(억원)': [investor_optimal['expected_loss']/1e8, issuer_optimal['expected_loss']/1e8, risk_optimal['expected_loss']/1e8],
                '순이익(억원)': [investor_optimal['net_profit']/1e8, issuer_optimal['net_profit']/1e8, risk_optimal['net_profit']/1e8],
                '수익률마진(%)': [investor_optimal['profit_margin'], issuer_optimal['profit_margin'], risk_optimal['profit_margin']],
                '발행자점수': [investor_optimal['issuer_score'], issuer_optimal['issuer_score'], risk_optimal['issuer_score']]
            })
            
            return optimal_combinations
        except Exception as e:
            st.error(f"최적 조합 찾기 실패: {e}")
            return pd.DataFrame()
    
    def calculate_model_predicted_loss_by_threshold(self, region_name: str, threshold_range: List[float]) -> pd.DataFrame:
        """임계값별 모델 기반 기대손실 계산 (Two-Part Model)"""
        try:
            # 지역 검증
            if self.data['feature_engineer'].region_stats is not None:
                trained_regions = self.data['feature_engineer'].region_stats['region'].unique()
                if region_name not in trained_regions:
                    st.error(f"❌ '{region_name}'은 모델 학습에 사용되지 않은 지역입니다. 학습된 지역만 사용 가능합니다.")
                    return pd.DataFrame()
            
            results = []
            # 필요한 객체
            trigger_model = self.models['trigger_model']
            severity_model = self.models['severity_model']
            feature_engineer = self.data['feature_engineer']
            data_loader = self.data['data_loader']
            df_full_orig = self.data['df_full']
            face_value = self.config['model']['face_value']
            years = self.config['expected_loss']['years']

            for threshold in threshold_range:
                # 1. 임계값 적용
                df_full = df_full_orig.copy()
                df_full['trigger_flag'] = (df_full['rain_mm'] >= threshold).astype(int)
                df_triggered = df_full[df_full['trigger_flag'] == 1].copy()
                if df_triggered.empty:
                    results.append({
                        'threshold': threshold,
                        'lambda_annual_pred': 0.0,
                        'loss_rate_pred': 0.0,
                        'expected_loss_pred': 0.0,
                        'event_probability_pred': 0.0,
                        'sample_size_pred': 0
                    })
                    continue
                # 2. severity, severity_log 생성 및 임베딩
                df_triggered = data_loader.prepare_severity_data(df_triggered)
                df_triggered = feature_engineer.create_region_embeddings(df_triggered)
                # 3. 트리거 피처 준비 및 스케일링 (전체)
                X_trigger, _ = feature_engineer.prepare_trigger_features(df_full)
                X_trigger_scaled = feature_engineer.scale_features(X_trigger, 'trigger')
                # 4. 손실 피처 준비 및 스케일링 (트리거 발생)
                X_severity, _ = feature_engineer.prepare_severity_features(df_triggered)
                X_severity_scaled = feature_engineer.scale_features(X_severity, 'severity')
                # 5. 모델 예측
                from numpy import expm1
                trigger_proba = trigger_model.predict_proba(X_trigger_scaled)[:, 1]
                severity_pred_log = severity_model.predict(X_severity_scaled)
                severity_pred_ratio = expm1(severity_pred_log)
                # 6. 전체 기간에 맞는 배열 생성
                N = len(df_full)
                severity_pred_full = np.zeros(N)
                trigger_idx = df_full.index[df_full['trigger_flag'] == 1].to_numpy()
                severity_pred_full[trigger_idx] = severity_pred_ratio
                # 7. 해당 지역 마스크
                region_mask_full = (df_full['region'] == region_name)
                region_mask_triggered = (df_triggered['region'] == region_name)
                # 8. λ (예측): 해당 지역에서의 연간 트리거 확률 합의 평균
                df_full_region = df_full[region_mask_full].copy()
                # year 컬럼이 없으면 date에서 추출
                if 'year' not in df_full_region.columns:
                    df_full_region['year'] = df_full_region['date'].dt.year
                years_unique = df_full_region['year'].nunique()
                lambda_annual_pred = trigger_proba[region_mask_full].sum() / years_unique if years_unique > 0 else 0.0
                # 9. L (예측): 해당 지역 트리거 발생 샘플의 평균 예측 손실률
                loss_rate_pred = severity_pred_ratio[region_mask_triggered].mean() if region_mask_triggered.sum() > 0 else 0.0
                # 10. 기대손실 (예측)
                expected_loss_pred = face_value * lambda_annual_pred * years * loss_rate_pred
                # 11. 발생확률 (예측)
                event_probability_pred = 1 - np.exp(-lambda_annual_pred * years)
                # 12. 표본 크기
                sample_size_pred = region_mask_triggered.sum()
                results.append({
                    'threshold': threshold,
                    'lambda_annual_pred': lambda_annual_pred,
                    'loss_rate_pred': loss_rate_pred,
                    'expected_loss_pred': expected_loss_pred,
                    'event_probability_pred': event_probability_pred,
                    'sample_size_pred': sample_size_pred
                })
            return pd.DataFrame(results)
        except Exception as e:
            st.error(f"모델 기반 기대손실 계산 실패: {e}")
            return pd.DataFrame()
    
    def _run_original_analysis(self, region_name: str, threshold_min: float, threshold_max: float, threshold_steps: int, coupon_rate: float):
        """기존 분석 (쿠폰율 → 임계값) 실행"""
        threshold_range = np.linspace(threshold_min, threshold_max, threshold_steps)
        st.subheader("📊 투자자 관점 분석")
        # Toggle for model-based results
        show_model_pred = st.checkbox("모델 기반 예측 결과도 함께 보기", value=True)
        investor_results = self.calculate_expected_loss_by_threshold(region_name, threshold_range)
        if show_model_pred:
            model_pred_results = self.calculate_model_predicted_loss_by_threshold(region_name, threshold_range)
        else:
            model_pred_results = None
        if not investor_results.empty:
            current_threshold = self.data['chungju_threshold']
            current_result = self._find_closest_threshold(investor_results, current_threshold)
            min_loss_idx = investor_results['expected_loss'].idxmin()
            optimal_threshold = investor_results.loc[min_loss_idx, 'threshold']
            optimal_loss = investor_results.loc[min_loss_idx, 'expected_loss']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("현재 임계값", f"{current_threshold:.1f}mm", help="현재 설정된 임계값")
            with col2:
                if not current_result.empty:
                    st.metric("현재 기대손실", f"{current_result['expected_loss']/1e8:.2f}억원", help="현재 임계값에서의 기대손실")
            with col3:
                st.metric("최적 임계값", f"{optimal_threshold:.1f}mm", help="기대손실 최소화 임계값")
            # 투자자 관점 그래프
            fig_investor = make_subplots(
                rows=2, cols=2,
                subplot_titles=('연간 빈도 (λ)', '손실률 (L)', '기대손실', '발생 확률'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            # 연간 빈도
            fig_investor.add_trace(
                go.Scatter(x=investor_results['threshold'], y=investor_results['lambda_annual'],
                          mode='lines+markers', name='λ(실측)', line=dict(color='blue')),
                row=1, col=1
            )
            if show_model_pred and model_pred_results is not None and not model_pred_results.empty:
                fig_investor.add_trace(
                    go.Scatter(x=model_pred_results['threshold'], y=model_pred_results['lambda_annual_pred'],
                               mode='lines+markers', name='λ(모델)', line=dict(color='blue', dash='dot')),
                    row=1, col=1
                )
            fig_investor.add_vline(x=current_threshold, line_dash="dash", line_color="red", annotation_text="현재", row=1, col=1)
            fig_investor.add_vline(x=optimal_threshold, line_dash="dash", line_color="green", annotation_text="최적", row=1, col=1)
            # 손실률
            fig_investor.add_trace(
                go.Scatter(x=investor_results['threshold'], y=investor_results['loss_rate'],
                          mode='lines+markers', name='L(실측)', line=dict(color='orange')),
                row=1, col=2
            )
            if show_model_pred and model_pred_results is not None and not model_pred_results.empty:
                fig_investor.add_trace(
                    go.Scatter(x=model_pred_results['threshold'], y=model_pred_results['loss_rate_pred'],
                               mode='lines+markers', name='L(모델)', line=dict(color='orange', dash='dot')),
                    row=1, col=2
                )
            fig_investor.add_vline(x=current_threshold, line_dash="dash", line_color="red", row=1, col=2)
            fig_investor.add_vline(x=optimal_threshold, line_dash="dash", line_color="green", row=1, col=2)
            # 기대손실
            fig_investor.add_trace(
                go.Scatter(x=investor_results['threshold'], y=investor_results['expected_loss']/1e8,
                          mode='lines+markers', name='기대손실(실측)', line=dict(color='red')),
                row=2, col=1
            )
            if show_model_pred and model_pred_results is not None and not model_pred_results.empty:
                fig_investor.add_trace(
                    go.Scatter(x=model_pred_results['threshold'], y=model_pred_results['expected_loss_pred']/1e8,
                               mode='lines+markers', name='기대손실(모델)', line=dict(color='red', dash='dot')),
                    row=2, col=1
                )
            fig_investor.add_vline(x=current_threshold, line_dash="dash", line_color="red", row=2, col=1)
            fig_investor.add_vline(x=optimal_threshold, line_dash="dash", line_color="green", row=2, col=1)
            # 발생 확률
            fig_investor.add_trace(
                go.Scatter(x=investor_results['threshold'], y=investor_results['event_probability'],
                          mode='lines+markers', name='발생확률(실측)', line=dict(color='purple')),
                row=2, col=2
            )
            if show_model_pred and model_pred_results is not None and not model_pred_results.empty:
                fig_investor.add_trace(
                    go.Scatter(x=model_pred_results['threshold'], y=model_pred_results['event_probability_pred'],
                               mode='lines+markers', name='발생확률(모델)', line=dict(color='purple', dash='dot')),
                    row=2, col=2
                )
            fig_investor.add_vline(x=current_threshold, line_dash="dash", line_color="red", row=2, col=2)
            fig_investor.add_vline(x=optimal_threshold, line_dash="dash", line_color="green", row=2, col=2)
            fig_investor.update_layout(height=600, title_text="투자자 관점 분석 결과")
            st.plotly_chart(fig_investor, use_container_width=True)
        
        # 발행자 관점 분석
        st.subheader("🏦 발행자 관점 분석")
        
        issuer_results = self.analyze_issuer_perspective(region_name, threshold_range, coupon_rate)
        
        if not issuer_results.empty:
            # 발행자 관점 최적값 찾기 (쿠폰율 기반 조정된 순이익 사용)
            max_profit_idx = issuer_results['issuer_score'].idxmax()
            optimal_profit_threshold = issuer_results.loc[max_profit_idx, 'threshold']
            optimal_profit = issuer_results.loc[max_profit_idx, 'net_profit']
            optimal_adjusted_profit = issuer_results.loc[max_profit_idx, 'risk_adjusted_profit']
            
            # 현재 발행자 결과
            current_issuer_result = self._find_closest_threshold(issuer_results, current_threshold)
            
            # 결과 표시
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "쿠폰 수입",
                    f"{issuer_results['coupon_income'].iloc[0]/1e8:.1f}억원",
                    help="총 쿠폰 수입"
                )
            
            with col2:
                if not current_issuer_result.empty:
                    st.metric(
                        "현재 순이익",
                        f"{current_issuer_result['net_profit']/1e8:.2f}억원",
                        help="현재 임계값에서의 순이익"
                    )
            
            with col3:
                st.metric(
                    "최적 순이익",
                    f"{optimal_profit/1e8:.2f}억원",
                    help="순이익 최대화 임계값에서의 순이익"
                )
            
            with col4:
                st.metric(
                    "최적 임계값",
                    f"{optimal_profit_threshold:.1f}mm",
                    help="순이익 최대화 임계값"
                )
            
            # 추가 지표 표시
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.metric(
                    "리스크 프리미엄",
                    f"{issuer_results['risk_premium'].iloc[0]:.1f}%",
                    help="쿠폰율 - 무위험 이자율"
                )
            
            with col6:
                if not current_issuer_result.empty:
                    st.metric(
                        "현재 Break-Even 쿠폰율",
                        f"{current_issuer_result['break_even_coupon_rate']:.2f}%",
                        help="기대손실을 보전하는 최소 쿠폰율"
                    )
            
            with col7:
                optimal_break_even = issuer_results.loc[max_profit_idx, 'break_even_coupon_rate']
                st.metric(
                    "최적 Break-Even 쿠폰율",
                    f"{optimal_break_even:.2f}%",
                    help="최적 임계값에서의 Break-Even 쿠폰율"
                )
            
            # 발행자 관점 그래프
            fig_issuer = make_subplots(
                rows=2, cols=2,
                subplot_titles=('위험조정 수익률', '수익률 마진', 'Break-Even 쿠폰율', '발행자 종합점수'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 위험조정 수익률
            fig_issuer.add_trace(
                go.Scatter(x=issuer_results['threshold'], y=issuer_results['risk_adjusted_profit']/1e8,
                          mode='lines+markers', name='위험조정수익률', line=dict(color='green')),
                row=1, col=1
            )
            fig_issuer.add_vline(x=current_threshold, line_dash="dash", line_color="red", 
                                annotation_text="현재", row=1, col=1)
            fig_issuer.add_vline(x=optimal_profit_threshold, line_dash="dash", line_color="green", 
                                annotation_text="최적", row=1, col=1)
            
            # 수익률 마진
            fig_issuer.add_trace(
                go.Scatter(x=issuer_results['threshold'], y=issuer_results['profit_margin'],
                          mode='lines+markers', name='수익률마진', line=dict(color='orange')),
                row=1, col=2
            )
            fig_issuer.add_hline(y=20, line_dash="dash", line_color="red", 
                                annotation_text="20% 마진", row=1, col=2)
            fig_issuer.add_vline(x=current_threshold, line_dash="dash", line_color="red", row=1, col=2)
            fig_issuer.add_vline(x=optimal_profit_threshold, line_dash="dash", line_color="green", row=1, col=2)
            
            # Break-Even 쿠폰율
            fig_issuer.add_trace(
                go.Scatter(x=issuer_results['threshold'], y=issuer_results['break_even_coupon_rate'],
                          mode='lines+markers', name='Break-Even 쿠폰율', line=dict(color='purple')),
                row=2, col=1
            )
            fig_issuer.add_hline(y=coupon_rate*100, line_dash="dash", line_color="red", 
                                annotation_text=f"현재 쿠폰율 ({coupon_rate*100:.1f}%)", row=2, col=1)
            fig_issuer.add_vline(x=current_threshold, line_dash="dash", line_color="red", row=2, col=1)
            fig_issuer.add_vline(x=optimal_profit_threshold, line_dash="dash", line_color="green", row=2, col=1)
            
            # 발행자 종합점수
            fig_issuer.add_trace(
                go.Scatter(x=issuer_results['threshold'], y=issuer_results['issuer_score'],
                          mode='lines+markers', name='발행자종합점수', line=dict(color='blue')),
                row=2, col=2
            )
            fig_issuer.add_vline(x=current_threshold, line_dash="dash", line_color="red", row=2, col=2)
            fig_issuer.add_vline(x=optimal_profit_threshold, line_dash="dash", line_color="green", row=2, col=2)
            
            fig_issuer.update_layout(height=600, title_text="발행자 관점 분석 결과")
            st.plotly_chart(fig_issuer, use_container_width=True)
        
        # 상세 데이터 표시
        st.subheader("📋 상세 분석 데이터")
        
        if not investor_results.empty:
            # 투자자 관점 데이터
            st.write("**투자자 관점 데이터**")
            investor_display = investor_results[['threshold', 'lambda_annual', 'loss_rate', 'expected_loss', 'event_probability']].copy()
            investor_display['expected_loss'] = investor_display['expected_loss'] / 1e8
            investor_display.columns = ['임계값(mm)', '연간빈도(λ)', '손실률(L)', '기대손실(억원)', '발생확률']
            # 숫자 컬럼을 float으로 변환
            for col in ['임계값(mm)', '연간빈도(λ)', '손실률(L)', '기대손실(억원)', '발생확률']:
                investor_display[col] = pd.to_numeric(investor_display[col], errors='coerce')
            st.dataframe(investor_display.round(4), use_container_width=True)
        
        if not issuer_results.empty:
            # 발행자 관점 데이터
            st.write("**발행자 관점 데이터**")
            issuer_display = issuer_results[['threshold', 'expected_loss', 'net_profit', 'risk_adjusted_profit', 'profit_margin', 'profit_to_coupon_ratio', 'issuer_score', 'break_even_coupon_rate']].copy()
            issuer_display['expected_loss'] = issuer_display['expected_loss'] / 1e8
            issuer_display['net_profit'] = issuer_display['net_profit'] / 1e8
            issuer_display['risk_adjusted_profit'] = issuer_display['risk_adjusted_profit'] / 1e8
            issuer_display.columns = ['임계값(mm)', '기대손실(억원)', '순이익(억원)', '위험조정수익률(억원)', '수익률마진(%)', '수익성비율(%)', '발행자종합점수', 'Break-Even쿠폰율(%)']
            # 숫자 컬럼을 float으로 변환
            for col in ['임계값(mm)', '기대손실(억원)', '순이익(억원)', '위험조정수익률(억원)', '수익률마진(%)', '수익성비율(%)', '발행자종합점수', 'Break-Even쿠폰율(%)']:
                issuer_display[col] = pd.to_numeric(issuer_display[col], errors='coerce')
            st.dataframe(issuer_display.round(4), use_container_width=True)
    
    def _run_reverse_analysis(self, region_name: str, target_threshold: float):
        """역산 분석 (임계값 → 쿠폰율) 실행"""
        st.subheader("🔄 역산 분석: 임계값 → Break-Even 쿠폰율")
        
        # Break-Even 쿠폰율 계산
        result = self.calculate_break_even_coupon_rate(region_name, target_threshold)
        
        if result:
            # 결과 표시
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "목표 임계값",
                    f"{result['target_threshold']:.1f}mm",
                    help="설정한 목표 임계값"
                )
            
            with col2:
                st.metric(
                    "기대손실",
                    f"{result['expected_loss']/1e8:.2f}억원",
                    help="해당 임계값에서의 기대손실"
                )
            
            with col3:
                st.metric(
                    "Break-Even 쿠폰율",
                    f"{result['break_even_rate']:.2f}%",
                    help="기대손실을 보전하는 최소 쿠폰율"
                )
            
            with col4:
                st.metric(
                    "리스크 프리미엄 (α×EL)",
                    f"{result['risk_premium']:.2f}%",
                    help="Loading Factor × 기대손실률"
                )
            
            # 추가 지표
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.metric(
                    "무위험 이자율",
                    f"{result['risk_free_rate']:.1f}%",
                    help="기준 무위험 이자율"
                )
            
            with col6:
                st.metric(
                    "필요 쿠폰율",
                    f"{result['required_coupon_rate']:.2f}%",
                    help="무위험 이자율 + α × EL"
                )
            
            with col7:
                # 위험 수준 표시
                if result['risk_level'] == "매우 낮음":
                    st.success(f"위험 수준: {result['risk_level']}")
                elif result['risk_level'] == "낮음":
                    st.success(f"위험 수준: {result['risk_level']}")
                elif result['risk_level'] == "보통":
                    st.warning(f"위험 수준: {result['risk_level']}")
                elif result['risk_level'] == "높음":
                    st.error(f"위험 수준: {result['risk_level']}")
                else:  # 매우 높음
                    st.error(f"위험 수준: {result['risk_level']}")
                
                st.caption(result['risk_description'])
            
            # 분석 결과 시각화
            st.subheader("📊 Break-Even 분석 결과")
            
            # 쿠폰율 구성 분석
            fig_reverse = make_subplots(
                rows=1, cols=2,
                subplot_titles=('쿠폰율 구성', '손익 분석'),
                specs=[[{"type": "pie"}, {"type": "bar"}]]
            )
            
            # 쿠폰율 구성 파이 차트 (Loading Factor 방식)
            coupon_components = {
                '무위험 이자율': result['risk_free_rate'],
                '리스크 프리미엄 (α×EL)': result['risk_premium']
            }
            
            fig_reverse.add_trace(
                go.Pie(
                    labels=list(coupon_components.keys()),
                    values=list(coupon_components.values()),
                    hole=0.4,
                    marker_colors=['lightblue', 'orange', 'red', 'green', 'purple']
                ),
                row=1, col=1
            )
            
            # 손익 분석 바 차트
            face_value = self.config['model']['face_value']
            years = self.config['expected_loss']['years']
            coupon_income = face_value * (result['required_coupon_rate']/100) * years
            
            profit_metrics = {
                '기대손실': result['expected_loss']/1e8,
                '쿠폰 수입': coupon_income/1e8,
                '순이익': (coupon_income - result['expected_loss'])/1e8
            }
            
            fig_reverse.add_trace(
                go.Bar(
                    x=list(profit_metrics.keys()),
                    y=list(profit_metrics.values()),
                    marker_color=['red', 'blue', 'green']
                ),
                row=1, col=2
            )
            
            fig_reverse.update_layout(height=400, title_text="Break-Even 분석 결과")
            st.plotly_chart(fig_reverse, use_container_width=True)
            
            # 리스크 프리미엄 상세 분석
            st.subheader("🔍 Loading Factor 상세 분석")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Loading Factor (α)",
                    f"{result['alpha']:.1f}",
                    help="리스크 프리미엄 계산을 위한 로딩 팩터"
                )
            
            with col2:
                st.metric(
                    "기대손실률 (EL)",
                    f"{result['expected_loss_rate']:.4f}%",
                    help="연간 기대손실률"
                )
            
            with col3:
                st.metric(
                    "리스크 프리미엄 (α×EL)",
                    f"{result['risk_premium']:.4f}%",
                    help="Loading Factor × 기대손실률"
                )
            
            with col4:
                st.metric(
                    "총 쿠폰율",
                    f"{result['required_coupon_rate']:.2f}%",
                    help="무위험 이자율 + 리스크 프리미엄"
                )
            
            # Loading Factor 상세 정보
            st.info(f"""
            **Loading Factor 분석:**
            - **Loading Factor (α)**: {result['alpha']:.1f} (고정값)
            - **기대손실률 (EL)**: {result['expected_loss_rate']:.4f}% (연간)
            - **리스크 프리미엄**: {result['risk_premium']:.4f}% (α × EL)
            - **총 쿠폰율**: {result['required_coupon_rate']:.2f}% (무위험 이자율 + α × EL)
            """)
            
            # 권장사항
            st.subheader("💡 위험 수준별 권장사항")
            
            if result['risk_level'] == "매우 낮음":
                st.success("✅ 매우 안전한 투자입니다.")
                st.info("**권장사항**: 무위험 이자율만으로도 충분하므로 매우 안전한 투자입니다.")
            elif result['risk_level'] == "낮음":
                st.success("✅ 시장에서 수용 가능한 수준입니다.")
                st.info("**권장사항**: 낮은 리스크 프리미엄으로 시장에서 수용 가능한 수준입니다.")
            elif result['risk_level'] == "보통":
                st.warning("⚠️ 적정 수준이지만 검토가 필요합니다.")
                st.info("**권장사항**: 적정 수준이지만 투자자 관점에서 검토가 필요할 수 있습니다.")
            elif result['risk_level'] == "높음":
                st.error("⚠️ 높은 리스크 프리미엄이 필요합니다.")
                st.info("**권장사항**: 높은 리스크 프리미엄이 필요하므로 임계값 조정을 고려해보세요.")
            else:  # 매우 높음
                st.error("❌ 매우 높은 리스크 프리미엄이 필요합니다.")
                st.info("**권장사항**: 매우 높은 리스크 프리미엄이 필요합니다. 임계값을 크게 낮추거나 다른 지역을 고려해보세요.")
            
            # 시나리오 분석
            st.subheader("🎯 임계값별 위험 수준 비교")
            
            # 다양한 임계값에서의 Break-Even 쿠폰율 계산
            threshold_range = [50, 100, 150, 200, 250, 300, 350, 400]
            scenarios = []
            
            for threshold in threshold_range:
                scenario_result = self.calculate_break_even_coupon_rate(region_name, threshold)
                if scenario_result:
                    scenarios.append({
                        '임계값(mm)': threshold,
                        'Break-Even쿠폰율(%)': scenario_result['break_even_rate'],
                        '리스크프리미엄(%)': scenario_result['risk_premium'],
                        '필요쿠폰율(%)': scenario_result['required_coupon_rate'],
                        '위험수준': scenario_result['risk_level'],
                        '기대손실(억원)': scenario_result['expected_loss']/1e8
                    })
            
            if scenarios:
                scenario_df = pd.DataFrame(scenarios)
                st.dataframe(scenario_df.round(2), use_container_width=True)
                
                # 임계값별 쿠폰율 비교 그래프
                fig_scenario = go.Figure()
                
                # Break-Even 쿠폰율
                fig_scenario.add_trace(
                    go.Scatter(
                        x=scenario_df['임계값(mm)'],
                        y=scenario_df['Break-Even쿠폰율(%)'],
                        mode='lines+markers',
                        name='Break-Even 쿠폰율',
                        line=dict(color='blue', width=3)
                    )
                )
                
                # 필요 쿠폰율
                fig_scenario.add_trace(
                    go.Scatter(
                        x=scenario_df['임계값(mm)'],
                        y=scenario_df['필요쿠폰율(%)'],
                        mode='lines+markers',
                        name='필요 쿠폰율',
                        line=dict(color='red', width=3)
                    )
                )
                
                # 무위험 이자율 기준선
                fig_scenario.add_hline(
                    y=result['risk_free_rate'],
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="무위험 이자율"
                )
                
                # 목표 임계값 기준선
                fig_scenario.add_vline(
                    x=target_threshold,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="목표 임계값"
                )
                
                fig_scenario.update_layout(
                    title="임계값별 쿠폰율 비교",
                    xaxis_title="임계값 (mm)",
                    yaxis_title="쿠폰율 (%)",
                    height=400
                )
                st.plotly_chart(fig_scenario, use_container_width=True)
    
    def _run_bidirectional_analysis(self, region_name: str, threshold_min: float, threshold_max: float, threshold_steps: int,
                                   coupon_min: float, coupon_max: float, coupon_steps: int):
        """양방향 관계 분석 실행"""
        st.subheader("🔄 양방향 관계 분석: 임계값 ↔ 쿠폰율")
        
        # 임계값과 쿠폰율 범위 생성
        threshold_range = np.linspace(threshold_min, threshold_max, threshold_steps)
        coupon_range = np.linspace(coupon_min, coupon_max, coupon_steps)
        
        # 최적 조합 찾기
        optimal_combinations = self.find_optimal_threshold_coupon_combinations(region_name, threshold_range, coupon_range)
        
        if not optimal_combinations.empty:
            # 최적 조합 결과 표시
            st.subheader("🏆 최적 조합 분석")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                investor_optimal = optimal_combinations[optimal_combinations['최적화 관점'] == '투자자 관점'].iloc[0]
                st.metric(
                    "투자자 최적",
                    f"{investor_optimal['임계값(mm)']:.1f}mm / {investor_optimal['쿠폰율(%)']:.1f}%",
                    help="기대손실 최소화"
                )
            
            with col2:
                issuer_optimal = optimal_combinations[optimal_combinations['최적화 관점'] == '발행자 관점'].iloc[0]
                st.metric(
                    "발행자 최적",
                    f"{issuer_optimal['임계값(mm)']:.1f}mm / {issuer_optimal['쿠폰율(%)']:.1f}%",
                    help="발행자 점수 최대화"
                )
            
            with col3:
                risk_optimal = optimal_combinations[optimal_combinations['최적화 관점'] == '위험대비수익률'].iloc[0]
                st.metric(
                    "위험대비 최적",
                    f"{risk_optimal['임계값(mm)']:.1f}mm / {risk_optimal['쿠폰율(%)']:.1f}%",
                    help="위험 대비 수익률 최대화"
                )
            
            # 최적 조합 비교 테이블
            st.dataframe(optimal_combinations, use_container_width=True)
            
            # 관계 분석
            st.subheader("📊 임계값-쿠폰율 관계 분석")
            
            # 모든 조합 분석 (샘플링하여 성능 개선)
            sample_thresholds = np.linspace(threshold_min, threshold_max, min(10, threshold_steps))
            sample_coupons = np.linspace(coupon_min, coupon_max, min(10, coupon_steps))
            
            relationship_df = self.analyze_threshold_coupon_relationship(region_name, sample_thresholds, sample_coupons)
            
            if not relationship_df.empty:
                # 히트맵 생성
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=relationship_df['issuer_score'].values.reshape(len(sample_thresholds), len(sample_coupons)),
                    x=[f"{c*100:.1f}%" for c in sample_coupons],
                    y=[f"{t:.0f}mm" for t in sample_thresholds],
                    colorscale='Viridis',
                    colorbar=dict(title="발행자 점수")
                ))
                
                fig_heatmap.update_layout(
                    title="임계값-쿠폰율 관계 히트맵 (발행자 점수)",
                    xaxis_title="쿠폰율",
                    yaxis_title="임계값",
                    height=500
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # 3D 산점도
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=relationship_df['threshold'],
                    y=relationship_df['coupon_rate_pct'],
                    z=relationship_df['issuer_score'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=relationship_df['issuer_score'],
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=[f"임계값: {t:.0f}mm<br>쿠폰율: {c:.1f}%<br>점수: {s:.1f}" 
                          for t, c, s in zip(relationship_df['threshold'], 
                                           relationship_df['coupon_rate_pct'], 
                                           relationship_df['issuer_score'])],
                    hovertemplate='%{text}<extra></extra>'
                )])
                
                fig_3d.update_layout(
                    title="임계값-쿠폰율-발행자점수 3D 관계",
                    scene=dict(
                        xaxis_title="임계값 (mm)",
                        yaxis_title="쿠폰율 (%)",
                        zaxis_title="발행자 점수"
                    ),
                    height=600
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # 권장사항
            st.subheader("💡 양방향 분석 권장사항")
            
            # 투자자와 발행자 관점의 차이 분석
            threshold_diff = abs(investor_optimal['임계값(mm)'] - issuer_optimal['임계값(mm)'])
            coupon_diff = abs(investor_optimal['쿠폰율(%)'] - issuer_optimal['쿠폰율(%)'])
            
            if threshold_diff < 20 and coupon_diff < 2:
                st.success("✅ 투자자와 발행자 관점이 유사합니다. 협의 가능성이 높습니다.")
            elif threshold_diff < 50 and coupon_diff < 5:
                st.warning("⚠️ 투자자와 발행자 관점에 차이가 있지만, 협의를 통해 해결 가능합니다.")
            else:
                st.error("❌ 투자자와 발행자 관점에 큰 차이가 있습니다. 상세한 협의가 필요합니다.")
            
            st.info("**권장 협의 포인트:**")
            st.info(f"- 임계값 범위: {min(investor_optimal['임계값(mm)'], issuer_optimal['임계값(mm)']):.0f}mm ~ {max(investor_optimal['임계값(mm)'], issuer_optimal['임계값(mm)']):.0f}mm")
            st.info(f"- 쿠폰율 범위: {min(investor_optimal['쿠폰율(%)'], issuer_optimal['쿠폰율(%)']):.1f}% ~ {max(investor_optimal['쿠폰율(%)'], issuer_optimal['쿠폰율(%)']):.1f}%")
        
        else:
            st.error("❌ 양방향 분석을 위한 데이터를 생성할 수 없습니다.")
    
    def run_app(self):
        """Streamlit 앱 실행"""
        
        # 헤더
        st.title("🌧️ CatBond 임계값 분석 대시보드")
        st.markdown("---")
        
        # 사이드바 설정
        with st.sidebar:
            # 간단한 헤더
            st.title("🌧️ CatBond 분석")
            st.caption("임계값 → 쿠폰율 역산 분석")
            st.divider()
            
            available_regions = self.data['df_rain']['region'].unique() if self.data else []
            
            # 지역 선택
            st.subheader("📍 지역 선택")
            
            # 학습된 지역 통계에서 사용 가능한 지역만 필터링
            if self.data and self.data['feature_engineer'].region_stats is not None:
                trained_regions = self.data['feature_engineer'].region_stats['region'].unique()
                available_regions = [r for r in available_regions if r in trained_regions]
                
                if not available_regions:
                    st.error("❌ 학습된 지역이 없습니다. 모델을 다시 학습해주세요.")
                    return
                
                # 충주 지역이 있으면 기본값으로 설정
                default_index = 0
                if self.data['chungju_region_name'] in available_regions:
                    default_index = list(available_regions).index(self.data['chungju_region_name'])
                
                selected_region = st.selectbox(
                    "분석할 지역을 선택하세요",
                    available_regions,
                    index=default_index,
                    help="모델 학습 시 사용된 지역만 선택 가능합니다."
                )
                
                # 선택된 지역 표시
                st.info(f"✅ 선택된 지역: {selected_region}")
                
                # 학습되지 않은 지역 경고
                if len(available_regions) < len(self.data['df_rain']['region'].unique()):
                    st.warning(f"⚠️ 일부 지역({len(self.data['df_rain']['region'].unique()) - len(available_regions)}개)은 모델 학습에 사용되지 않아 선택할 수 없습니다.")
            else:
                st.error("❌ 지역 통계 정보가 없습니다. 모델을 다시 학습해주세요.")
                return
            
            st.divider()
            
            # 임계값 설정
            st.subheader("🎯 임계값 설정")
            
            target_threshold = st.slider(
                "강수량 임계값 (mm)",
                min_value=50,
                max_value=500,
                value=200,
                step=5,
                help="트리거 발생 기준이 되는 강수량 임계값을 설정하세요 (5mm 단위)"
            )
            
            # 임계값 수준 표시
            if target_threshold <= 100:
                threshold_level = "매우 낮음"
                st.error(f"📊 임계값 수준: {threshold_level} ({target_threshold}mm)")
            elif target_threshold <= 200:
                threshold_level = "낮음"
                st.warning(f"📊 임계값 수준: {threshold_level} ({target_threshold}mm)")
            elif target_threshold <= 300:
                threshold_level = "보통"
                st.info(f"📊 임계값 수준: {threshold_level} ({target_threshold}mm)")
            elif target_threshold <= 400:
                threshold_level = "높음"
                st.success(f"📊 임계값 수준: {threshold_level} ({target_threshold}mm)")
            else:
                threshold_level = "매우 높음"
                st.success(f"📊 임계값 수준: {threshold_level} ({target_threshold}mm)")
            
            st.divider()
            
            # 기준 정보
            st.subheader("💰 기준 정보")
            
            # 무위험 이자율 표시
            risk_free_rate = 0.025  # 2.5%
            st.metric("무위험 이자율", f"{risk_free_rate*100:.1f}%")
            
            # 분석 설명
            with st.expander("💡 분석 설명", expanded=False):
                st.markdown("""
                **역산 분석 (임계값 → 쿠폰율):**
                
                📈 **Break-Even 쿠폰율**: 기대손실을 보전하는 최소 쿠폰율
                
                🎯 **Loading Factor 방식**: 
                - 무위험 이자율 + α × EL(Expected Loss)
                - α(Loading Factor) = 2.0 (고정값)
                - 간단하고 투명한 리스크 프리미엄 계산
                
                ⚠️ **위험 수준 평가**: 임계값별 위험도 및 권장사항
                
                📊 **시나리오 비교**: 다양한 임계값에서의 쿠폰율 비교
                """)
            
            st.divider()
            
            # 분석 실행 버튼
            analyze_button = st.button(
                "🚀 분석 실행",
                type="primary",
                use_container_width=True,
                help="설정된 임계값으로 역산 분석을 실행합니다"
            )
        
        # 메인 컨텐츠
        if analyze_button:
            if self.models is None or self.data is None:
                st.error("❌ 모델 또는 데이터 로드에 실패했습니다.")
                return
            
            # 진행 상황 표시
            with st.spinner("역산 분석 중..."):
                self._run_reverse_analysis(selected_region, target_threshold)
        
        else:
            # 초기 화면
            st.info("👈 사이드바에서 분석 설정을 조정하고 '분석 실행' 버튼을 클릭하세요.")
            
            # 사용법 안내
            with st.expander("📖 사용법"):
                st.markdown("""
                ### 사용법
                1. **지역 선택**: 분석할 지역을 선택합니다.
                2. **목표 임계값 설정**: 분석할 임계값을 설정합니다 (50-500mm).
                3. **분석 실행**: 설정을 완료한 후 '분석 실행' 버튼을 클릭합니다.
                
                ### 분석 결과
                - **Break-Even 쿠폰율**: 기대손실을 보전하는 최소 쿠폰율
                - **다층 리스크 프리미엄**: 기본 + 임계값 + 빈도 + 손실률 기반
                - **위험 수준 평가**: 임계값별 위험도 및 권장사항
                - **시나리오 비교**: 다양한 임계값에서의 쿠폰율 비교
                """)
            
            # 현재 설정 정보
            if self.data:
                st.subheader("📊 현재 설정 정보")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("분석 지역", self.data['chungju_region_name'])
                
                with col2:
                    st.metric("현재 임계값", f"{self.data['chungju_threshold']:.1f}mm")
                
                with col3:
                    st.metric("액면가", f"{self.config['model']['face_value']/1e8:.0f}억원")
                
                with col4:
                    if self.data.get('inje_threshold') is not None:
                        st.metric("인제군 임계값", f"{self.data['inje_threshold']:.1f}mm")
                    else:
                        st.metric("인제군 임계값", "N/A")

def main():
    """메인 함수"""
    try:
        app = ThresholdAnalysisApp()
        app.run_app()
    except Exception as e:
        st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
        st.exception(e)

if __name__ == "__main__":
    main() 