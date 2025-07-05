"""
CatBond 모델 전체 학습/예측 파이프라인
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import joblib
from scipy.stats import poisson
from .data import CatBondDataLoader
from .features import FeatureEngineer
from .model import TwoPartModel
from .utils import timer, set_seed, ensure_dir

class ExpectedLossCalculator:
    """Cat Bond Expected Loss 계산기"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.face_value = config['expected_loss']['face_value']  # 액면가 (F)
        self.years = config['expected_loss']['years']  # 만기 (n)
        
    def calculate_el(self, lambda_hat: float, loss_rate: float) -> Dict[str, float]:
        """
        Cat Bond 기대손실(EL) 계산
        
        Args:
            lambda_hat: 연간 기대 사건 수 (λ)
            loss_rate: 사건당 손실률 (L) - 비율값 (0~1)
            
        Returns:
            계산 결과 딕셔너리
        """
        # loss_rate를 1.0으로 clipping (최대 100% 손실로 제한)
        clipped_loss_rate = min(loss_rate, 1.0)
        
        # 계산 공식: EL = F × λ × n × L (clipped)
        expected_loss = self.face_value * lambda_hat * self.years * clipped_loss_rate
        
        # 기대 상환액: F × (1 - λ × n × L)
        expected_repayment = self.face_value * (1 - lambda_hat * self.years * clipped_loss_rate)
        
        # 기대 손실률: λ × n × L
        expected_loss_rate = lambda_hat * self.years * clipped_loss_rate
        
        return {
            'expected_loss': expected_loss,  # 기대손실 (원)
            'expected_repayment': expected_repayment,  # 기대 상환액 (원)
            'expected_loss_rate': expected_loss_rate,  # 기대 손실률 (비율)
            'face_value': self.face_value,  # 액면가 (원)
            'lambda_hat': lambda_hat,  # 연간 기대 사건 수
            'loss_rate': loss_rate,  # 원본 사건당 손실률 (비율)
            'clipped_loss_rate': clipped_loss_rate,  # clipping된 사건당 손실률 (비율)
            'years': self.years  # 만기
        }
        
    def estimate_premium(self, expected_loss: float, 
                        loading_factor: float = 0.05) -> float:
        """보험료 추정"""
        return expected_loss * loading_factor

@timer
def train_catbond_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    CatBond 모델 전체 학습 파이프라인
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        학습 결과 딕셔너리
    """
    from .utils import load_config, setup_logging
    
    # 1. 설정 및 초기화
    logger = setup_logging()
    config = load_config(config_path)
    set_seed(config['training']['random_state'])
    
    print("[INFO] CatBond 모델 학습 시작")
    
    # 2. 데이터 로딩
    data_loader = CatBondDataLoader(config)
    
    # 강수 데이터 로드
    df_rain = data_loader.load_rain_data()
    
    # 손실 및 피처 데이터 로드
    df_loss = data_loader.load_loss_data()
    
    # 기후지수 데이터 로드
    clim = data_loader.load_climate_data()
    
    # 3. 임계값 계산
    region_thresholds = data_loader.calculate_thresholds(df_rain)
    chungju_region_name, chungju_threshold = data_loader.get_chungju_info(
        df_rain, region_thresholds
    )
    
    # 4. 피처 엔지니어링
    feature_engineer = FeatureEngineer(config)
    
    # 빈도 모델 데이터 준비
    freq_df = data_loader.prepare_frequency_data(df_rain, region_thresholds, clim)
    freq_df = feature_engineer.calculate_exposure(freq_df)
    X_freq, y_freq, sample_weights_freq = feature_engineer.prepare_frequency_features(freq_df)
    
    # 트리거 모델 데이터 준비
    df_full = data_loader.prepare_trigger_data(df_loss, region_thresholds)
    X_trigger, y_trigger = feature_engineer.prepare_trigger_features(df_full)
    
    # 손실 모델 데이터 준비
    df_triggered = df_full[df_full["trigger_flag"] == 1].copy()
    df_triggered = data_loader.prepare_severity_data(df_triggered)
    df_triggered = feature_engineer.create_region_embeddings(df_triggered)
    X_severity, y_severity = feature_engineer.prepare_severity_features(df_triggered)
    
    # 피처 스케일링
    X_trigger_scaled = feature_engineer.scale_features(X_trigger, 'trigger')
    X_severity_scaled = feature_engineer.scale_features(X_severity, 'severity')
    
    # 지역별 가중치
    sample_weights_severity = feature_engineer.get_region_weights(
        df_triggered, chungju_region_name
    )
    
    # 그룹 정보 (지역별 분할용)
    groups_trigger = X_trigger['region'].astype('category').cat.codes.values
    groups_severity = X_severity['region'].astype('category').cat.codes.values
    
    # 5. 모델 학습
    two_part_model = TwoPartModel(config)
    
    data_dict = {
        'X_freq': X_freq,
        'y_freq': y_freq,
        'sample_weights_freq': sample_weights_freq,
        'X_trigger': X_trigger_scaled,
        'y_trigger': y_trigger,
        'groups_trigger': groups_trigger,
        'X_severity': X_severity_scaled,
        'y_severity': y_severity,
        'sample_weights_severity': sample_weights_severity,
        'groups_severity': groups_severity
    }
    
    two_part_model.fit(data_dict)
    
    # 6. Cat Bond Expected Loss 계산
    el_calculator = ExpectedLossCalculator(config)
    
    # 충주 지역 기대 사건 수 (λ)
    lambda_hat_chj = two_part_model.frequency_model.get_lambda_hat()
    
    # 충주 지역 사건당 손실률 (L)
    loss_rate = two_part_model.get_conditional_loss(df_triggered, chungju_region_name)
    
    # Cat Bond EL 계산: EL = F × λ × n × L
    el_results = el_calculator.calculate_el(lambda_hat_chj, loss_rate)
    
    # 3년간 최소 1회 발생 확률 계산
    lambda_3yr = lambda_hat_chj * el_results['years']
    poisson_dist = poisson(lambda_3yr)
    prob_at_least_one = 1 - poisson_dist.pmf(0)
    prob_at_least_two = 1 - poisson_dist.pmf(0) - poisson_dist.pmf(1)
    
    # 보험료 추정
    loading_factor = config['expected_loss']['loading_factor']
    estimated_premium = el_calculator.estimate_premium(el_results['expected_loss'], loading_factor)
    
    # 7. 모델 저장
    ensure_dir("models")
    two_part_model.save_models(config['output'])
    
    # 지역 정보 저장
    region_info = {
        'region_thresholds': region_thresholds,
        'chungju_region_name': chungju_region_name,
        'chungju_threshold': chungju_threshold,
        'region_stats': feature_engineer.region_stats,
        'region_embedding_df': feature_engineer.region_embedding_df,
        'trigger_scaler': feature_engineer.trigger_scaler,
        'severity_scaler': feature_engineer.severity_scaler,
        'face_value': config['model']['face_value'],  # cover_limit 대신 face_value 저장
        'config': config
    }
    
    joblib.dump(region_info, config['output']['region_info'])
    
    # 8. 결과 반환
    results = {
        'lambda_hat_chj': lambda_hat_chj,  # 연간 기대 사건 수 (λ)
        'loss_rate': loss_rate,  # 사건당 손실률 (L)
        'face_value': el_results['face_value'],  # 액면가 (F)
        'years': el_results['years'],  # 만기 (n)
        'expected_loss': el_results['expected_loss'],  # 기대손실 (EL)
        'expected_repayment': el_results['expected_repayment'],  # 기대 상환액
        'expected_loss_rate': el_results['expected_loss_rate'],  # 기대 손실률
        'estimated_premium': estimated_premium,  # 추정 보험료
        'prob_at_least_one_3y': prob_at_least_one,  # 3년간 최소 1회 발생 확률
        'prob_at_least_two_3y': prob_at_least_two,  # 3년간 최소 2회 발생 확률
        'lambda_3yr': lambda_3yr,  # 3년간 기대 사건 수
        'chungju_region_name': chungju_region_name,
        'chungju_threshold': chungju_threshold,
        'config': config
    }
    
    print("\n[DONE] Cat Bond 모델 학습 완료")
    print(f"  • 충주 지역: {chungju_region_name}")
    print(f"  • 임계값: {chungju_threshold:.1f}mm")
    print(f"  • 연간 기대 사건 수 (λ): {lambda_hat_chj:.3f}건")
    print(f"  • 3년간 기대 사건 수: {lambda_3yr:.3f}건")
    print(f"  • 3년간 최소 1회 발생 확률: {prob_at_least_one:.3%}")
    print(f"  • 3년간 최소 2회 발생 확률: {prob_at_least_two:.3%}")
    print(f"  • 사건당 손실률 (L): {loss_rate:.6f}")
    if loss_rate > 1.0:
        print(f"  • clipping된 손실률: {el_results['clipped_loss_rate']:.6f}")
    print(f"  • 기대 손실률: {el_results['expected_loss_rate']:.6f}")
    print(f"  • 액면가 (F): {el_results['face_value']:,.0f} 원 ({el_results['face_value']/1e8:.1f}억원)")
    print(f"  • 기대손실 (EL): {el_results['expected_loss']:,.0f} 원 ({el_results['expected_loss']/1e8:.1f}억원)")
    print(f"  • 기대 상환액: {el_results['expected_repayment']:,.0f} 원 ({el_results['expected_repayment']/1e8:.1f}억원)")
    print(f"  • 추정 보험료: {estimated_premium:,.0f} 원 ({estimated_premium/1e8:.1f}억원)")
    
    return results

@timer
def predict_catbond(input_data: pd.DataFrame, 
                   model_paths: Dict[str, str],
                   region_info_path: str) -> pd.DataFrame:
    """
    CatBond 모델 예측
    
    Args:
        input_data: 입력 데이터
        model_paths: 모델 파일 경로들
        region_info_path: 지역 정보 파일 경로
        
    Returns:
        예측 결과 데이터프레임
    """
    # 1. 모델 및 정보 로드
    region_info = joblib.load(region_info_path)
    trigger_model = joblib.load(model_paths['trigger_model'])
    severity_model = lgb.Booster(model_file=model_paths['severity_model'])
    
    # 2. 피처 준비
    trigger_features = region_info['config']['features']['trigger_features']
    severity_features = region_info['config']['features']['severity_features']
    
    # 3. 피처 스케일링
    trigger_scaler = region_info['trigger_scaler']
    severity_scaler = region_info['severity_scaler']
    
    # 트리거 모델용 데이터
    X_trigger = input_data[trigger_features + ["region"]].copy()
    trigger_numeric_features = [col for col in trigger_features if col not in ['region']]
    X_trigger[trigger_numeric_features] = trigger_scaler.transform(X_trigger[trigger_numeric_features])
    
    # 손실 모델용 데이터
    X_severity = input_data[severity_features + ["region"]].copy()
    severity_numeric_features = [col for col in severity_features if col not in ['region']]
    X_severity[severity_numeric_features] = severity_scaler.transform(X_severity[severity_numeric_features])
    
    # 4. 예측
    trigger_proba = trigger_model.predict_proba(X_trigger)[:, 1]
    severity_pred_log = severity_model.predict(X_severity)
    severity_pred_ratio = np.expm1(severity_pred_log)
    
    # 5. Two-Part 모델 예측
    final_pred_ratio = trigger_proba * severity_pred_ratio
    final_pred_absolute = final_pred_ratio * region_info['cover_limit']
    
    # 6. 결과 반환
    results = input_data.copy()
    results['trigger_proba'] = trigger_proba
    results['severity_pred_ratio'] = severity_pred_ratio
    results['final_pred_ratio'] = final_pred_ratio
    results['final_pred_absolute'] = final_pred_absolute
    
    return results 