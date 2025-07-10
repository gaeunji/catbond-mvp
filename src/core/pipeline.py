"""
CatBond 모델 전체 학습/예측 파이프라인
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import joblib
import lightgbm as lgb
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging
from .data import CatBondDataLoader
from .features import FeatureEngineer
from .model import TwoPartModel
from .utils import timer, set_seed, ensure_dir

# 로깅 설정
logger = logging.getLogger(__name__)

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
    
    def calculate_el_confidence_interval(self, lambda_samples: np.ndarray, 
                                       loss_rate_samples: np.ndarray,
                                       confidence_level: float = 0.95) -> Dict[str, float]:
        """
        부트스트랩을 통한 EL 신뢰구간 계산
        
        Args:
            lambda_samples: λ의 부트스트랩 샘플들
            loss_rate_samples: L의 부트스트랩 샘플들
            confidence_level: 신뢰수준 (기본값: 0.95)
            
        Returns:
            신뢰구간 정보
        """
        # EL 샘플들 계산
        el_samples = []
        for lambda_hat, loss_rate in zip(lambda_samples, loss_rate_samples):
            clipped_loss_rate = min(loss_rate, 1.0)
            el = self.face_value * lambda_hat * self.years * clipped_loss_rate
            el_samples.append(el)
        
        el_samples = np.array(el_samples)
        
        # 신뢰구간 계산
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        el_lower = np.percentile(el_samples, lower_percentile)
        el_upper = np.percentile(el_samples, upper_percentile)
        el_mean = np.mean(el_samples)
        el_std = np.std(el_samples)
        
        # 변동계수 (CV = 표준편차/평균)
        cv = el_std / el_mean if el_mean > 0 else 0
        
        return {
            'el_mean': el_mean,
            'el_std': el_std,
            'el_cv': cv,  # 변동계수
            'el_lower': el_lower,
            'el_upper': el_upper,
            'confidence_level': confidence_level,
            'lambda_mean': np.mean(lambda_samples),
            'lambda_std': np.std(lambda_samples),
            'loss_rate_mean': np.mean(loss_rate_samples),
            'loss_rate_std': np.std(loss_rate_samples)
        }
        
    def estimate_premium(self, expected_loss: float, 
                        loading_factor: float = 0.05) -> float:
        """보험료 추정"""
        return expected_loss * loading_factor
    
    def calculate_coupon_rate(self, expected_loss: float, 
                             face_value: float, years: int,
                             risk_free_rate: float = 0.03,
                             risk_premium: float = 0.02) -> Dict[str, float]:
        """
        채권 발행자 관점의 쿠폰률 산정
        
        Args:
            expected_loss: 기대손실 (EL)
            face_value: 액면가 (F)
            years: 만기 (n)
            risk_free_rate: 무위험 이자율 (기본값: 3%)
            risk_premium: 위험 프리미엄 (기본값: 2%)
            
        Returns:
            쿠폰률 관련 정보
        """
        # 1. 기본 쿠폰률 (무위험 이자율 + 위험 프리미엄)
        base_coupon_rate = risk_free_rate + risk_premium
        
        # 2. 재해 위험에 대한 추가 쿠폰률
        # EL을 연간 비용으로 환산
        annual_expected_loss = expected_loss / years
        
        # 액면가 대비 연간 기대손실 비율
        annual_loss_rate = annual_expected_loss / face_value
        
        # 재해 위험 프리미엄 (EL을 보상하기 위한 추가 이자)
        catastrophe_premium = annual_loss_rate
        
        # 3. 총 쿠폰률
        total_coupon_rate = base_coupon_rate + catastrophe_premium
        
        # 4. 연간 쿠폰 지급액
        annual_coupon_payment = face_value * total_coupon_rate
        
        # 5. 3년간 총 쿠폰 지급액
        total_coupon_payments = annual_coupon_payment * years
        
        # 6. 기대 비용 대비 쿠폰 수익
        net_benefit = total_coupon_payments - expected_loss
        
        return {
            'base_coupon_rate': base_coupon_rate,           # 기본 쿠폰률
            'catastrophe_premium': catastrophe_premium,     # 재해 위험 프리미엄
            'total_coupon_rate': total_coupon_rate,         # 총 쿠폰률
            'annual_coupon_payment': annual_coupon_payment, # 연간 쿠폰 지급액
            'total_coupon_payments': total_coupon_payments, # 3년간 총 쿠폰 지급액
            'expected_loss': expected_loss,                 # 기대손실
            'net_benefit': net_benefit,                     # 순이익 (쿠폰 - EL)
            'risk_free_rate': risk_free_rate,               # 무위험 이자율
            'risk_premium': risk_premium                     # 위험 프리미엄
        }
    
    def calculate_investor_yield(self, coupon_rate: float, 
                                face_value: float, years: int,
                                prob_at_least_one: float,
                                expected_loss_rate: float) -> Dict[str, float]:
        """
        투자자 관점의 수익률 계산
        
        Args:
            coupon_rate: 쿠폰률
            face_value: 액면가
            years: 만기
            prob_at_least_one: 최소 1회 발생 확률
            expected_loss_rate: 기대 손실률
            
        Returns:
            투자자 수익률 정보
        """
        # 1. 쿠폰 수익
        annual_coupon = face_value * coupon_rate
        total_coupon_income = annual_coupon * years
        
        # 2. 원금 상환 기대값
        expected_repayment = face_value * (1 - expected_loss_rate)
        
        # 3. 총 기대 수익
        total_expected_return = total_coupon_income + expected_repayment
        
        # 4. 투자 원금
        initial_investment = face_value
        
        # 5. 총 수익률
        total_return_rate = (total_expected_return - initial_investment) / initial_investment
        
        # 6. 연평균 수익률 (복리 기준)
        annual_return_rate = (1 + total_return_rate) ** (1/years) - 1
        
        # 7. 위험 조정 수익률 (재해 발생 확률 고려)
        risk_adjusted_return = annual_return_rate * (1 - prob_at_least_one)
        
        return {
            'annual_coupon': annual_coupon,                    # 연간 쿠폰
            'total_coupon_income': total_coupon_income,        # 총 쿠폰 수익
            'expected_repayment': expected_repayment,          # 기대 상환액
            'total_expected_return': total_expected_return,    # 총 기대 수익
            'total_return_rate': total_return_rate,            # 총 수익률
            'annual_return_rate': annual_return_rate,          # 연평균 수익률
            'risk_adjusted_return': risk_adjusted_return,      # 위험 조정 수익률
            'prob_at_least_one': prob_at_least_one,            # 재해 발생 확률
            'expected_loss_rate': expected_loss_rate           # 기대 손실률
        }

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
    
    logger.info("CatBond 모델 학습 시작")
    
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
    
    # 인제군 정보 추가
    inje_region_name = "강원도 인제군"
    inje_threshold = region_thresholds.get(inje_region_name, None)
    if inje_threshold is not None:
        logger.info(f"인제군 임계값: {inje_threshold:.1f}mm")
    else:
        logger.warning(f"인제군 임계값 정보를 찾을 수 없습니다.")
    
    # 4. 피처 엔지니어링
    feature_engineer = FeatureEngineer(config)
    
    # 빈도 모델 데이터 준비
    freq_df = data_loader.prepare_frequency_data(df_rain, region_thresholds, clim)
    freq_df = feature_engineer.calculate_exposure(freq_df)
    X_freq, y_freq, sample_weights_freq = feature_engineer.prepare_frequency_features(freq_df)
    
    # 트리거 모델 데이터 준비
    df_full = data_loader.prepare_trigger_data(df_loss, region_thresholds)
    X_trigger, y_trigger = feature_engineer.prepare_trigger_features(df_full)
    
    # 손실 모델 데이터 준비 (모든 날의 데이터 포함)
    # 1. 전체 데이터에 대해 severity 초기화 (0으로 설정)
    df_full['severity'] = 0.0
    df_full['severity_log'] = 0.0
    
    # 2. 트리거가 발생한 날만 실제 손실률 계산
    df_triggered = df_full[df_full["trigger_flag"] == 1].copy()
    df_triggered = data_loader.prepare_severity_data(df_triggered)
    
    # 3. 계산된 severity를 전체 데이터에 업데이트
    df_full.loc[df_triggered.index, 'severity'] = df_triggered['severity']
    df_full.loc[df_triggered.index, 'severity_log'] = df_triggered['severity_log']
    
    # 4. 지역 임베딩 생성 (전체 데이터 사용)
    df_full = feature_engineer.create_region_embeddings(df_full)
    X_severity, y_severity = feature_engineer.prepare_severity_features(df_full)
    
    # 손실 데이터 분포 분석 및 로깅 (전체 데이터 기준)
    zero_loss_count = (y_severity == 0).sum()
    positive_loss_count = (y_severity > 0).sum()
    total_severity_count = len(y_severity)
    
    logger.info(f"SEVERITY-DATA 전체 데이터: {total_severity_count:,}개")
    logger.info(f"SEVERITY-DATA 손실이 0인 데이터: {zero_loss_count:,}개 ({zero_loss_count/total_severity_count:.1%})")
    logger.info(f"SEVERITY-DATA 양수 손실 데이터: {positive_loss_count:,}개 ({positive_loss_count/total_severity_count:.1%})")
    
    if positive_loss_count > 0:
        logger.info(f"SEVERITY-DATA 양수 손실 평균: {y_severity[y_severity > 0].mean():.6f}")
        logger.info(f"SEVERITY-DATA 양수 손실 중앙값: {y_severity[y_severity > 0].median():.6f}")
        logger.info(f"SEVERITY-DATA 양수 손실 최대값: {y_severity[y_severity > 0].max():.6f}")
    
    # 트리거 발생률도 로깅
    trigger_count = df_full['trigger_flag'].sum()
    logger.info(f"SEVERITY-DATA 트리거 발생률: {trigger_count:,}개 ({trigger_count/total_severity_count:.1%})")
    
    # 피처 스케일링
    X_trigger_scaled = feature_engineer.scale_features(X_trigger, 'trigger')
    X_severity_scaled = feature_engineer.scale_features(X_severity, 'severity')
    
    # 지역별 가중치 (전체 데이터에 대한 가중치 조정)
    sample_weights_severity = feature_engineer.get_region_weights(
        df_full, chungju_region_name
    )
    
    # 손실이 0인 데이터에 대한 가중치 조정 (선택적)
    # 손실이 0인 데이터가 많을 때 양수 손실 데이터에 더 높은 가중치 부여
    if zero_loss_count > positive_loss_count * 10:  # 0 손실이 양수 손실의 10배 이상일 때 (전체 데이터 사용으로 인해 비율이 크게 증가)
        zero_loss_mask = (y_severity == 0)
        positive_loss_mask = (y_severity > 0)
        
        # 양수 손실 데이터에 더 높은 가중치 부여 (전체 데이터 대비 균형 맞추기)
        weight_multiplier = min(10.0, zero_loss_count / positive_loss_count * 0.1)  # 최대 10배로 제한
        sample_weights_severity[positive_loss_mask] *= weight_multiplier
        
        logger.info(f"SEVERITY-WEIGHT 양수 손실 데이터에 {weight_multiplier:.1f}배 가중치 적용")
        logger.info(f"SEVERITY-WEIGHT 0 손실 데이터 평균 가중치: {sample_weights_severity[zero_loss_mask].mean():.3f}")
        logger.info(f"SEVERITY-WEIGHT 양수 손실 데이터 평균 가중치: {sample_weights_severity[positive_loss_mask].mean():.3f}")
    
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
    
    # 6. 모델 성능 평가
    logger.info("모델 성능 평가 시작")
    
    # 트리거 모델 성능 평가
    trigger_pred_proba = two_part_model.trigger_model.predict_proba(X_trigger_scaled)
    trigger_pred = (trigger_pred_proba >= two_part_model.trigger_model.optimal_threshold).astype(int)
    
    trigger_metrics = {
        'accuracy': accuracy_score(y_trigger, trigger_pred),
        'precision': precision_score(y_trigger, trigger_pred, zero_division=0),
        'recall': recall_score(y_trigger, trigger_pred, zero_division=0),
        'f1': f1_score(y_trigger, trigger_pred, zero_division=0),
        'auc': roc_auc_score(y_trigger, trigger_pred_proba),
        'optimal_threshold': two_part_model.trigger_model.optimal_threshold
    }
    
    logger.info(f"TRIGGER 성능 - 정확도: {trigger_metrics['accuracy']:.4f}, 정밀도: {trigger_metrics['precision']:.4f}")
    logger.info(f"TRIGGER 성능 - 재현율: {trigger_metrics['recall']:.4f}, F1점수: {trigger_metrics['f1']:.4f}")
    logger.info(f"TRIGGER 성능 - AUC: {trigger_metrics['auc']:.4f}, 최적 임계값: {trigger_metrics['optimal_threshold']:.3f}")
    
    # 손실 모델 성능 평가 (교차검증 결과 활용)
    # MLP 모델은 이미 스케일링된 데이터를 사용하므로 별도 스케일링 불필요
    severity_pred_log = two_part_model.severity_model.predict(X_severity_scaled)
    severity_pred_ratio = np.expm1(severity_pred_log)
    
    # 인제군 손실률 계산
    inje_loss_info = two_part_model.get_conditional_loss(df_full, inje_region_name)
    if inje_loss_info['loss_rate'] > 0:
        logger.info(f"인제군 손실률: {inje_loss_info['loss_rate']:.6f} (사건 수: {inje_loss_info['event_count']}개, 전체 데이터: {inje_loss_info['total_count']}개)")
    else:
        logger.warning(f"인제군 손실률: 손실이 발생한 사건이 없습니다.")
    
    # 교차검증 결과 가져오기
    cv_summary = two_part_model.severity_model.get_cv_summary()
    
    if cv_summary:
        logger.info(f"MLP-SEVERITY-CV 교차검증 결과 - 최적 Fold: {cv_summary['best_fold']}")
        logger.info(f"MLP-SEVERITY-CV 평균 R²: {cv_summary['mean_r2']:.4f} ± {cv_summary['std_r2']:.4f}")
        logger.info(f"MLP-SEVERITY-CV 평균 MAE: {cv_summary['mean_mae']:.6f} ± {cv_summary['std_mae']:.6f}")
        logger.info(f"MLP-SEVERITY-CV 평균 RMSE: {cv_summary['mean_rmse']:.6f} ± {cv_summary['std_rmse']:.6f}")
        logger.info(f"MLP-SEVERITY-CV 평균 F1 Score: {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")
        logger.info(f"MLP-SEVERITY-CV 평균 Precision: {cv_summary['mean_precision']:.4f} ± {cv_summary['std_precision']:.4f}")
        logger.info(f"MLP-SEVERITY-CV 평균 Recall: {cv_summary['mean_recall']:.4f} ± {cv_summary['std_recall']:.4f}")
        logger.info(f"MLP-SEVERITY-CV 평균 Accuracy: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        logger.info(f"MLP-SEVERITY-CV 손실률 분류 임계값: {cv_summary['severity_threshold']:.6f}")
    
    # 최종 모델 성능 평가 (교차검증과 동일한 방식)
    from sklearn.model_selection import GroupKFold
    
    # 최적 Fold와 동일한 분할로 재평가
    n_splits = config['training'].get('cv_folds', 5)
    random_state = config['training']['random_state']
    groups_severity = X_severity['region'].astype('category').cat.codes.values
    
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_fold = cv_summary['best_fold'] if cv_summary else 1
    
    # 최적 Fold의 검증 세트 인덱스 찾기
    fold_splits = list(cv.split(X_severity, y_severity, groups=groups_severity))
    best_train_idx, best_val_idx = fold_splits[best_fold - 1]
    
    # 최적 Fold의 검증 세트에서만 평가
    y_severity_best_val = y_severity.iloc[best_val_idx]
    severity_pred_ratio_best_val = severity_pred_ratio[best_val_idx]
    
    # 최적 Fold 검증 성능
    best_fold_metrics = {
        'mae': mean_absolute_error(y_severity_best_val, severity_pred_ratio_best_val),
        'rmse': np.sqrt(mean_squared_error(y_severity_best_val, severity_pred_ratio_best_val)),
        'r2': r2_score(y_severity_best_val, severity_pred_ratio_best_val),
        'mae_absolute': mean_absolute_error(y_severity_best_val * config['model']['face_value'], 
                                          severity_pred_ratio_best_val * config['model']['face_value']),
        'rmse_absolute': np.sqrt(mean_squared_error(y_severity_best_val * config['model']['face_value'], 
                                                   severity_pred_ratio_best_val * config['model']['face_value']))
    }
    
    # F1 Score 계산 (최적 Fold 검증 세트)
    severity_threshold = cv_summary.get('severity_threshold', 0.001) if cv_summary else 0.001
    y_true_binary_best_val = (y_severity_best_val >= severity_threshold).astype(int)
    y_pred_binary_best_val = (severity_pred_ratio_best_val >= severity_threshold).astype(int)
    
    try:
        best_fold_metrics['f1'] = f1_score(y_true_binary_best_val, y_pred_binary_best_val, zero_division=0)
        best_fold_metrics['precision'] = precision_score(y_true_binary_best_val, y_pred_binary_best_val, zero_division=0)
        best_fold_metrics['recall'] = recall_score(y_true_binary_best_val, y_pred_binary_best_val, zero_division=0)
        best_fold_metrics['accuracy'] = accuracy_score(y_true_binary_best_val, y_pred_binary_best_val)
    except Exception as e:
        logger.warning(f"최적 Fold F1 Score 계산 실패: {e}")
        best_fold_metrics['f1'] = 0.0
        best_fold_metrics['precision'] = 0.0
        best_fold_metrics['recall'] = 0.0
        best_fold_metrics['accuracy'] = 0.0
    
    logger.info(f"MLP-SEVERITY 최적 Fold {best_fold} 검증 성능 - MAE: {best_fold_metrics['mae']:.6f}, RMSE: {best_fold_metrics['rmse']:.6f}")
    logger.info(f"MLP-SEVERITY 최적 Fold {best_fold} 검증 성능 - R²: {best_fold_metrics['r2']:.4f}")
    logger.info(f"MLP-SEVERITY 최적 Fold {best_fold} 절대액 - MAE: {best_fold_metrics['mae_absolute']:,.0f}원, RMSE: {best_fold_metrics['rmse_absolute']:,.0f}원")
    logger.info(f"MLP-SEVERITY 최적 Fold {best_fold} 분류 성능 - F1: {best_fold_metrics['f1']:.4f}, Precision: {best_fold_metrics['precision']:.4f}, Recall: {best_fold_metrics['recall']:.4f}, Accuracy: {best_fold_metrics['accuracy']:.4f}")
    
    # 전체 데이터 성능 (참고용)
    severity_metrics = {
        'mae': mean_absolute_error(y_severity, severity_pred_ratio),
        'rmse': np.sqrt(mean_squared_error(y_severity, severity_pred_ratio)),
        'r2': r2_score(y_severity, severity_pred_ratio),
        'mae_absolute': mean_absolute_error(y_severity * config['model']['face_value'], 
                                          severity_pred_ratio * config['model']['face_value']),
        'rmse_absolute': np.sqrt(mean_squared_error(y_severity * config['model']['face_value'], 
                                                   severity_pred_ratio * config['model']['face_value']))
    }
    
    logger.info(f"MLP-SEVERITY 전체 데이터 성능 (참고) - MAE: {severity_metrics['mae']:.6f}, RMSE: {severity_metrics['rmse']:.6f}")
    logger.info(f"MLP-SEVERITY 전체 데이터 성능 (참고) - R²: {severity_metrics['r2']:.4f}")
    logger.info(f"MLP-SEVERITY 전체 데이터 절대액 (참고) - MAE: {severity_metrics['mae_absolute']:,.0f}원, RMSE: {severity_metrics['rmse_absolute']:,.0f}원")
    
    # Two-Part 모델 통합 성능 평가 (트리거된 사건만)
    # 트리거된 사건에 대해서만 Two-Part 모델 성능 평가
    trigger_mask = y_trigger == 1
    if trigger_mask.sum() > 0:
        trigger_indices = np.where(trigger_mask)[0]
        severity_indices = np.arange(len(y_severity))
        
        # 트리거된 사건에 대해서만 예측
        trigger_pred_proba_triggered = trigger_pred_proba[trigger_indices]
        severity_pred_ratio_triggered = severity_pred_ratio
        
        # 실제 손실이 있는 사건만 필터링
        loss_mask = y_severity > 0
        if loss_mask.sum() > 0:
            loss_indices = np.where(loss_mask)[0]
            y_severity_loss = y_severity[loss_indices]
            severity_pred_ratio_loss = severity_pred_ratio[loss_indices]
            
            # 손실 모델 성능 (실제 손실이 있는 사건만)
            severity_loss_metrics = {
                'mae': mean_absolute_error(y_severity_loss, severity_pred_ratio_loss),
                'rmse': np.sqrt(mean_squared_error(y_severity_loss, severity_pred_ratio_loss)),
                'r2': r2_score(y_severity_loss, severity_pred_ratio_loss),
                'mae_absolute': mean_absolute_error(y_severity_loss * config['model']['face_value'], 
                                                  severity_pred_ratio_loss * config['model']['face_value']),
                'rmse_absolute': np.sqrt(mean_squared_error(y_severity_loss * config['model']['face_value'], 
                                                           severity_pred_ratio_loss * config['model']['face_value']))
            }
            
            # F1 Score 계산 (실제 손실이 있는 사건만)
            y_true_binary_loss = (y_severity_loss >= severity_threshold).astype(int)
            y_pred_binary_loss = (severity_pred_ratio_loss >= severity_threshold).astype(int)
            
            try:
                severity_loss_metrics['f1'] = f1_score(y_true_binary_loss, y_pred_binary_loss, zero_division=0)
                severity_loss_metrics['precision'] = precision_score(y_true_binary_loss, y_pred_binary_loss, zero_division=0)
                severity_loss_metrics['recall'] = recall_score(y_true_binary_loss, y_pred_binary_loss, zero_division=0)
                severity_loss_metrics['accuracy'] = accuracy_score(y_true_binary_loss, y_pred_binary_loss)
            except Exception as e:
                logger.warning(f"손실 사건 F1 Score 계산 실패: {e}")
                severity_loss_metrics['f1'] = 0.0
                severity_loss_metrics['precision'] = 0.0
                severity_loss_metrics['recall'] = 0.0
                severity_loss_metrics['accuracy'] = 0.0
            
            logger.info(f"MLP-SEVERITY-LOSS 성능 - MAE: {severity_loss_metrics['mae']:.6f}, RMSE: {severity_loss_metrics['rmse']:.6f}")
            logger.info(f"MLP-SEVERITY-LOSS 성능 - R²: {severity_loss_metrics['r2']:.4f}")
            logger.info(f"MLP-SEVERITY-LOSS 절대액 - MAE: {severity_loss_metrics['mae_absolute']:,.0f}원, RMSE: {severity_loss_metrics['rmse_absolute']:,.0f}원")
            logger.info(f"MLP-SEVERITY-LOSS 분류 성능 - F1: {severity_loss_metrics['f1']:.4f}, Precision: {severity_loss_metrics['precision']:.4f}, Recall: {severity_loss_metrics['recall']:.4f}, Accuracy: {severity_loss_metrics['accuracy']:.4f}")
    
    logger.info("TWO-PART 통합 모델 성능은 개별 모델 성능으로 평가")
    
    # 7. Cat Bond Expected Loss 계산
    el_calculator = ExpectedLossCalculator(config)
    
    # 충주 지역 기대 사건 수 (λ)
    lambda_hat_chj = two_part_model.frequency_model.get_lambda_hat()
    
    # 충주 지역 사건당 손실률 (L) - 전체 데이터 사용
    loss_info = two_part_model.get_conditional_loss(df_full, chungju_region_name)
    loss_rate = loss_info['loss_rate']
    
    # Cat Bond EL 계산: EL = F × λ × n × L
    el_results = el_calculator.calculate_el(lambda_hat_chj, loss_rate)
    
    # 인제군 기대손실 계산
    inje_el_results = el_calculator.calculate_el(lambda_hat_chj, inje_loss_info['loss_rate'])
    logger.info(f"인제군 기대손실: {inje_el_results['expected_loss']:,.0f}원 ({inje_el_results['expected_loss']/1e8:.1f}억원)")
    logger.info(f"인제군 기대 상환액: {inje_el_results['expected_repayment']:,.0f}원 ({inje_el_results['expected_repayment']/1e8:.1f}억원)")
    logger.info(f"인제군 기대 손실률: {inje_el_results['expected_loss_rate']:.6f}")
    
    # EL 신뢰도 평가 (부트스트랩)
    logger.info("EL 예측 신뢰도 평가 시작")
    lambda_samples, loss_rate_samples = bootstrap_parameters(
        df_full, chungju_region_name, n_bootstrap=1000
    )
    
    el_ci_results = el_calculator.calculate_el_confidence_interval(
        lambda_samples, loss_rate_samples, confidence_level=0.95
    )
    
    logger.info(f"EL-RELIABILITY 평균 EL: {el_ci_results['el_mean']:,.0f}원, 표준편차: {el_ci_results['el_std']:,.0f}원")
    logger.info(f"EL-RELIABILITY 변동계수: {el_ci_results['el_cv']:.4f}")
    logger.info(f"EL-RELIABILITY 95% 신뢰구간: [{el_ci_results['el_lower']:,.0f}, {el_ci_results['el_upper']:,.0f}]원")
    logger.info(f"EL-RELIABILITY 신뢰구간 폭: {el_ci_results['el_upper'] - el_ci_results['el_lower']:,.0f}원")
    logger.info(f"EL-RELIABILITY λ 평균: {el_ci_results['lambda_mean']:.6f} ± {el_ci_results['lambda_std']:.6f}")
    logger.info(f"EL-RELIABILITY L 평균: {el_ci_results['loss_rate_mean']:.6f} ± {el_ci_results['loss_rate_std']:.6f}")
    
    # 신뢰도 등급 평가
    cv = el_ci_results['el_cv']
    if cv < 0.1:
        reliability_grade = "매우 높음"
    elif cv < 0.2:
        reliability_grade = "높음"
    elif cv < 0.3:
        reliability_grade = "보통"
    elif cv < 0.5:
        reliability_grade = "낮음"
    else:
        reliability_grade = "매우 낮음"
    
    logger.info(f"EL-RELIABILITY 신뢰도 등급: {reliability_grade} (CV: {cv:.4f})")
    
    # 손실액 분포 분석
    logger.info("손실액 분포 분석 시작")
    
    # 실제 손실 데이터에서 통계 계산
    actual_losses = df_triggered[df_triggered['severity'] > 0]['severity'] * el_results['face_value']
    
    if len(actual_losses) > 0:
        logger.info(f"LOSS-STATS 실제 손실 사건 수: {len(actual_losses)}개")
        logger.info(f"LOSS-STATS 평균 손실액: {actual_losses.mean():,.0f}원, 중앙값: {actual_losses.median():,.0f}원")
        logger.info(f"LOSS-STATS 최소 손실액: {actual_losses.min():,.0f}원, 최대 손실액: {actual_losses.max():,.0f}원")
        logger.info(f"LOSS-STATS 손실액 표준편차: {actual_losses.std():,.0f}원, 변동계수: {actual_losses.std()/actual_losses.mean():.4f}")
        
        # 분위수별 손실액
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            loss_at_percentile = actual_losses.quantile(p/100)
            logger.info(f"LOSS-STATS {p}% 분위수 손실액: {loss_at_percentile:,.0f}원")
        
        # 충주 지역 손실 통계
        chungju_losses = df_triggered[
            (df_triggered['region'] == chungju_region_name) & 
            (df_triggered['severity'] > 0)
        ]['severity'] * el_results['face_value']
        
        if len(chungju_losses) > 0:
            logger.info(f"LOSS-STATS 충주 지역 손실 사건 수: {len(chungju_losses)}개")
            logger.info(f"LOSS-STATS 충주 지역 평균 손실액: {chungju_losses.mean():,.0f}원, 최대 손실액: {chungju_losses.max():,.0f}원")
            logger.info(f"LOSS-STATS 충주 지역 손실액 표준편차: {chungju_losses.std():,.0f}원")
        
        # 인제군 손실 통계
        inje_losses = df_triggered[
            (df_triggered['region'] == inje_region_name) & 
            (df_triggered['severity'] > 0)
        ]['severity'] * el_results['face_value']
        
        if len(inje_losses) > 0:
            logger.info(f"LOSS-STATS 인제군 손실 사건 수: {len(inje_losses)}개")
            logger.info(f"LOSS-STATS 인제군 평균 손실액: {inje_losses.mean():,.0f}원, 최대 손실액: {inje_losses.max():,.0f}원")
            logger.info(f"LOSS-STATS 인제군 손실액 표준편차: {inje_losses.std():,.0f}원")
        else:
            logger.info(f"LOSS-STATS 인제군 손실 사건: 없음")
    
    # 3년간 최소 1회 발생 확률 계산
    lambda_3yr = lambda_hat_chj * el_results['years']
    poisson_dist = poisson(lambda_3yr)
    prob_at_least_one = 1 - poisson_dist.pmf(0)
    prob_at_least_two = 1 - poisson_dist.pmf(0) - poisson_dist.pmf(1)
    
    # 쿠폰률 산정 (채권 발행자 관점)
    logger.info("쿠폰률 산정 시작")
    
    # 기본 파라미터
    risk_free_rate = 0.025  # 무위험 이자율 3%
    risk_premium = 0.02    # 위험 프리미엄 2%
    
    coupon_results = el_calculator.calculate_coupon_rate(
        expected_loss=el_results['expected_loss'],
        face_value=el_results['face_value'],
        years=el_results['years'],
        risk_free_rate=risk_free_rate,
        risk_premium=risk_premium
    )
    
    logger.info(f"COUPON 기본 쿠폰률: {coupon_results['base_coupon_rate']:.3%}, 재해 위험 프리미엄: {coupon_results['catastrophe_premium']:.3%}")
    logger.info(f"COUPON 총 쿠폰률: {coupon_results['total_coupon_rate']:.3%}")
    logger.info(f"COUPON 연간 쿠폰 지급액: {coupon_results['annual_coupon_payment']:,.0f}원, 3년간 총 쿠폰 지급액: {coupon_results['total_coupon_payments']:,.0f}원")
    logger.info(f"COUPON 기대손실: {coupon_results['expected_loss']:,.0f}원, 순이익 (쿠폰 - EL): {coupon_results['net_benefit']:,.0f}원")
    
    # 손실액 상세 분석
    logger.info(f"LOSS-DETAIL 연간 기대 손실액: {coupon_results['expected_loss']/el_results['years']:,.0f}원")
    logger.info(f"LOSS-DETAIL 월간 기대 손실액: {coupon_results['expected_loss']/(el_results['years']*12):,.0f}원")
    logger.info(f"LOSS-DETAIL 일일 기대 손실액: {coupon_results['expected_loss']/(el_results['years']*365):,.0f}원")
    logger.info(f"LOSS-DETAIL 액면가 대비 손실액 비율: {coupon_results['expected_loss']/el_results['face_value']:.6f}")
    logger.info(f"LOSS-DETAIL 쿠폰 대비 손실액 비율: {coupon_results['expected_loss']/coupon_results['total_coupon_payments']:.6f}")
    
    # 투자자 수익률 계산
    investor_results = el_calculator.calculate_investor_yield(
        coupon_rate=coupon_results['total_coupon_rate'],
        face_value=el_results['face_value'],
        years=el_results['years'],
        prob_at_least_one=prob_at_least_one,
        expected_loss_rate=el_results['expected_loss_rate']
    )
    
    logger.info(f"INVESTOR 연간 쿠폰 수익: {investor_results['annual_coupon']:,.0f}원, 기대 상환액: {investor_results['expected_repayment']:,.0f}원")
    logger.info(f"INVESTOR 총 기대 수익: {investor_results['total_expected_return']:,.0f}원, 총 수익률: {investor_results['total_return_rate']:.3%}")
    logger.info(f"INVESTOR 연평균 수익률: {investor_results['annual_return_rate']:.3%}, 위험 조정 수익률: {investor_results['risk_adjusted_return']:.3%}")
    
    # 보험료 추정 (기존 호환성)
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
        'mlp_severity_scaler': two_part_model.severity_model.scaler,  # MLP 스케일러 추가
        'face_value': config['model']['face_value'],  # cover_limit 대신 face_value 저장
        'config': config,
        'w': 0.5,
        'max_loss_rate': 1.0
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
        'inje_region_name': inje_region_name,
        'inje_threshold': inje_threshold,
        'inje_loss_info': inje_loss_info,
        'inje_el_results': inje_el_results,
        'config': config,
        # EL 신뢰도 정보 추가
        'el_reliability': el_ci_results,
        # 쿠폰률 정보 추가
        'coupon_results': coupon_results,
        'investor_results': investor_results,
        # 교차검증 정보 추가
        'severity_cv_summary': two_part_model.severity_model.get_cv_summary()
    }
    
    logger.info("Cat Bond 모델 학습 완료")
    logger.info(f"충주 지역: {chungju_region_name}, 임계값: {chungju_threshold:.1f}mm")
    logger.info(f"인제군: {inje_region_name}, 임계값: {inje_threshold:.1f}mm" if inje_threshold is not None else f"인제군: {inje_region_name}, 임계값: N/A")
    logger.info(f"연간 기대 사건 수 (λ): {lambda_hat_chj:.3f}건, 3년간 기대 사건 수: {lambda_3yr:.3f}건")
    logger.info(f"3년간 최소 1회 발생 확률: {prob_at_least_one:.3%}, 3년간 최소 2회 발생 확률: {prob_at_least_two:.3%}")
    logger.info(f"사건당 손실률 (L): {loss_rate:.6f}")
    if loss_rate > 1.0:
        logger.info(f"clipping된 손실률: {el_results['clipped_loss_rate']:.6f}")
    logger.info(f"기대 손실률: {el_results['expected_loss_rate']:.6f}")
    logger.info(f"액면가 (F): {el_results['face_value']:,.0f}원 ({el_results['face_value']/1e8:.1f}억원)")
    logger.info(f"기대손실 (EL): {el_results['expected_loss']:,.0f}원 ({el_results['expected_loss']/1e8:.1f}억원)")
    logger.info(f"기대 상환액: {el_results['expected_repayment']:,.0f}원 ({el_results['expected_repayment']/1e8:.1f}억원)")
    logger.info(f"추정 보험료: {estimated_premium:,.0f}원 ({estimated_premium/1e8:.1f}억원)")
    
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
    severity_model = joblib.load(model_paths['severity_model'])
    
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
    
    # 5. 가중 평균 방식 적용
    w = region_info.get('w', 0.5)
    max_loss_rate = region_info.get('max_loss_rate', 1.0)
    final_pred_ratio = w * severity_pred_ratio + (1 - w) * (trigger_proba * max_loss_rate)
    final_pred_absolute = final_pred_ratio * region_info['face_value']
    
    # 6. 결과 반환
    results = input_data.copy()
    results['trigger_proba'] = trigger_proba
    results['severity_pred_ratio'] = severity_pred_ratio
    results['final_pred_ratio'] = final_pred_ratio
    results['final_pred_absolute'] = final_pred_absolute
    
    return results

def bootstrap_parameters(df_full: pd.DataFrame, chungju_region_name: str, 
                        n_bootstrap: int = 1000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    λ와 L의 부트스트랩 샘플 생성
    
    Args:
        df_full: 전체 데이터
        chungju_region_name: 충주 지역명
        n_bootstrap: 부트스트랩 반복 횟수
        random_state: 랜덤 시드
        
    Returns:
        lambda_samples, loss_rate_samples
    """
    np.random.seed(random_state)
    
    # 충주 지역 손실 데이터
    chungju_mask = (df_full['region'] == chungju_region_name)
    chungju_loss_data = df_full[chungju_mask & (df_full['severity'] > 0)]
    
    # 전국 손실 데이터 (충주 데이터가 부족한 경우)
    national_loss_data = df_full[df_full['severity'] > 0]
    
    lambda_samples = []
    loss_rate_samples = []
    
    for _ in range(n_bootstrap):
        # λ 샘플링 (연간 기대 사건 수는 고정값 사용)
        lambda_sample = 0.05  # 고정값
        
        # L 샘플링 (손실률)
        if len(chungju_loss_data) >= 5:  # 충주 데이터가 충분한 경우
            # 충주 지역에서 부트스트랩 샘플링
            bootstrap_indices = np.random.choice(len(chungju_loss_data), 
                                               size=len(chungju_loss_data), 
                                               replace=True)
            loss_sample = chungju_loss_data.iloc[bootstrap_indices]['severity'].mean()
        else:
            # 전국 데이터에서 부트스트랩 샘플링
            bootstrap_indices = np.random.choice(len(national_loss_data), 
                                               size=len(national_loss_data), 
                                               replace=True)
            loss_sample = national_loss_data.iloc[bootstrap_indices]['severity'].mean()
        
        lambda_samples.append(lambda_sample)
        loss_rate_samples.append(loss_sample)
    
    return np.array(lambda_samples), np.array(loss_rate_samples) 