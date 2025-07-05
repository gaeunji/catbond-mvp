"""
CatBond 모델 코어 로직
Two-Part 모델 (Trigger + Severity)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, Optional, Any
import joblib
from .utils import timer

class GroupTimeSeriesSplit:
    """지역별 시간순 분할"""
    
    def __init__(self, n_splits=1, test_size=0.2, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
    
    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups 파라미터가 필요합니다")
        
        unique_groups = np.unique(groups)
        
        for _ in range(self.n_splits):
            train_indices = []
            val_indices = []
            
            for group in unique_groups:
                group_mask = groups == group
                group_indices = np.where(group_mask)[0]
                
                if len(group_indices) > 1:
                    # 데이터가 이미 지역·날짜 순으로 정렬되어 있으므로 인덱스 순서가 시간 순서와 일치
                    # 시간 순서를 보장하기 위해 인덱스 순서대로 분할
                    split_point = int(len(group_indices) * (1 - self.test_size))
                    train_indices.extend(group_indices[:split_point])
                    val_indices.extend(group_indices[split_point:])
                else:
                    train_indices.extend(group_indices)
            
            yield np.array(train_indices), np.array(val_indices)

class FrequencyModel:
    """빈도 모델 (Poisson GLM)"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.lambda_hat_global = None
        self.lambda_hat_chj = None
        
    @timer
    def fit(self, X_multi: pd.DataFrame, y_rate: pd.Series, 
            sample_weights: pd.Series) -> None:
        """Poisson GLM 모델 학습"""
        print("[INFO] 빈도 모델 학습...")
        
        freq_params = self.config['frequency']
        self.model = PoissonRegressor(
            alpha=freq_params['alpha'],
            max_iter=freq_params['max_iter'],
            fit_intercept=freq_params['fit_intercept']
        )
        
        self.model.fit(X_multi, y_rate, sample_weight=sample_weights)
        
        # 예측값 계산
        lambda_pred_rate = self.model.predict(X_multi)
        lambda_pred_counts = lambda_pred_rate * sample_weights.values
        self.lambda_hat_global = lambda_pred_counts.mean()
        
        # 연간 기대 사건 수를 0.03으로 강제 설정
        self.lambda_hat_global = 0.03
        
        print(f"[FREQUENCY] λ̂_global = {self.lambda_hat_global:.3f} /yr (강제 설정)")
        
    def predict_lambda(self, X: pd.DataFrame, exposure: pd.Series) -> np.ndarray:
        """연간 사건 수 예측"""
        rate_pred = self.model.predict(X)
        return rate_pred * exposure.values
    
    def get_lambda_hat(self, region_name: Optional[str] = None) -> float:
        """기대 사건 수 반환"""
        if region_name is None:
            return self.lambda_hat_global
        return self.lambda_hat_chj
    
    def save_model(self, filepath: str) -> None:
        """모델 저장"""
        joblib.dump(self.model, filepath)
        print(f"[SAVE] 빈도 모델: {filepath}")

class TriggerModel:
    """트리거 모델 (LightGBM)"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.optimal_threshold = 0.5
        
    @timer
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            groups: Optional[np.ndarray] = None) -> None:
        """트리거 모델 학습"""
        print("[INFO] 트리거 모델 학습...")
        
        # 데이터 분할
        if groups is not None:
            tscv = GroupTimeSeriesSplit(
                n_splits=self.config['training']['n_splits'],
                test_size=self.config['training']['test_size']
            )
            train_idx, val_idx = list(tscv.split(X, groups=groups))[0]
        else:
            tscv = TimeSeriesSplit(n_splits=5)
            train_idx, val_idx = list(tscv.split(X))[-1]
        
        # 클래스 가중치 계산
        class_counts = y.iloc[train_idx].value_counts()
        class_weight_ratio = class_counts[0] / class_counts[1] if 1 in class_counts else 10.0
        class_weight = {0: 1.0, 1: min(class_weight_ratio, self.config['trigger']['class_weight_max'])}
        
        # 모델 파라미터
        trigger_params = self.config['trigger']
        model_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": trigger_params['learning_rate'],
            "num_leaves": trigger_params['num_leaves'],
            "feature_fraction": trigger_params['feature_fraction'],
            "bagging_fraction": trigger_params['bagging_fraction'],
            "bagging_freq": trigger_params['bagging_freq'],
            "lambda_l2": trigger_params['lambda_l2'],
            "lambda_l1": trigger_params['lambda_l1'],
            "min_child_samples": trigger_params['min_child_samples'],
            "min_child_weight": trigger_params['min_child_weight'],
            "max_depth": trigger_params['max_depth'],
            "class_weight": class_weight,
            "seed": self.config['training']['random_state'],
            "verbose": -1
        }
        
        self.model = lgb.LGBMClassifier(**model_params)
        
        # 모델 학습
        self.model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            categorical_feature=['region'],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=trigger_params['early_stopping_rounds']),
                lgb.log_evaluation(period=150)
            ]
        )
        
        # 임계값 최적화
        self.optimize_threshold(X.iloc[val_idx], y.iloc[val_idx])
        
    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """임계값 최적화"""
        trigger_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for threshold in thresholds:
            pred = (trigger_pred_proba >= threshold).astype(int)
            
            try:
                precision = precision_score(y_val, pred, zero_division=0)
                recall = recall_score(y_val, pred, zero_division=0)
                f1 = f1_score(y_val, pred, zero_division=0)
                
                results.append({
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            except:
                continue
        
        if results:
            results_df = pd.DataFrame(results)
            recall_threshold = self.config['training']['recall_threshold']
            recall_filtered = results_df[results_df['recall'] >= recall_threshold]
            
            if len(recall_filtered) > 0:
                best_idx = recall_filtered['f1'].idxmax()
                self.optimal_threshold = recall_filtered.loc[best_idx, 'threshold']
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """트리거 발생 확률 예측"""
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, filepath: str) -> None:
        """모델 저장"""
        joblib.dump(self.model, filepath)
        print(f"[SAVE] 트리거 모델: {filepath}")

class SeverityModel:
    """손실 모델 (LightGBM)"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = None
        
    @timer
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[pd.Series] = None,
            groups: Optional[np.ndarray] = None) -> None:
        """손실 모델 학습"""
        print("[INFO] 손실 모델 학습...")
        
        # 데이터 분할
        if groups is not None:
            tscv = GroupTimeSeriesSplit(
                n_splits=self.config['training']['n_splits'],
                test_size=self.config['training']['test_size']
            )
            train_idx, val_idx = list(tscv.split(X, groups=groups))[0]
        else:
            tscv = TimeSeriesSplit(n_splits=5)
            train_idx, val_idx = list(tscv.split(X))[-1]
        
        # 모델 파라미터
        severity_params = self.config['severity']
        model_params = {
            "objective": "regression",
            "metric": "mse",
            "learning_rate": severity_params['learning_rate'],
            "num_leaves": severity_params['num_leaves'],
            "feature_fraction": severity_params['feature_fraction'],
            "bagging_fraction": severity_params['bagging_fraction'],
            "bagging_freq": severity_params['bagging_freq'],
            "lambda_l2": severity_params['lambda_l2'],
            "lambda_l1": severity_params['lambda_l1'],
            "min_child_samples": severity_params['min_child_samples'],
            "min_child_weight": severity_params['min_child_weight'],
            "max_depth": severity_params['max_depth'],
            "seed": self.config['training']['random_state'],
            "verbose": -1
        }
        
        # LightGBM 데이터셋 생성
        train_weight = sample_weight.iloc[train_idx] if sample_weight is not None else None
        val_weight = sample_weight.iloc[val_idx] if sample_weight is not None else None
        
        train_set = lgb.Dataset(
            X.iloc[train_idx], 
            y.iloc[train_idx], 
            weight=train_weight,
            categorical_feature=['region']
        )
        val_set = lgb.Dataset(
            X.iloc[val_idx], 
            y.iloc[val_idx], 
            weight=val_weight,
            categorical_feature=['region']
        )
        
        # 모델 학습
        self.model = lgb.train(
            model_params,
            train_set,
            num_boost_round=severity_params['num_boost_round'],
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=severity_params['early_stopping_rounds']),
                lgb.log_evaluation(period=150)
            ]
        )
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """손실 규모 예측"""
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """모델 저장"""
        self.model.save_model(filepath)
        print(f"[SAVE] 손실 모델: {filepath}")

class TwoPartModel:
    """Two-Part 모델 통합"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.frequency_model = FrequencyModel(config)
        self.trigger_model = TriggerModel(config)
        self.severity_model = SeverityModel(config)
        self.face_value = config['model']['face_value']  # cover_limit 대신 face_value 사용
        
    def fit(self, data_dict: Dict[str, Any]) -> None:
        """전체 모델 학습"""
        print("[INFO] Two-Part 모델 학습 시작...")
        
        # 1. 빈도 모델 학습
        self.frequency_model.fit(
            data_dict['X_freq'],
            data_dict['y_freq'],
            data_dict['sample_weights_freq']
        )
        
        # 2. 트리거 모델 학습
        self.trigger_model.fit(
            data_dict['X_trigger'],
            data_dict['y_trigger'],
            data_dict['groups_trigger']
        )
        
        # 3. 손실 모델 학습
        self.severity_model.fit(
            data_dict['X_severity'],
            data_dict['y_severity'],
            data_dict['sample_weights_severity'],
            data_dict['groups_severity']
        )
        
        print("[INFO] Two-Part 모델 학습 완료")
        
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Two-Part 모델 예측"""
        # 트리거 확률
        trigger_proba = self.trigger_model.predict_proba(X)
        
        # 손실 규모 (로그 변환된 값)
        severity_pred_log = self.severity_model.predict(X)
        severity_pred_ratio = np.expm1(severity_pred_log)
        
        # 최종 예측
        final_pred_ratio = trigger_proba * severity_pred_ratio
        
        return final_pred_ratio, trigger_proba, severity_pred_ratio
    
    def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """모델 성능 평가"""
        final_pred, trigger_proba, severity_pred = self.predict(X_val)
        
        # 절대손실액으로 변환
        y_true_absolute = y_val * self.face_value
        y_pred_absolute = final_pred * self.face_value
        
        metrics = {
            'mae': mean_absolute_error(y_true_absolute, y_pred_absolute),
            'rmse': np.sqrt(mean_squared_error(y_true_absolute, y_pred_absolute)),
            'r2': r2_score(y_true_absolute, y_pred_absolute),
            'trigger_proba_mean': trigger_proba.mean(),
            'severity_pred_mean': severity_pred.mean()
        }
        
        return metrics
    
    def save_models(self, output_config: Dict) -> None:
        """모든 모델 저장"""
        self.frequency_model.save_model(output_config['frequency_model'])
        self.trigger_model.save_model(output_config['trigger_model'])
        self.severity_model.save_model(output_config['severity_model'])
        
    def get_conditional_loss(self, df_triggered: pd.DataFrame, 
                           chungju_region_name: str) -> float:
        """조건부 손실 기대값 계산 (비율 반환)"""
        chungju_mask = (df_triggered['region'] == chungju_region_name)
        if chungju_mask.sum() > 0:
            chungju_loss_positive = df_triggered[
                (chungju_mask) & (df_triggered['severity'] > 0)
            ]
            if len(chungju_loss_positive) > 0:
                # severity는 이미 손실률 (비율)
                chungju_avg_severity = chungju_loss_positive['severity'].mean()
                print(f"[INFO] 충주 지역 조건부 손실률: {chungju_avg_severity:.8f} (사건 수: {len(chungju_loss_positive)}개)")
                return chungju_avg_severity
        
        # 전국 평균 사용
        national_loss_positive = df_triggered[df_triggered['severity'] > 0]
        if len(national_loss_positive) > 0:
            national_avg_severity = national_loss_positive['severity'].mean()
            print(f"[INFO] 전국 평균 조건부 손실률: {national_avg_severity:.8f} (사건 수: {len(national_loss_positive)}개)")
            return national_avg_severity
        
        print("[WARNING] 손실이 있는 사건이 없습니다.")
        return 0.0 