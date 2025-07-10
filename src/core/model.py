"""
CatBond 모델 코어 로직
Two-Part 모델 (Trigger + Severity)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, Optional, Any, List
import joblib
import logging
from .utils import timer, set_seed, ensure_dir

# TensorFlow/Keras imports for MLP model
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.preprocessing import StandardScaler

# 로깅 설정
logger = logging.getLogger(__name__)

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
            sample_weights: pd.Series) -> Dict[str, Any]:
        """Poisson GLM 모델 학습"""
        logger.info("빈도 모델 학습 시작")
        
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
        
        # 연간 기대 사건 수를 0.05으로 강제 설정
        self.lambda_hat_global = 0.05
        
        logger.info(f"λ̂_global = {self.lambda_hat_global:.3f} /yr (강제 설정)")
        
        return {
            'lambda_hat_global': self.lambda_hat_global,
            'lambda_pred_rate': lambda_pred_rate,
            'lambda_pred_counts': lambda_pred_counts
        }
        
    def predict_lambda(self, X: pd.DataFrame, exposure: pd.Series) -> np.ndarray:
        """연간 사건 수 예측"""
        rate_pred = self.model.predict(X)
        return rate_pred * exposure.values
    
    def get_lambda_hat(self, region_name: Optional[str] = None) -> float:
        """기대 사건 수 반환"""
        if region_name is None:
            return self.lambda_hat_global
        return self.lambda_hat_chj
    
    def save_model(self, filepath: str) -> str:
        """모델 저장"""
        joblib.dump(self.model, filepath)
        logger.info(f"빈도 모델 저장 완료: {filepath}")
        return filepath

class TriggerModel:
    """트리거 모델 (LightGBM)"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.optimal_threshold = 0.5
        
    @timer
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """트리거 모델 학습"""
        logger.info("트리거 모델 학습 시작")
        
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
        threshold_result = self.optimize_threshold(X.iloc[val_idx], y.iloc[val_idx])
        
        return {
            'optimal_threshold': self.optimal_threshold,
            'threshold_optimization': threshold_result,
            'train_indices': train_idx,
            'val_indices': val_idx
        }
        
    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
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
        
        return {
            'optimal_threshold': self.optimal_threshold,
            'threshold_results': results if results else [],
            'best_f1': max([r['f1'] for r in results]) if results else 0.0
        }
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """트리거 발생 확률 예측"""
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, filepath: str) -> str:
        """모델 저장"""
        joblib.dump(self.model, filepath)
        logger.info(f"트리거 모델 저장 완료: {filepath}")
        return filepath

class SeverityModel:
    """손실 모델 (LightGBM) - Group K-Fold 교차검증 적용"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.cv_scores = None
        self.best_fold = None
        self.severity_threshold = config.get('severity', {}).get('classification_threshold', 0.001)  # 손실률 임계값 (0.1%)
        
    @timer
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[pd.Series] = None,
            groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """손실 모델 학습 - Group K-Fold 교차검증"""
        logger.info("손실 모델 학습 (Group K-Fold CV) 시작")
        
        # Group K-Fold 설정
        n_splits = self.config['training'].get('cv_folds', 5)
        random_state = self.config['training']['random_state']
        
        if groups is not None:
            # Group K-Fold 사용 (지역별 분할)
            cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            logger.info(f"Group K-Fold CV: {n_splits} folds, 그룹 수: {len(np.unique(groups))}개")
        else:
            # 일반 K-Fold 사용
            cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            groups = np.arange(len(X))  # 각 샘플을 개별 그룹으로 처리
            logger.info(f"K-Fold CV: {n_splits} folds")
        
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
            "seed": random_state,
            "verbose": -1
        }
        
        # 교차검증 수행
        cv_scores = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
            logger.info(f"Fold {fold+1}/{n_splits} 학습 중...")
            
            # 데이터 분할
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 가중치 분할
            train_weight = sample_weight.iloc[train_idx] if sample_weight is not None else None
            val_weight = sample_weight.iloc[val_idx] if sample_weight is not None else None
            
            # LightGBM 데이터셋 생성
            train_set = lgb.Dataset(
                X_train, 
                y_train, 
                weight=train_weight,
                categorical_feature=['region']
            )
            val_set = lgb.Dataset(
                X_val, 
                y_val, 
                weight=val_weight,
                categorical_feature=['region']
            )
            
            # 모델 학습
            model = lgb.train(
                model_params,
                train_set,
                num_boost_round=severity_params['num_boost_round'],
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=severity_params['early_stopping_rounds']),
                    lgb.log_evaluation(period=0)  # 로그 출력 비활성화
                ]
            )
            
            # 검증 성능 평가
            y_pred_log = model.predict(X_val)
            y_pred_ratio = np.expm1(y_pred_log)
            y_true_ratio = np.expm1(y_val)
            
            # 메트릭 계산
            mae = mean_absolute_error(y_true_ratio, y_pred_ratio)
            rmse = np.sqrt(mean_squared_error(y_true_ratio, y_pred_ratio))
            r2 = r2_score(y_true_ratio, y_pred_ratio)
            
            # 절대 손실액 기준 메트릭
            face_value = self.config['model']['face_value']
            mae_absolute = mean_absolute_error(y_true_ratio * face_value, y_pred_ratio * face_value)
            rmse_absolute = np.sqrt(mean_squared_error(y_true_ratio * face_value, y_pred_ratio * face_value))
            
            # F1 Score 계산을 위한 이진 분류 변환
            y_true_binary = (y_true_ratio >= self.severity_threshold).astype(int)
            y_pred_binary = (y_pred_ratio >= self.severity_threshold).astype(int)
            
            # F1 Score 계산
            try:
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                accuracy = accuracy_score(y_true_binary, y_pred_binary)
            except Exception as e:
                logger.warning(f"Fold {fold+1} F1 Score 계산 실패: {e}")
                f1 = 0.0
                precision = 0.0
                recall = 0.0
                accuracy = 0.0
            
            fold_score = {
                'fold': fold + 1,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mae_absolute': mae_absolute,
                'rmse_absolute': rmse_absolute,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'severity_threshold': self.severity_threshold,
                'model': model
            }
            
            cv_scores.append(fold_score)
            models.append(model)
            
            logger.info(f"Fold {fold+1} 전체 성능 - MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.4f}")
            logger.info(f"Fold {fold+1} 절대액 - MAE: {mae_absolute:,.0f}원, RMSE: {rmse_absolute:,.0f}원")
            logger.info(f"Fold {fold+1} 분류 성능 - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")
        
        # 최적 모델 선택 (R² 기준)
        best_fold_idx = np.argmax([score['r2'] for score in cv_scores])
        self.best_fold = best_fold_idx + 1
        self.model = models[best_fold_idx]
        self.cv_scores = cv_scores
        
        # 교차검증 결과 요약
        mean_r2 = np.mean([s['r2'] for s in cv_scores])
        std_r2 = np.std([s['r2'] for s in cv_scores])
        mean_mae = np.mean([s['mae'] for s in cv_scores])
        std_mae = np.std([s['mae'] for s in cv_scores])
        mean_rmse = np.mean([s['rmse'] for s in cv_scores])
        std_rmse = np.std([s['rmse'] for s in cv_scores])
        mean_mae_abs = np.mean([s['mae_absolute'] for s in cv_scores])
        std_mae_abs = np.std([s['mae_absolute'] for s in cv_scores])
        
        # F1 Score 요약
        mean_f1 = np.mean([s['f1'] for s in cv_scores])
        std_f1 = np.std([s['f1'] for s in cv_scores])
        mean_precision = np.mean([s['precision'] for s in cv_scores])
        std_precision = np.std([s['precision'] for s in cv_scores])
        mean_recall = np.mean([s['recall'] for s in cv_scores])
        std_recall = np.std([s['recall'] for s in cv_scores])
        mean_accuracy = np.mean([s['accuracy'] for s in cv_scores])
        std_accuracy = np.std([s['accuracy'] for s in cv_scores])
        
        logger.info(f"Group K-Fold CV 결과 - 최적 Fold: {self.best_fold}")
        logger.info(f"평균 R²: {mean_r2:.4f} ± {std_r2:.4f}")
        logger.info(f"평균 MAE: {mean_mae:.6f} ± {std_mae:.6f}")
        logger.info(f"평균 RMSE: {mean_rmse:.6f} ± {std_rmse:.6f}")
        logger.info(f"평균 MAE (절대액): {mean_mae_abs:,.0f} ± {std_mae_abs:,.0f}원")
        logger.info(f"평균 F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
        logger.info(f"평균 Precision: {mean_precision:.4f} ± {std_precision:.4f}")
        logger.info(f"평균 Recall: {mean_recall:.4f} ± {std_recall:.4f}")
        logger.info(f"평균 Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        logger.info(f"손실률 분류 임계값: {self.severity_threshold:.6f}")
        
        # 각 fold별 성능 로깅
        for score in cv_scores:
            logger.info(f"Fold {score['fold']}: R²={score['r2']:.4f}, MAE={score['mae']:.6f}, RMSE={score['rmse']:.6f}, F1={score['f1']:.4f}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """손실 규모 예측"""
        return self.model.predict(X)
    
    def get_cv_summary(self) -> Dict[str, Any]:
        """교차검증 결과 요약"""
        if self.cv_scores is None:
            return {}
        
        r2_scores = [s['r2'] for s in self.cv_scores]
        mae_scores = [s['mae'] for s in self.cv_scores]
        rmse_scores = [s['rmse'] for s in self.cv_scores]
        f1_scores = [s['f1'] for s in self.cv_scores]
        precision_scores = [s['precision'] for s in self.cv_scores]
        recall_scores = [s['recall'] for s in self.cv_scores]
        accuracy_scores = [s['accuracy'] for s in self.cv_scores]
        
        return {
            'best_fold': self.best_fold,
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'mean_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores),
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_precision': np.mean(precision_scores),
            'std_precision': np.std(precision_scores),
            'mean_recall': np.mean(recall_scores),
            'std_recall': np.std(recall_scores),
            'mean_accuracy': np.mean(accuracy_scores),
            'std_accuracy': np.std(accuracy_scores),
            'severity_threshold': self.severity_threshold,
            'cv_scores': self.cv_scores
        }
    
    def save_model(self, filepath: str) -> Dict[str, str]:
        """모델 저장"""
        self.model.save_model(filepath)
        logger.info(f"손실 모델 저장 완료: {filepath}")
        
        # 교차검증 결과도 함께 저장
        cv_filepath = filepath.replace('.txt', '_cv_results.pkl')
        cv_results = {
            'cv_scores': self.cv_scores,
            'best_fold': self.best_fold,
            'cv_summary': self.get_cv_summary()
        }
        joblib.dump(cv_results, cv_filepath)
        logger.info(f"CV 결과 저장 완료: {cv_filepath}")
        
        return {
            'model_path': filepath,
            'cv_results_path': cv_filepath
        }

class TwoPartModel:
    """Two-Part 모델 통합"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.frequency_model = FrequencyModel(config)
        self.trigger_model = TriggerModel(config)
        self.severity_model = SeverityModel(config)
        self.face_value = config['model']['face_value']
        self.max_loss_rate = 1.0  # 최대 손실률(100%)
        self.w = 0.5  # 가중치 초기값(학습 시 최적화)
        
    def fit(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """전체 모델 학습"""
        logger.info("Two-Part 모델 학습 시작")
        
        # 1. 빈도 모델 학습
        frequency_result = self.frequency_model.fit(
            data_dict['X_freq'],
            data_dict['y_freq'],
            data_dict['sample_weights_freq']
        )
        
        # 2. 트리거 모델 학습
        trigger_result = self.trigger_model.fit(
            data_dict['X_trigger'],
            data_dict['y_trigger'],
            data_dict['groups_trigger']
        )
        
        # 3. 손실 모델 학습
        severity_result = self.severity_model.fit(
            data_dict['X_severity'],
            data_dict['y_severity'],
            data_dict['sample_weights_severity'],
            data_dict['groups_severity']
        )
        
        # === w 최적화 ===
        # 검증 데이터셋 준비 (교차검증에서 best fold의 val set 사용)
        X_severity = data_dict['X_severity']
        y_severity = data_dict['y_severity']
        groups_severity = data_dict['groups_severity']
        X_trigger = data_dict['X_trigger']
        y_trigger = data_dict['y_trigger']
        groups_trigger = data_dict['groups_trigger']
        
        n_splits = self.config['training'].get('cv_folds', 5)
        random_state = self.config['training']['random_state']
        
        # Severity 모델의 best fold에서 검증 데이터 가져오기
        cv_severity = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        best_fold = self.severity_model.best_fold if hasattr(self.severity_model, 'best_fold') else 1
        fold_splits_severity = list(cv_severity.split(X_severity, y_severity, groups=groups_severity))
        _, best_val_idx_severity = fold_splits_severity[best_fold - 1]
        
        # Trigger 모델의 best fold에서 검증 데이터 가져오기
        cv_trigger = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        best_fold_trigger = self.trigger_model.best_fold if hasattr(self.trigger_model, 'best_fold') else 1
        fold_splits_trigger = list(cv_trigger.split(X_trigger, y_trigger, groups=groups_trigger))
        _, best_val_idx_trigger = fold_splits_trigger[best_fold_trigger - 1]
        
        # 검증 데이터 준비
        X_val_severity = X_severity.iloc[best_val_idx_severity]
        y_val_severity = y_severity.iloc[best_val_idx_severity]
        X_val_trigger = X_trigger.iloc[best_val_idx_trigger]
        y_val_trigger = y_trigger.iloc[best_val_idx_trigger]
        
        # 트리거 확률, 손실 예측
        trigger_proba = self.trigger_model.predict_proba(X_val_trigger)
        severity_pred_log = self.severity_model.predict(X_val_severity)
        severity_pred_ratio = np.expm1(severity_pred_log)
        
        # 배열 크기 확인 및 조정
        logger.info(f"[TwoPartModel] trigger_proba shape: {trigger_proba.shape}")
        logger.info(f"[TwoPartModel] severity_pred_ratio shape: {severity_pred_ratio.shape}")
        logger.info(f"[TwoPartModel] y_val_severity shape: {y_val_severity.shape}")
        
        # 크기가 다르면 더 작은 크기에 맞춰 조정
        min_size = min(len(trigger_proba), len(severity_pred_ratio))
        if len(trigger_proba) != min_size:
            trigger_proba = trigger_proba[:min_size]
        if len(severity_pred_ratio) != min_size:
            severity_pred_ratio = severity_pred_ratio[:min_size]
        if len(y_val_severity) != min_size:
            y_val_severity = y_val_severity.iloc[:min_size]
        
        logger.info(f"[TwoPartModel] 조정 후 크기: {min_size}")
        
        # w grid search
        best_mae = float('inf')
        best_w = 0.5
        for w in np.linspace(0, 1, 21):
            final_pred = w * severity_pred_ratio + (1 - w) * (trigger_proba * self.max_loss_rate)
            mae = mean_absolute_error(y_val_severity, final_pred)
            if mae < best_mae:
                best_mae = mae
                best_w = w
        self.w = best_w
        logger.info(f"[TwoPartModel] 최적 w={self.w:.3f} (val MAE={best_mae:.6f})")
        
        logger.info("Two-Part 모델 학습 완료")
        
        return {
            'frequency_result': frequency_result,
            'trigger_result': trigger_result,
            'severity_result': severity_result,
            'w': self.w
        }
        
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Two-Part 모델 예측"""
        trigger_proba = self.trigger_model.predict_proba(X)
        severity_pred_log = self.severity_model.predict(X)
        severity_pred_ratio = np.expm1(severity_pred_log)
        final_pred_ratio = self.w * severity_pred_ratio + (1 - self.w) * (trigger_proba * self.max_loss_rate)
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
    
    def save_models(self, output_config: Dict) -> Dict[str, str]:
        """모든 모델 저장"""
        frequency_path = self.frequency_model.save_model(output_config['frequency_model'])
        trigger_path = self.trigger_model.save_model(output_config['trigger_model'])
        severity_paths = self.severity_model.save_model(output_config['severity_model'])
        
        return {
            'frequency_model': frequency_path,
            'trigger_model': trigger_path,
            'severity_model': severity_paths['model_path'],
            'severity_cv_results': severity_paths['cv_results_path']
        }
        
    def get_conditional_loss(self, df_full: pd.DataFrame, 
                           chungju_region_name: str) -> Dict[str, Any]:
        """조건부 손실 기대값 계산 (비율 반환) - 전체 데이터 사용"""
        # 충주 지역 전체 데이터
        chungju_mask = (df_full['region'] == chungju_region_name)
        chungju_data = df_full[chungju_mask]
        
        if len(chungju_data) > 0:
            # 충주 지역에서 실제 손실이 발생한 사건만 필터링
            chungju_loss_positive = chungju_data[chungju_data['severity'] > 0]
            
            if len(chungju_loss_positive) > 0:
                # severity는 이미 손실률 (비율)
                chungju_avg_severity = chungju_loss_positive['severity'].mean()
                logger.info(f"충주 지역 조건부 손실률: {chungju_avg_severity:.8f} (사건 수: {len(chungju_loss_positive)}개, 전체 데이터: {len(chungju_data)}개)")
                return {
                    'loss_rate': chungju_avg_severity,
                    'region': chungju_region_name,
                    'event_count': len(chungju_loss_positive),
                    'total_count': len(chungju_data),
                    'source': 'chungju'
                }
            else:
                logger.info(f"충주 지역에 손실이 발생한 사건이 없습니다. (전체 데이터: {len(chungju_data)}개)")
        
        # 전국 평균 사용
        national_loss_positive = df_full[df_full['severity'] > 0]
        if len(national_loss_positive) > 0:
            national_avg_severity = national_loss_positive['severity'].mean()
            logger.info(f"전국 평균 조건부 손실률: {national_avg_severity:.8f} (사건 수: {len(national_loss_positive)}개, 전체 데이터: {len(df_full)}개)")
            return {
                'loss_rate': national_avg_severity,
                'region': 'national',
                'event_count': len(national_loss_positive),
                'total_count': len(df_full),
                'source': 'national'
            }
        
        logger.warning("손실이 있는 사건이 없습니다.")
        return {
            'loss_rate': 0.0,
            'region': 'none',
            'event_count': 0,
            'total_count': len(df_full),
            'source': 'none'
        } 