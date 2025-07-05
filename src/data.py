"""
CatBond 모델 데이터 로딩 및 전처리
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .utils import timer, print_data_info, validate_data, find_chungju_region

class CatBondDataLoader:
    """CatBond 모델 데이터 로더"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.face_value = config['model']['face_value']  # cover_limit 대신 face_value 사용
        self.quantile_level = config['model']['quantile_level']
        
    @timer
    def load_rain_data(self) -> pd.DataFrame:
        """전국 강수 데이터 로드"""
        rain_file = self.config['data']['rain_file']
        df_rain = pd.read_parquet(f"{self.data_dir}{rain_file}")
        df_rain = df_rain.reset_index()
        df_rain['date'] = pd.to_datetime(df_rain['date'])
        df_rain["year"] = df_rain['date'].dt.year
        
        print_data_info(df_rain, "강수 데이터")
        return df_rain
    
    @timer
    def load_loss_data(self) -> pd.DataFrame:
        """손실 및 피처 데이터 로드"""
        loss_file = self.config['data']['loss_file']
        
        # 손실 데이터 로드 (CSV 파일 - 이미 rain_mm 포함)
        df = pd.read_csv(f"{self.data_dir}{loss_file}")
        df['date'] = pd.to_datetime(df['date'])
        df['daily_loss'] = df['daily_loss'].fillna(0)
        
        # daily_loss를 원 단위로 변환 (천원 단위에서 원 단위로)
        df['daily_loss'] = df['daily_loss'] * 1000
        
        # 필요한 피처 컬럼들이 없는 경우 기본값 설정
        required_features = ['rain_3d_sum', 'rain_7d_sum', 'rain_max_last30d', 
                           'imperv_area_m2', 'risk_index', 'pop_density_km2', 
                           'grdp', 'sewer_length_m', 'gov_spend']
        
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0  # 기본값 0으로 설정
        
        print_data_info(df, "병합된 손실 데이터")
        
        return df
    
    @timer
    def load_climate_data(self) -> pd.DataFrame:
        """기후지수 데이터 로드"""
        climate_file = self.config['data']['climate_file']
        clim = pd.read_csv(f"{self.data_dir}{climate_file}")
        clim = clim.rename(columns={"ENSP":"ENSO"}) if "ENSP" in clim.columns else clim
        
        print_data_info(clim, "기후지수 데이터")
        return clim
    
    @timer
    def calculate_thresholds(self, df_rain: pd.DataFrame) -> Dict[str, float]:
        """지역별 임계값 계산"""
        print(f"[INFO] 확률-기준 임계값 계산:")
        print(f"  • 분위수 레벨 = {self.quantile_level:.3f}")
        
        region_thresholds = (
            df_rain
              .groupby('region')['rain_mm']
              .quantile(self.quantile_level, interpolation='nearest')
              .to_dict()
        )
        
        # 처음 10개만 출력
        for region, threshold in list(region_thresholds.items())[:10]:
            print(f"  • {region}: {threshold:.1f}mm")
        print(f"  • ... (총 {len(region_thresholds)}개 지역)")
        
        return region_thresholds
    
    @timer
    def prepare_frequency_data(self, df_rain: pd.DataFrame, 
                             region_thresholds: Dict[str, float],
                             clim: pd.DataFrame) -> pd.DataFrame:
        """빈도 모델용 데이터 준비"""
        print("[INFO] 빈도 모델 데이터 준비...")
        
        # 지역별 초과 이벤트 계산
        events_by_region = {}
        for region in df_rain['region'].unique():
            region_data = df_rain[df_rain['region'] == region].copy()
            threshold = region_thresholds[region]
            
            exceed = region_data['rain_mm'] >= threshold
            events = exceed.groupby(region_data['year']).sum().reset_index()
            
            if events.shape[1] >= 2:
                events.columns = ['year', 'N_y']
            
            # 0건 연도 보존
            all_years = pd.DataFrame({
                'year': np.arange(region_data['year'].min(), region_data['year'].max()+1)
            })
            events = all_years.merge(events, on='year', how='left').fillna(0)
            events_by_region[region] = events
        
        # 다지역 빈도 데이터 생성
        freq_df = (pd.concat(events_by_region, names=['region'])
                     .reset_index()
                     .merge(clim, on='year'))
        
        print_data_info(freq_df, "빈도 모델 데이터")
        return freq_df
    
    @timer
    def prepare_trigger_data(self, df: pd.DataFrame,
                           region_thresholds: Dict[str, float]) -> pd.DataFrame:
        """트리거 모델용 데이터 준비"""
        print("[INFO] 트리거 모델 데이터 준비...")
        
        # 데이터 복사 및 지역 카테고리 변환
        df_full = df.copy()
        df_full['region'] = df_full['region'].astype('category')
        
        # 결측값 처리
        climate_features_to_fill = ['rain_mm', 'rain_3d_sum', 'rain_7d_sum', 'rain_max_last30d']
        for feature in climate_features_to_fill:
            if feature in df_full.columns:
                df_full[feature] = df_full[feature].fillna(0)
        
        df_full['daily_loss'] = df_full['daily_loss'].fillna(0)
        
        # 날짜 파생
        df_full["month"] = df_full["date"].dt.month
        df_full["month_sin"] = np.sin(2*np.pi*df_full["month"]/12)
        df_full["month_cos"] = np.cos(2*np.pi*df_full["month"]/12)
        
        # 임계값 적용
        df_full['threshold'] = df_full['region'].map(region_thresholds)
        df_full['trigger_flag'] = (df_full['rain_mm'] >= df_full['threshold']).astype(int)
        
        # 지역·날짜 순으로 정렬하여 시간 순서 보장
        df_full.sort_values(['region', 'date'], inplace=True)
        df_full.reset_index(drop=True, inplace=True)
        
        print_data_info(df_full, "트리거 모델 데이터")
        return df_full
    
    @timer
    def prepare_severity_data(self, df_triggered: pd.DataFrame) -> pd.DataFrame:
        """손실 모델용 데이터 준비"""
        print("[INFO] 손실 모델 데이터 준비...")
        
        # severity를 손실률로 계산 (daily_loss / face_value)
        df_triggered["severity"] = df_triggered["daily_loss"] / self.face_value
        
        # 소규모 손실 검증 및 로깅
        positive_loss_count = (df_triggered['daily_loss'] > 0).sum()
        positive_severity_count = (df_triggered['severity'] > 0).sum()
        
        print(f"[INFO] 손실 데이터 검증:")
        print(f"  • daily_loss > 0인 사건: {positive_loss_count}개")
        print(f"  • severity > 0인 사건: {positive_severity_count}개")
        print(f"  • 액면가 (F): {self.face_value:,.0f} 원 ({self.face_value/1e8:.1f}억원)")
        
        # 소규모 손실 사례 확인
        small_losses = df_triggered[(df_triggered['daily_loss'] > 0) & (df_triggered['daily_loss'] < 1e9)]  # 10억원 미만
        if len(small_losses) > 0:
            print(f"  • 소규모 손실 사례 (10억원 미만): {len(small_losses)}개")
            print(f"  • 최소 손실: {small_losses['daily_loss'].min():,.0f} 원")
            print(f"  • 최소 severity: {small_losses['severity'].min():.8f}")
        
        # 로그 변환
        df_triggered["severity_log"] = np.log1p(df_triggered["severity"])
        
        # 이상치 처리 제거 (클리핑하지 않음) - 소규모 손실 보존
        # outlier_quantile = self.config['training']['outlier_quantile']
        # q999 = df_triggered["severity_log"].quantile(outlier_quantile)
        # outlier_mask = df_triggered["severity_log"] > q999
        # if outlier_mask.sum() > 0:
        #     print(f"[INFO] {outlier_mask.sum()}개 이상치를 {q999:.6f}로 클리핑")
        #     df_triggered.loc[outlier_mask, "severity_log"] = q999
        
        print_data_info(df_triggered, "손실 모델 데이터")
        return df_triggered
    
    def get_chungju_info(self, df_rain: pd.DataFrame, 
                        region_thresholds: Dict[str, float]) -> Tuple[str, float]:
        """충주 지역 정보 반환"""
        chungju_region_name = find_chungju_region(df_rain['region'].unique())
        chungju_threshold = region_thresholds.get(chungju_region_name, 225.0)
        
        print(f"[INFO] 충주 지역명: {chungju_region_name}")
        print(f"[INFO] 충주 임계값: {chungju_threshold:.1f}mm")
        
        return chungju_region_name, chungju_threshold 