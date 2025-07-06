#!/usr/bin/env python3
"""
CatBond 모델 학습 스크립트
python train.py --config config.yaml
"""

import argparse
import sys
import os
import logging

# src 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.pipeline import train_catbond_pipeline
from src.core.utils import setup_logging

# 로깅 설정
logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description='CatBond 모델 학습')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='설정 파일 경로 (기본값: config.yaml)')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='모델 저장 디렉토리 (기본값: models)')
    
    args = parser.parse_args()
    
    # 설정 파일 존재 확인
    if not os.path.exists(args.config):
        logger.error(f"설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)
    
    try:
        # 모델 학습 실행
        results = train_catbond_pipeline(args.config)
        
        logger.info("="*50)
        logger.info("학습 완료!")
        logger.info("="*50)
        logger.info(f"충주 지역: {results['chungju_region_name']}")
        logger.info(f"임계값: {results['chungju_threshold']:.1f}mm")
        logger.info(f"연간 기대 사건 수: {results['lambda_hat_chj']:.3f}건")
        logger.info(f"사건당 손실률: {results['loss_rate']:.6f}")
        logger.info(f"액면가: {results['face_value']:,.0f}원 ({results['face_value']/1e8:.1f}억원)")
        logger.info(f"기대 손실률: {results['expected_loss_rate']:.6f}")
        logger.info(f"기대손실: {results['expected_loss']:,.0f}원 ({results['expected_loss']/1e8:.1f}억원)")
        logger.info(f"추정 보험료: {results['estimated_premium']:,.0f}원 ({results['estimated_premium']/1e8:.1f}억원)")
        logger.info(f"3년간 최소 1회 발생 확률: {results['prob_at_least_one_3y']:.3%}")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 