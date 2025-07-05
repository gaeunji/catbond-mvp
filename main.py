#!/usr/bin/env python3
"""
Catastrophe Bond Risk Modeling MVP
Main execution script
"""

import os
import sys
from pathlib import Path

def main():
    """Main execution function"""
    print("🚀 Catastrophe Bond Risk Modeling MVP")
    print("=" * 50)
    
    # Check if required data files exist
    data_dir = Path("data/processed")
    required_files = [
        "rain_nation_daily_glm.parquet",
        "features.parquet", 
        "merged_flood_data.csv",
        "climate_idx.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📋 Please ensure all data files are in data/processed/ directory")
        return 1
    
    print("✅ All required data files found")
    
    # Run the main pipeline
    try:
        print("\n🔧 Running CatBond risk modeling pipeline...")
        # 두 가지 옵션 제공
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--legacy":
            # 레거시 파이프라인 사용
            from src.models.train_catbond_pipeline import main as run_pipeline
            run_pipeline()
        else:
            # 새로운 모듈화된 파이프라인 사용
            from src.pipeline import train_catbond_pipeline
            results = train_catbond_pipeline()
            print(f"\n📊 Pipeline Results:")
            print(f"  • Expected Loss: {results['expected_loss']:,.0f} 원")
            print(f"  • Estimated Premium: {results['estimated_premium']:,.0f} 원")
        
        print("\n✅ Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
