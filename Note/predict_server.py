from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = FastAPI(title="Harvest Slot ML Service")

# 1. 모델 및 정보 설정
MODEL_PATH = "model.joblib"
MODEL_VERSION = "rf-apple-harvest-v1"
FEATURES = ['mar_avg_temp', 'aug_sunshine', 'oct_rainfall', 'aug_humidity']

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ 모델 로드 성공: {MODEL_VERSION}")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")

# 2. 입력 데이터 규격 (백엔드 DB와 Flutter 앱 규격에 맞춤)
class PredictionRequest(BaseModel):
    farm_id: int
    product_id: int
    past_yield_kg: float      # 사용자의 과거 평년 수확량
    market_price: float       # KAMIS 기준 시장가
    variety: str              # 품종 (부사 / 황옥)
    mar_avg_temp: float       # 3월 평균 기온 (℃)
    aug_sunshine: float       # 8월 일조시간 (hr)
    oct_rainfall: float       # 10월 강수량 (mm)
    aug_humidity: float       # 8월 평균 습도 (%)

@app.post("/api/v1/owner/ml/predictions")
async def predict_harvest(request: PredictionRequest):
    try:
        # [A] 모델 입력 데이터 생성 (피처 순서 강제 엄수)
        input_data = pd.DataFrame([[
            request.mar_avg_temp,
            request.aug_sunshine,
            request.oct_rainfall,
            request.aug_humidity
        ]], columns=FEATURES)

        # [B] 수확량 단수 예측 (kg/10a)
        unit_yield = model.predict(input_data)[0]

        # [C] 비즈니스 로직 적용 (단수 -> 농장 맞춤 수확량 및 가격)
        # 1. 수확량: 기준 단수(1500) 대비 예측 비율을 과거 수확량에 적용
        variety_weight = 1.1 if request.variety == "부사" else 1.0
        estimated_yield = (unit_yield / 1500.0) * request.past_yield_kg * variety_weight
        
        # 2. 예약 가능 범위 (안전 계수 적용)
        safety_factor = 0.8 if request.oct_rainfall < 200 else 0.6
        max_reservable = estimated_yield * safety_factor
        min_reservable = estimated_yield * 0.4
        
        # 3. 수확 시기 (가상 로직: 3월 기온에 따라 변동)
        harvest_start = datetime(2024, 10, 20) - timedelta(days=int((request.mar_avg_temp - 10) * 2))
        harvest_end = harvest_start + timedelta(days=14)

        # [D] 결과 반환 (백엔드 DB 컬럼 구조에 100% 매칭)
        return {
            "data": {
                "prediction_id": 1,  # DB에서 자동 생성될 값이므로 예시 전달
                "farm_id": request.farm_id,
                "product_id": request.product_id,
                "input_feature_json": request.dict(),
                "estimated_yield_kg": round(estimated_yield, 2),
                "suggested_reservable_min_kg": round(min_reservable, 2),
                "suggested_reservable_max_kg": round(max_reservable, 2),
                "recommended_price": int(request.market_price * variety_weight),
                "predicted_harvest_start": harvest_start.strftime("%Y-%m-%d"),
                "predicted_harvest_end": harvest_end.strftime("%Y-%m-%d"),
                "confidence": 0.78,
                "safety_factor": safety_factor,
                "warning_message": "정상" if request.oct_rainfall < 200 else "수확기 강수량 주의 필요",
                "model_version": MODEL_VERSION
            },
            "message": "success",
            "error": None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)