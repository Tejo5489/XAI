from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend2 import ClinicalAI, CloudManager
import uvicorn
import os

app = FastAPI(title="XAI Sentinel AI Server")

# Enable CORS for the React Website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engines
ai_engine = ClinicalAI()
# Note: CloudManager requires service_account.json to exist
try:
    cloud_engine = CloudManager()
except Exception as e:
    print(f"Cloud Warning: {e}. Check service_account.json")
    cloud_engine = None

class AssessmentRequest(BaseModel):
    userId: str
    appId: str
    vitals: dict
    symptoms: dict

@app.get("/")
async def health():
    return {"status": "online", "engine": "XGBoost + SHAP active"}

@app.post("/analyze")
async def analyze_patient(req: AssessmentRequest):
    try:
        # 1. Run AI Inference
        results = ai_engine.predict_with_xai(req.vitals, req.symptoms)
        
        # 2. Sync to Cloud (If configured)
        if cloud_engine:
            cloud_engine.sync_assessment(req.userId, req.appId, results)
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
