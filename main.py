from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from backend2 import ClinicalAI

app = FastAPI(title="XAI Elite Clinical Backend")

# ENABLE CORS FOR PRODUCTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel URL once deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize XGBoost & SHAP engine
ai_engine = ClinicalAI()

class PatientData(BaseModel):
    vitals: dict
    symptoms: dict = {}
    userId: str = None
    appId: str = "xai-pro-elite-v5"

@app.get("/")
def root():
    return {"status": "XAI Elite Online", "engine": "XGBoost + SHAP Integrated"}

@app.post("/analyze")
async def analyze_patient(data: PatientData):
    try:
        # Calculate Risk Probability and SHAP Values
        results = ai_engine.predict_with_xai(data.vitals, data.symptoms)
        return results
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Clinical Logic Failure")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
