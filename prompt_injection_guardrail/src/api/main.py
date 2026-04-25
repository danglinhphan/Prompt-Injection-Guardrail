from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import torch
import os
import traceback

from src.models.tfidf_model import TfidfClassifier
from src.mitigation.engine import PolicyEngine
from src.core.constants import CATEGORIES
from src.explainers.ig_explainer import DeepExplainer
from src.core.preprocessing import normalize_text

app = FastAPI(
    title="Explainable Prompt Injection Guardrail API",
    description="SOTA Dual-tier detection with XAI attribution"
)

# Initialize components lazily
tfidf_classifier = TfidfClassifier()
policy_engine = PolicyEngine()
bert_explainer = None

class PromptRequest(BaseModel):
    text: str
    model_tier: Optional[str] = "tier1" # tier1 or tier2

class GuardrailResponse(BaseModel):
    text: str
    label: str
    confidence: float
    action: str
    explanation: Optional[Dict[str, Any]] = None

@app.post("/check", response_model=GuardrailResponse)
async def check_prompt(request: PromptRequest):
    global bert_explainer
    
    # Preprocessing (Adversarial Robustness)
    text = normalize_text(request.text)
    tier = request.model_tier
    
    try:
        if tier == "tier2":
            # Lazy load Tier 2 model
            if bert_explainer is None:
                try:
                    bert_explainer = DeepExplainer()
                except Exception as e:
                    print(f"Failed to load Tier 2: {e}")
                    # Fallback to Tier 1
                    label, confidence = tfidf_classifier.predict(text)
                    explanation = {"info": f"Tier 2 failed to load, using Tier 1 fallback. Error: {str(e)}"}
                    return build_response(text, label, confidence, explanation)

            try:
                # Tier 2 (DistilBERT Multilingual + LoRA)
                inputs = {k: v.to(bert_explainer.device) for k, v in bert_explainer.tokenizer(text, return_tensors="pt", padding=True, truncation=True).items()}
                
                with torch.no_grad():
                    outputs = bert_explainer.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    confidence, label_idx = torch.max(probs, dim=-1)
                    label = CATEGORIES[label_idx.item()]
                    confidence = confidence.item()
                    label_idx_val = label_idx.item()
                    
                # BERT Explanation (XAI)
                explanation_data = bert_explainer.explain(text, label_idx_val)
                explanation = {"top_tokens": explanation_data[:10]}
                
                # Apply mitigation with XAI awareness
                action = policy_engine.recommend_action(label, confidence)
                final_text = policy_engine.apply_mitigation(request.text, action, explainer=bert_explainer)
                
                return GuardrailResponse(
                    text=final_text,
                    label=label,
                    confidence=float(confidence),
                    action=action,
                    explanation=explanation
                )
            except Exception as e:
                print(f"Tier 2 Inference Error: {e}")
                traceback.print_exc()
                label, confidence = tfidf_classifier.predict(text)
                explanation = {"info": f"Tier 2 inference failed, using Tier 1 fallback. Error: {str(e)}"}
        else:
            # Tier 1 (TF-IDF + Logistic Regression)
            label, confidence = tfidf_classifier.predict(text)
            explanation = {"info": "Switch to Tier 2 for token-level attribution"}

        return build_response(text, label, confidence, explanation)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Guardrail internal error: {str(e)}")

def build_response(text, label, confidence, explanation):
    action = policy_engine.recommend_action(label, confidence)
    final_text = policy_engine.apply_mitigation(text, action)
    
    return GuardrailResponse(
        text=final_text,
        label=label,
        confidence=float(confidence),
        action=action,
        explanation=explanation
    )

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "tier1_ready": tfidf_classifier.pipeline is not None or os.path.exists(tfidf_classifier.model_path),
        "tier2_ready": bert_explainer is not None and bert_explainer.lig is not None
    }

@app.get("/")
async def root():
    return {"status": "active", "framework": "Explainable Prompt Injection Guardrail"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
