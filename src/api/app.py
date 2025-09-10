from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import os

app = FastAPI(title="Business Name Generator API", version="1.0.0")

# Load model globally
BASE_MODEL = os.getenv('MODEL_NAME', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
tokenizer = None
model = None

class GenerateRequest(BaseModel):
    description: str
    approach: str = "baseline"
    num_suggestions: int = 5

class GenerateResponse(BaseModel):
    suggestions: List[str]
    approach: str
    description: str

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    try:
        print("Loading TinyLlama model...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None
        )
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/generate", response_model=GenerateResponse)
async def generate_names(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        suggestions = []
        prompt = f"""### Instruction:
You are a business naming expert. Suggest a short, memorable and professional company name.

### Description:
{request.description}

### Suggested Name:
"""
        
        # Generate names
        for _ in range(request.num_suggestions * 2):  # Generate more to filter
            inputs = tokenizer(prompt, return_tensors='pt')
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = text.split('### Suggested Name:')[-1].strip().split('\n')[0].strip()
            pred = re.sub(r'[^\w\s-]', '', pred).strip()
            
            if pred and len(pred) >= 3 and pred not in suggestions:
                suggestions.append(pred.capitalize())
                if len(suggestions) >= request.num_suggestions:
                    break
        
        # Fallbacks if not enough suggestions generated
        if len(suggestions) < request.num_suggestions:
            fallbacks = ["TechFlow", "BizCore", "ProHub", "SmartEdge", "NextGen"]
            for fallback in fallbacks:
                if fallback not in suggestions:
                    suggestions.append(fallback)
                    if len(suggestions) >= request.num_suggestions:
                        break
        
        return GenerateResponse(
            suggestions=suggestions[:request.num_suggestions],
            approach=request.approach,
            description=request.description
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.post("/generate-enhanced")
async def generate_enhanced_names(request: GenerateRequest):
    """Enhanced endpoint with better prompt engineering"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Enhanced prompt with better context
        business_type = ""
        desc_lower = request.description.lower()
        if any(word in desc_lower for word in ['tech', 'ai', 'digital']):
            business_type = "technology"
        elif any(word in desc_lower for word in ['food', 'restaurant', 'cafe']):
            business_type = "food & beverage"
        
        enhanced_prompt = f"""### Task: Business Name Generation Expert
You are a professional business naming consultant. Generate a short, memorable, and brandable company name.

### Business Description:
{request.description}

### Requirements:
- Length: 4-12 characters preferred
- Style: Modern, professional, memorable
- Avoid: Generic words like "Company", "Corp", "Business", "Solutions"
- Target: {business_type if business_type else "general business"}

### Company Name:"""
        
        suggestions = []
        attempts = 0
        max_attempts = request.num_suggestions * 4
        
        while len(suggestions) < request.num_suggestions and attempts < max_attempts:
            attempts += 1
            
            inputs = tokenizer(enhanced_prompt, return_tensors='pt')
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Try multiple extraction patterns
            pred = ""
            extraction_patterns = ["### Company Name:", "Company Name:", "Name:"]
            for pattern in extraction_patterns:
                if pattern in text:
                    pred = text.split(pattern)[-1].strip().split('\n')[0].strip()
                    break
            
            if not pred:
                pred = text.split(enhanced_prompt)[-1].strip().split('\n')[0].strip()
            
            pred = re.sub(r'[^\w\s-]', '', pred).strip()
            pred = ' '.join(pred.split())  # Clean extra spaces
            
            # Enhanced quality filters
            is_valid = (
                pred and 
                2 <= len(pred) <= 25 and
                pred not in suggestions and
                not pred.lower() in ['company', 'business', 'corp', 'inc', 'ltd'] and
                not any(placeholder in pred.lower() for placeholder in [
                    'insert', 'your', 'name', 'here', 'idea', 'placeholder'
                ])
            )
            
            if is_valid:
                suggestions.append(pred.title())
        
        # Smart fallbacks based on description
        if len(suggestions) < request.num_suggestions:
            if 'tech' in desc_lower or 'ai' in desc_lower:
                fallbacks = ["TechFlow", "NeoLab", "ByteCore", "SmartHub", "FlexTech"]
            elif 'food' in desc_lower:
                fallbacks = ["FreshFlow", "TasteHub", "FoodCore", "EatSmart", "QuickBite"]
            elif 'health' in desc_lower or 'fitness' in desc_lower:
                fallbacks = ["VitalTech", "HealthHub", "FitCore", "WellFlow", "ActiveLab"]
            else:
                fallbacks = ["ProFlow", "NextGen", "CoreHub", "SmartEdge", "FlexCore"]
            
            for fallback in fallbacks:
                if fallback not in suggestions:
                    suggestions.append(fallback)
                    if len(suggestions) >= request.num_suggestions:
                        break
        
        return {
            "suggestions": suggestions[:request.num_suggestions],
            "approach": "enhanced",
            "description": request.description,
            "analysis": {
                "generation_attempts": attempts,
                "business_type_detected": business_type or "general",
                "overall_score": 4.0  # Mock score
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced generation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
