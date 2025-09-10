from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from typing import List

app = FastAPI(title="Business Name Generator API", version="1.0.0")

# Global variables for model and tokenizer
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    description: str
    num_suggestions: int = 3

class GenerateResponse(BaseModel):
    suggestions: List[str]
    description: str
    response_time_ms: float

@app.on_event("startup")
async def load_model():
    """Load the model on startup"""
    global model, tokenizer
    try:
        print("Loading TinyLlama model...")
        BASE_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_names(request: GenerateRequest):
    """Generate business names"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    try:
        # Generate names using the model
        suggestions = generate_business_names(request.description, request.num_suggestions)
        
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        return GenerateResponse(
            suggestions=suggestions,
            description=request.description,
            response_time_ms=response_time_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

def generate_business_names(description: str, num_suggestions: int = 3) -> List[str]:
    """Generate business names using the loaded model"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return [f"FallbackName{i+1}" for i in range(num_suggestions)]
    
    PROMPT_TEMPLATE = """### Instruction:
You are a business naming expert. Suggest a short, memorable and professional company name.

### Description:
{description}

### Suggested Name:
"""
    
    suggestions = []
    
    try:
        prompt = PROMPT_TEMPLATE.format(description=description)
        
        # Generate multiple attempts
        for _ in range(num_suggestions * 2):
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
            
            # Clean and validate
            if pred and len(pred) >= 3 and len(pred) <= 20 and pred not in suggestions:
                if not pred.lower() in ['company', 'business', 'corp', 'inc']:
                    suggestions.append(pred.capitalize())
                    
            if len(suggestions) >= num_suggestions:
                break
                
    except Exception as e:
        print(f"Generation error: {e}")
    
    # Ensure we have enough suggestions
    while len(suggestions) < num_suggestions:
        suggestions.append(f"BizName{len(suggestions)+1}")
    
    return suggestions[:num_suggestions]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)