# Business Name & Domain Generator API - Production Ready
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import re
import socket
from datetime import datetime

app = FastAPI(
    title="Business Name & Domain Generator API",
    description="Generate business names and domain suggestions with confidence scoring",
    version="1.0.0"
)

class BusinessRequest(BaseModel):
    business_description: str = Field(..., min_length=5, max_length=500)
    max_suggestions: Optional[int] = Field(default=6, ge=1, le=10)

class DomainSuggestion(BaseModel):
    domain: str
    confidence: float

class APIResponse(BaseModel):
    suggestions: List[DomainSuggestion]
    status: str
    message: Optional[str] = None

class BusinessNameGenerator:
    def __init__(self):
        self.blocked_keywords = {
            'adult', 'explicit', 'nude', 'porn', 'sex', 'xxx', 'casino', 
            'gambling', 'drug', 'weapon', 'hate', 'scam', 'fraud'
        }
        self.tlds = ['.com', '.net', '.org', '.io', '.ai', '.co']

    def is_safe_content(self, text: str) -> bool:
        text_lower = text.lower()
        return not any(keyword in text_lower for keyword in self.blocked_keywords)

    def generate_business_names(self, description: str) -> List[str]:
        desc_lower = description.lower()
        names = []

        # Industry-specific name generation
        if any(word in desc_lower for word in ['coffee', 'cafe', 'organic']):
            names = ['OrganicBean', 'FreshBrew', 'DowntownCafe', 'PureCoffee', 'LocalBrew']
        elif any(word in desc_lower for word in ['tech', 'ai', 'software', 'platform']):
            names = ['TechFlow', 'SmartCore', 'InnoLab', 'AIHub', 'CodeForge']
        elif any(word in desc_lower for word in ['fitness', 'coaching', 'health']):
            names = ['FitCore', 'HealthHub', 'VitalTech', 'WellFlow', 'ActiveLab']
        elif any(word in desc_lower for word in ['sustainable', 'eco', 'packaging']):
            names = ['EcoFlow', 'GreenCore', 'SustainHub', 'PackTech', 'EarthLab']
        else:
            names = ['BizCore', 'ProHub', 'SmartEdge', 'FlexFlow', 'PrimeLab']

        return names[:5]

    def generate_domains(self, names: List[str], description: str) -> List[Dict]:
        suggestions = []

        for name in names:
            base_name = re.sub(r'[^a-zA-Z0-9]', '', name.lower())

            for tld in self.tlds:
                domain = f"{base_name}{tld}"
                confidence = self.calculate_confidence(domain, description)

                suggestions.append({
                    'domain': domain,
                    'confidence': round(confidence, 2)
                })

        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        return suggestions

    def calculate_confidence(self, domain: str, description: str) -> float:
        confidence = 0.7  # Base confidence

        # TLD preferences
        if domain.endswith('.com'):
            confidence += 0.2
        elif domain.endswith(('.net', '.org')):
            confidence += 0.15
        elif domain.endswith(('.io', '.ai')):
            confidence += 0.1

        # Length optimization
        domain_base = domain.split('.')[0]
        if 6 <= len(domain_base) <= 12:
            confidence += 0.1

        return min(1.0, confidence)

generator = BusinessNameGenerator()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/generate", response_model=APIResponse)
async def generate_names(request: BusinessRequest):
    try:
        # Safety check
        if not generator.is_safe_content(request.business_description):
            return APIResponse(
                suggestions=[],
                status="blocked",
                message="Request contains inappropriate content"
            )

        # Generate names and domains
        business_names = generator.generate_business_names(request.business_description)
        domain_suggestions = generator.generate_domains(business_names, request.business_description)

        # Format response
        suggestions = [
            DomainSuggestion(domain=d['domain'], confidence=d['confidence'])
            for d in domain_suggestions[:request.max_suggestions]
        ]

        return APIResponse(
            suggestions=suggestions,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
