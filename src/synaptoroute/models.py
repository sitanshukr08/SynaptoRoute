from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

class Route(BaseModel):
    name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    utterances: List[str]
    threshold: float = Field(0.0, ge=-1.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('utterances')
    @classmethod
    def deduplicate_utterances(cls, v: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for utt in v:
            if utt not in seen:
                seen.add(utt)
                deduped.append(utt)
        return deduped
