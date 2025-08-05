from pydantic import BaseModel
from typing import List

class OCRResult(BaseModel):
    text: str
    pii_found: bool

class RepresentativeImageResult(BaseModel):
    filename: str

class DescriptionResult(BaseModel):
    description: str
    embedding: List[float]
