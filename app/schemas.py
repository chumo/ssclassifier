from pydantic import BaseModel, Field, ValidationInfo, field_validator
from typing import List

class DetectRequest(BaseModel):
    image_path: str = Field(description="The absolute path to the local image file")
    coords: List[float] = Field(description="A list of coordinate floats, length must be a multiple of 6")

    @field_validator('coords')
    @classmethod
    def validate_coords_length(cls, v: List[float], info: ValidationInfo) -> List[float]:
        if not v:
            raise ValueError("coords list cannot be empty")
        if len(v) % 6 != 0:
            raise ValueError(f"coords list length must be a multiple of 6, got {len(v)}")
        return v

class DetectResponse(BaseModel):
    result: str

class TrainingSample(BaseModel):
    image_path: str = Field(description="The absolute path to the local image file")
    coords: List[float] = Field(description="A list of exactly 6 coordinate floats")
    label: str = Field(description="The assigned label/character for the digit")

    @field_validator('coords')
    @classmethod
    def validate_coords_length(cls, v: List[float], info: ValidationInfo) -> List[float]:
        if len(v) != 6:
            raise ValueError(f"coords list length must be exactly 6, got {len(v)}")
        return v

class TrainRequest(BaseModel):
    samples: List[TrainingSample]
