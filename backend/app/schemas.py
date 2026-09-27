from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class WaterbodyCreate(BaseModel):
    name: str = Field(..., min_length=1)
    bbox: List[float]
    ward: Optional[str] = None
    region: Optional[str] = None

    @field_validator("bbox")
    @classmethod
    def bbox_must_have_4_values(cls, v):
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 values: [min_lon, min_lat, max_lon, max_lat]")
        return v


class CommunityReportCreate(BaseModel):
    waterbody_id: str
    note: Optional[str] = None
    photo_path: Optional[str] = None


class SimulateRequest(BaseModel):
    waterbody_id: str
    vegetation_increase_pct: float = 0
    desilt_pct: float = 0
