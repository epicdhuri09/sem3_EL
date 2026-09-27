from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db

router = APIRouter(prefix="/api/simulate", tags=["simulate"])


@router.post("")
def simulate_intervention(payload: dict, db: Session = Depends(get_db)):
    """
    Rule-based placeholder (per Bible Chapter 3 realism note -- this is NOT
    a trained model). Takes a waterbody_id and simple intervention deltas,
    returns a projected priority score.
    """
    waterbody_id = payload.get("waterbody_id")
    vegetation_increase_pct = payload.get("vegetation_increase_pct", 0)
    desilt_pct = payload.get("desilt_pct", 0)

    current = (
        db.query(models.PriorityScore)
        .filter(models.PriorityScore.waterbody_id == waterbody_id)
        .first()
    )
    current_score = current.priority_score if current else 50.0

    # Simple, transparent rule: each 1% vegetation increase or desilting
    # reduces projected priority score by 0.3 points, floor at 0.
    reduction = (vegetation_increase_pct * 0.3) + (desilt_pct * 0.3)
    projected_score = max(0.0, current_score - reduction)

    return {
        "waterbody_id": waterbody_id,
        "current_priority_score": current_score,
        "projected_priority_score": round(projected_score, 1),
        "assumptions": "Rule-based estimate, not a trained model -- see Bible Chapter 3.",
    }
