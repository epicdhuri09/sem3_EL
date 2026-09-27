from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db

router = APIRouter(prefix="/api/priority", tags=["priority"])


@router.get("")
def get_priority_ranking(
    ward: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    query = (
        db.query(models.PriorityScore, models.Waterbody)
        .join(models.Waterbody, models.PriorityScore.waterbody_id == models.Waterbody.id)
    )
    if ward:
        query = query.filter(models.Waterbody.ward == ward)

    results = query.order_by(models.PriorityScore.priority_score.desc()).all()

    return [
        {
            "waterbody_id": wb.id,
            "name": wb.name,
            "ward": wb.ward,
            "priority_score": ps.priority_score,
            "rank": ps.rank,
            "severity": ps.severity,
            "persistence": ps.persistence,
        }
        for ps, wb in results
    ]
