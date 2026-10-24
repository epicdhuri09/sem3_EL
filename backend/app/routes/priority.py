from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..services import agent_client
from .dashboard import find_wb, waterbody_index

router = APIRouter(prefix="/api/priority", tags=["priority"])


@router.get("")
async def get_priority_ranking(
    ward: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """Ranking computed live from the lake measurements (AI service).
    Falls back to stored rows only if the AI service is unreachable."""
    try:
        ranked = await agent_client.call_real("/agent/rank", {})
    except Exception:
        ranked = None

    if ranked:
        index = waterbody_index(db)
        out = []
        for item in ranked["items"]:
            wb = find_wb(index, item["lon"], item["lat"])
            if wb is None or (ward and wb.ward != ward):
                continue
            out.append({
                "waterbody_id": wb.id, "name": wb.name, "ward": wb.ward,
                "priority_score": item["priority_score"], "rank": len(out) + 1,
                "severity": item["severity"], "persistence": item["persistence"],
                "ecological_importance": item["ecological_importance"],
                "population_impact": item["population_impact"],
                "status": item["status"], "open_water_ha": item["open_water_ha"],
                "date": item["date"], "note": ranked["note"],
            })
        return out

    query = (
        db.query(models.PriorityScore, models.Waterbody)
        .join(models.Waterbody, models.PriorityScore.waterbody_id == models.Waterbody.id)
    )
    if ward:
        query = query.filter(models.Waterbody.ward == ward)
    results = query.order_by(models.PriorityScore.priority_score.desc()).all()
    return [
        {
            "waterbody_id": wb.id, "name": wb.name, "ward": wb.ward,
            "priority_score": ps.priority_score, "rank": ps.rank,
            "severity": ps.severity, "persistence": ps.persistence,
        }
        for ps, wb in results
    ]
