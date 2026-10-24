from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..services import agent_client
from .dashboard import find_wb, waterbody_index

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

SEVERITY = {"persistent_degradation": "high", "below_seasonal_normal": "watch"}


@router.get("")
async def list_alerts(db: Session = Depends(get_db)):
    """Stored alerts plus alerts DERIVED from the newest lake measurements:
    a lake at least 15% below its normal for the season is "watch"; one that
    has been below normal on its last three dates is "high"."""
    stored = (
        db.query(models.Alert).order_by(models.Alert.created_at.desc()).all()
    )
    out = [
        {
            "id": a.id, "waterbody_id": a.waterbody_id, "message": a.message,
            "severity": a.severity,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "status": a.status, "derived": False,
        }
        for a in stored
    ]
    try:
        ranked = await agent_client.call_real("/agent/rank", {})
    except Exception:
        return out
    index = waterbody_index(db)
    for item in ranked["items"]:
        severity = SEVERITY.get(item["status"])
        if severity is None:
            continue
        wb = find_wb(index, item["lon"], item["lat"])
        if wb is None:
            continue
        out.append({
            "id": f"derived_{item['lake_id']}", "waterbody_id": wb.id,
            "message": (f"{wb.name}: open water {item['open_water_ha']} ha on "
                        f"{item['date']}, below its normal for this season."),
            "severity": severity, "created_at": item["date"],
            "status": "derived", "derived": True,
        })
    order = {"high": 0, "watch": 1}
    out.sort(key=lambda a: (not a["derived"], order.get(a["severity"], 2)))
    return out
