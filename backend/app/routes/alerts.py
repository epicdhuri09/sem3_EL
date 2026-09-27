from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("")
def list_alerts(db: Session = Depends(get_db)):
    alerts = (
        db.query(models.Alert)
        .order_by(models.Alert.created_at.desc())
        .all()
    )
    return [
        {
            "id": a.id,
            "waterbody_id": a.waterbody_id,
            "message": a.message,
            "severity": a.severity,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "status": a.status,
        }
        for a in alerts
    ]
