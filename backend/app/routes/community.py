import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db

router = APIRouter(prefix="/api/community-reports", tags=["community-reports"])


@router.get("")
def list_community_reports(db: Session = Depends(get_db)):
    reports = db.query(models.CommunityReport).all()
    return [
        {
            "id": r.id,
            "waterbody_id": r.waterbody_id,
            "note": r.note,
            "photo_path": r.photo_path,
            "submitted_at": r.submitted_at.isoformat() if r.submitted_at else None,
            "status": r.status,
        }
        for r in reports
    ]


@router.post("")
def create_community_report(payload: dict, db: Session = Depends(get_db)):
    report_id = f"cr_{uuid.uuid4().hex[:8]}"
    report = models.CommunityReport(
        id=report_id,
        waterbody_id=payload["waterbody_id"],
        note=payload.get("note"),
        photo_path=payload.get("photo_path"),
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return {
        "id": report.id,
        "waterbody_id": report.waterbody_id,
        "note": report.note,
        "photo_path": report.photo_path,
        "submitted_at": report.submitted_at.isoformat() if report.submitted_at else None,
        "status": report.status,
    }
