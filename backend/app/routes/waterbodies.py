import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api/waterbodies", tags=["waterbodies"])


def waterbody_to_dict(wb: models.Waterbody) -> dict:
    """Matches the Bible v2 Chapter 7.1 response shape."""
    return {
        "id": wb.id,
        "name": wb.name,
        "bbox": eval(wb.bbox_geojson) if wb.bbox_geojson else None,
        "ward": wb.ward,
        "latest_capture_date": None,
        "scores": {
            "water_health": None,
            "vegetation_health": None,
            "degradation_risk": None,
            "restoration_urgency": None,
            "data_reliability": None,
            "recovery_trend": None,
        },
        "segmentation": {
            "water_pct": None,
            "vegetation_pct": None,
            "bare_land_pct": None,
            "built_up_pct": None,
        },
        "change_point": None,
        "new_structures_since_baseline": None,
        "change_type": None,
        "confidence": None,
    }


@router.get("")
def list_waterbodies(db: Session = Depends(get_db)):
    waterbodies = db.query(models.Waterbody).all()
    return [waterbody_to_dict(wb) for wb in waterbodies]


@router.post("")
def create_waterbody(payload: schemas.WaterbodyCreate, db: Session = Depends(get_db)):
    wb_id = f"wb_{uuid.uuid4().hex[:8]}"
    wb = models.Waterbody(
        id=wb_id,
        name=payload.name,
        bbox_geojson=str(payload.bbox),
        ward=payload.ward,
        region=payload.region,
    )
    db.add(wb)
    db.commit()
    db.refresh(wb)
    return waterbody_to_dict(wb)


@router.get("/{waterbody_id}")
def get_waterbody(waterbody_id: str, db: Session = Depends(get_db)):
    wb = db.query(models.Waterbody).filter(models.Waterbody.id == waterbody_id).first()
    if wb is None:
        raise HTTPException(status_code=404, detail=f"Waterbody '{waterbody_id}' not found")
    return waterbody_to_dict(wb)


@router.get("/{waterbody_id}/timeline")
def get_waterbody_timeline(waterbody_id: str, db: Session = Depends(get_db)):
    wb = db.query(models.Waterbody).filter(models.Waterbody.id == waterbody_id).first()
    if wb is None:
        raise HTTPException(status_code=404, detail=f"Waterbody '{waterbody_id}' not found")

    captures = (
        db.query(models.ImageryCapture)
        .filter(models.ImageryCapture.waterbody_id == waterbody_id)
        .order_by(models.ImageryCapture.date)
        .all()
    )
    change_point = (
        db.query(models.ChangePoint)
        .filter(models.ChangePoint.waterbody_id == waterbody_id)
        .first()
    )

    return {
        "waterbody_id": waterbody_id,
        "captures": [
            {
                "capture_id": c.id,
                "date": c.date.isoformat() if c.date else None,
                "season": c.season,
                "cloud_cover_pct": c.cloud_cover_pct,
            }
            for c in captures
        ],
        "change_point": (
            {
                "year": change_point.change_point_year,
                "before_mean": change_point.before_mean,
                "after_mean": change_point.after_mean,
                "narrative": change_point.narrative,
            }
            if change_point
            else None
        ),
    }


@router.post("/{waterbody_id}/analyze")
async def analyze_waterbody(waterbody_id: str, payload: dict = None, db: Session = Depends(get_db)):
    """
    Triggers the full AI Agent pipeline for this waterbody. Works for ANY
    registered lake (no hardcoded list) -- payload optionally carries a
    'date' (defaults to most recent available imagery).
    """
    from ..services import agent_client

    wb = db.query(models.Waterbody).filter(models.Waterbody.id == waterbody_id).first()
    if wb is None:
        raise HTTPException(status_code=404, detail=f"Waterbody '{waterbody_id}' not found")

    date = (payload or {}).get("date")
    bbox = eval(wb.bbox_geojson)

    seg_result = await agent_client.segment({"waterbody_id": waterbody_id, "bbox": bbox, "date": date})
    anomaly_result = await agent_client.anomaly({"waterbody_id": waterbody_id, "bbox": bbox})
    changepoint_result = await agent_client.changepoint({"waterbody_id": waterbody_id})
    structures_result = await agent_client.structures({"waterbody_id": waterbody_id, "bbox": bbox})
    cause_result = await agent_client.cause({
        "waterbody_id": waterbody_id,
        "segmentation": seg_result.get("class_percentages"),
        "anomaly": anomaly_result,
    })
    priority_result = await agent_client.priority({
        "severity": anomaly_result.get("anomaly_score", 0) * 100,
        "persistence": 50, "ecological_importance": 50, "population_impact": 50,
    })

    return {
        "waterbody_id": waterbody_id,
        "segmentation": seg_result,
        "anomaly": anomaly_result,
        "change_point": changepoint_result,
        "structures": structures_result,
        "cause_hypotheses": cause_result,
        "priority": priority_result,
        "note": "Fields marked _mock: true are placeholder data -- AI Agent service not yet connected." if seg_result.get("_mock") else None,
    }
