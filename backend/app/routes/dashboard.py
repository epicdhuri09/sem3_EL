"""Dashboard endpoints: real lake measurements from the AI service.

None of these fall back to mock data. If the AI service is down they answer
503 and the frontend shows "Data unavailable".
"""
import ast
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..services import agent_client

router = APIRouter(tags=["dashboard"])


def _bbox(wb):
    return ast.literal_eval(wb.bbox_geojson) if wb.bbox_geojson else None


def _get_wb(db, waterbody_id):
    wb = db.query(models.Waterbody).filter(models.Waterbody.id == waterbody_id).first()
    if wb is None:
        raise HTTPException(404, f"Waterbody '{waterbody_id}' not found")
    return wb


def waterbody_index(db):
    """[(waterbody, bbox)] for point-in-box matching, smallest box first."""
    items = []
    for wb in db.query(models.Waterbody).all():
        bbox = _bbox(wb)
        if bbox and len(bbox) == 4:
            items.append((wb, bbox))
    items.sort(key=lambda i: (i[1][2] - i[1][0]) * (i[1][3] - i[1][1]))
    return items


def find_wb(index, lon, lat):
    for wb, b in index:
        if b[0] <= lon <= b[2] and b[1] <= lat <= b[3]:
            return wb
    return None


@router.get("/api/waterbodies/{waterbody_id}/series")
async def lake_series(waterbody_id: str, start: Optional[str] = None,
                      end: Optional[str] = None, db: Session = Depends(get_db)):
    wb = _get_wb(db, waterbody_id)
    data = await agent_client.call_real(
        "/agent/series", {"bbox": _bbox(wb), "from": start, "to": end})
    data["waterbody_id"] = waterbody_id
    return data


@router.get("/api/waterbodies/{waterbody_id}/seasonality")
async def lake_seasonality(waterbody_id: str, db: Session = Depends(get_db)):
    wb = _get_wb(db, waterbody_id)
    data = await agent_client.call_real("/agent/seasonality", {"bbox": _bbox(wb)})
    data["waterbody_id"] = waterbody_id
    return data


@router.post("/api/waterbodies/{waterbody_id}/xai")
async def lake_xai(waterbody_id: str, payload: dict = None,
                   db: Session = Depends(get_db)):
    wb = _get_wb(db, waterbody_id)
    data = await agent_client.call_real(
        "/agent/xai", {"bbox": _bbox(wb), "date": (payload or {}).get("date")})
    data["waterbody_id"] = waterbody_id
    return data


@router.get("/api/lookup")
async def lookup(lat: float, lon: float, radius_m: float = 1000,
                 db: Session = Depends(get_db)):
    data = await agent_client.call_real(
        "/agent/lookup", {"lat": lat, "lon": lon, "radius_m": radius_m})
    if data.get("found"):
        lake = data["lake"]
        b = lake["bbox"]
        wb = find_wb(waterbody_index(db), (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        data["waterbody_id"] = wb.id if wb else None
    return data


@router.get("/api/summary")
async def summary():
    return await agent_client.call_real("/agent/summary", {})


@router.get("/api/waterbodies.geojson")
async def lakes_geojson(db: Session = Depends(get_db)):
    data = await agent_client.call_real("/agent/geojson", {})
    index = waterbody_index(db)
    for feature in data["features"]:
        p = feature["properties"]
        wb = find_wb(index, p["lon"], p["lat"])
        p["id"] = wb.id if wb else None
        if wb and wb.name:
            p["name"] = p["name"] or wb.name
    return data


async def lake_info_by_wb(db):
    """waterbody id -> newest lake measurement (empty dict if AI is down)."""
    try:
        data = await agent_client.call_real("/agent/geojson", {})
    except HTTPException:
        return {}
    index = waterbody_index(db)
    info = {}
    for feature in data["features"]:
        p = feature["properties"]
        wb = find_wb(index, p["lon"], p["lat"])
        if wb is not None and wb.id not in info:
            info[wb.id] = p
    return info


def apply_info(item, p):
    """Fill the Bible 7.1 shape with real values where we have them."""
    if not p:
        return item
    item["latest_capture_date"] = p.get("date")
    item["change_type"] = p.get("status")
    item["open_water_ha"] = p.get("open_water_ha")
    item["outline_ha"] = p.get("outline_ha")
    item["deficit_fraction"] = p.get("deficit_fraction")
    return item
