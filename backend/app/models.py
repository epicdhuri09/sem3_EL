from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text
)
from sqlalchemy.sql import func
from .database import Base


class Waterbody(Base):
    __tablename__ = "waterbodies"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    bbox_geojson = Column(Text, nullable=False)
    ward = Column(String, nullable=True)
    region = Column(String, nullable=True)
    registered_at = Column(DateTime, server_default=func.now())


class ImageryCapture(Base):
    __tablename__ = "imagery_captures"

    id = Column(String, primary_key=True, index=True)
    waterbody_id = Column(String, ForeignKey("waterbodies.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    season = Column(String, nullable=True)
    source = Column(String, default="Sentinel-2")
    cloud_cover_pct = Column(Float, nullable=True)
    file_path = Column(String, nullable=True)


class SegmentationResult(Base):
    __tablename__ = "segmentation_results"

    id = Column(String, primary_key=True, index=True)
    capture_id = Column(String, ForeignKey("imagery_captures.id"), nullable=False)
    water_pct = Column(Float)
    vegetation_pct = Column(Float)
    bare_land_pct = Column(Float)
    built_up_pct = Column(Float)
    mask_path = Column(String, nullable=True)


class AnomalyScore(Base):
    __tablename__ = "anomaly_scores"

    id = Column(String, primary_key=True, index=True)
    waterbody_id = Column(String, ForeignKey("waterbodies.id"), nullable=False)
    capture_id = Column(String, ForeignKey("imagery_captures.id"), nullable=False)
    expected_ndvi = Column(Float, nullable=True)
    expected_ndwi = Column(Float, nullable=True)
    observed_ndvi = Column(Float, nullable=True)
    observed_ndwi = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    change_type = Column(String, nullable=True)


class ChangePoint(Base):
    __tablename__ = "change_points"

    id = Column(String, primary_key=True, index=True)
    waterbody_id = Column(String, ForeignKey("waterbodies.id"), nullable=False)
    change_point_year = Column(Integer, nullable=True)
    before_mean = Column(Float, nullable=True)
    after_mean = Column(Float, nullable=True)
    narrative = Column(Text, nullable=True)


class StructureCount(Base):
    __tablename__ = "structure_counts"

    id = Column(String, primary_key=True, index=True)
    waterbody_id = Column(String, ForeignKey("waterbodies.id"), nullable=False)
    baseline_year = Column(Integer, nullable=True)
    current_year = Column(Integer, nullable=True)
    new_structures = Column(Integer, nullable=True)


class XaiExplanation(Base):
    __tablename__ = "xai_explanations"

    id = Column(String, primary_key=True, index=True)
    capture_id = Column(String, ForeignKey("imagery_captures.id"), nullable=False)
    method = Column(String, nullable=False)
    overlay_path = Column(String, nullable=True)
    band_importance_json = Column(Text, nullable=True)
    counterfactual_text = Column(Text, nullable=True)


class CauseHypothesis(Base):
    __tablename__ = "cause_hypotheses"

    id = Column(String, primary_key=True, index=True)
    waterbody_id = Column(String, ForeignKey("waterbodies.id"), nullable=False)
    capture_id = Column(String, ForeignKey("imagery_captures.id"), nullable=False)
    cause = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    evidence_text = Column(Text, nullable=True)


class PriorityScore(Base):
    __tablename__ = "priority_scores"

    id = Column(String, primary_key=True, index=True)
    waterbody_id = Column(String, ForeignKey("waterbodies.id"), nullable=False)
    computed_at = Column(DateTime, server_default=func.now())
    severity = Column(Float, nullable=True)
    persistence = Column(Float, nullable=True)
    ecological_importance = Column(Float, nullable=True)
    population_impact = Column(Float, nullable=True)
    priority_score = Column(Float, nullable=True)
    rank = Column(Integer, nullable=True)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, index=True)
    waterbody_id = Column(String, ForeignKey("waterbodies.id"), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    status = Column(String, default="active")


class CommunityReport(Base):
    __tablename__ = "community_reports"

    id = Column(String, primary_key=True, index=True)
    waterbody_id = Column(String, ForeignKey("waterbodies.id"), nullable=False)
    note = Column(Text, nullable=True)
    photo_path = Column(String, nullable=True)
    submitted_at = Column(DateTime, server_default=func.now())
    status = Column(String, default="pending")
