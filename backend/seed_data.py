"""
One-time script to seed a few Bengaluru lakes for development/testing.
Run with: python seed_data.py (from inside backend/, venv active)
"""
from app.database import SessionLocal, engine
from app import models

models.Base.metadata.create_all(bind=engine)
db = SessionLocal()

lakes = [
    {"id": "wb_varthur01", "name": "Varthur Lake", "bbox": "[77.73, 12.93, 77.76, 12.96]", "ward": "Varthur Ward 150"},
    {"id": "wb_hebbal001", "name": "Hebbal Lake", "bbox": "[77.59, 13.04, 77.61, 13.06]", "ward": "Hebbal Ward 15"},
    {"id": "wb_ulsoor01", "name": "Ulsoor Lake", "bbox": "[77.61, 12.97, 77.63, 12.99]", "ward": "Ulsoor Ward 84"},
]

for lake in lakes:
    exists = db.query(models.Waterbody).filter(models.Waterbody.id == lake["id"]).first()
    if not exists:
        db.add(models.Waterbody(
            id=lake["id"], name=lake["name"],
            bbox_geojson=lake["bbox"], ward=lake["ward"],
        ))
        print(f"Added {lake['name']}")
    else:
        print(f"{lake['name']} already exists, skipping")

db.commit()
db.close()
