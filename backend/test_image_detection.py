import asyncio
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.models import Upload, DetectionResult
from app.services.image.detector import run_image_detection


async def main():
    test_path = r"C:\TrueLense\tl\test_sample.jpg"

    async with AsyncSessionLocal() as db:

        upload = Upload(
            file_name="test_sample.jpg",
            media_type="image",
            storage_url=test_path,
            file_size_kb=0,
        )

        db.add(upload)
        await db.flush()

        print("Upload created:", upload.upload_id)

        result_id = await run_image_detection(
            upload,
            db,
        )

        await db.commit()

        print("Detection result:", result_id)

        result = await db.execute(
            select(DetectionResult).where(
                DetectionResult.result_id == result_id
            )
        )

        detection = result.scalar_one()

        print()
        print("=" * 60)
        print("BACKEND IMAGE DETECTION TEST")
        print("=" * 60)

        print("Verdict:", detection.verdict)
        print("Confidence:", detection.confidence_score)
        print("Model:", detection.model_used)
        print(
            "Processing time:",
            detection.processing_time_ms,
            "ms",
        )

        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())