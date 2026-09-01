import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.jobs import Job, JobStatus, store
from app.pipeline import run as run_pipeline

logger = logging.getLogger("trellis-service")

app = FastAPI(title="trellis-service")

# Loosened for local dev against a Next.js dev server. Tighten to the
# deployed web app's origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=settings.data_dir), name="files")


@app.get("/health")
def health():
    return {"status": "ok", "use_real_trellis": settings.use_real_trellis}


@app.post("/jobs", status_code=201)
async def create_job(image: UploadFile, background_tasks: BackgroundTasks) -> Job:
    if image.content_type != "image/png":
        raise HTTPException(400, "Only PNG uploads are supported")

    job = store.create()

    upload_path = settings.data_dir / "uploads" / f"{job.id}.png"
    upload_path.write_bytes(await image.read())

    background_tasks.add_task(_process_job, job.id, upload_path)
    return job


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


def _process_job(job_id: str, upload_path) -> None:
    store.update(job_id, status=JobStatus.PROCESSING)
    output_path = settings.data_dir / "outputs" / f"{job_id}.glb"

    try:
        run_pipeline(upload_path, output_path)
        glb_url = f"{settings.public_base_url}/files/outputs/{job_id}.glb"
        store.update(job_id, status=JobStatus.DONE, glb_url=glb_url)
    except Exception as exc:  # noqa: BLE001 - report any failure back to the client
        logger.exception("Job %s failed", job_id)
        store.update(job_id, status=JobStatus.FAILED, error=str(exc))
