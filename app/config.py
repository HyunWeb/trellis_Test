from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Base URL this service is reachable at from the Next.js server, used to
    # build absolute URLs for generated GLB files (e.g. http://localhost:8000
    # locally, or your GPU pod's public URL in production).
    public_base_url: str = "http://localhost:8000"

    # Where uploaded PNGs and generated GLBs are kept.
    # Swap for S3/R2 storage (app/storage.py) once you leave local dev.
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    # If false, /jobs uses the placeholder cube pipeline instead of real
    # TRELLIS — the default, so this runs without a GPU for local dev.
    use_real_trellis: bool = False

    class Config:
        env_prefix = "TRELLIS_"
        env_file = ".env"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
(settings.data_dir / "uploads").mkdir(exist_ok=True)
(settings.data_dir / "outputs").mkdir(exist_ok=True)
