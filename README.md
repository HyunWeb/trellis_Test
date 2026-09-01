# trellis-service

FastAPI job server that turns an uploaded PNG into a GLB mesh, called by the
`web` (Next.js) repo.

- `POST /jobs` — multipart upload (`image`, PNG) → creates a job, kicks off
  processing in the background, returns `{id, status}`.
- `GET /jobs/{id}` — poll job status: `queued` → `processing` → `done` (with
  `glb_url`) or `failed` (with `error`).
- `GET /files/outputs/{id}.glb` — the generated mesh, served statically.

## Local dev (no GPU needed)

By default `TRELLIS_USE_REAL_TRELLIS=false`, so jobs run through a
placeholder pipeline (`app/pipeline.py::_run_placeholder`) that just exports
an orange cube. This lets you build/test the whole upload → job → three.js
render flow on a laptop.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Then point the `web` repo's `.env.local` at `TRELLIS_SERVICE_URL=http://localhost:8000`.

## Wiring up real TRELLIS

Real inference needs an NVIDIA GPU and cannot run on a Mac. On a GPU
box/pod:

1. Follow [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) setup
   instructions and install its deps (`requirements-trellis.txt` documents
   the shape — exact versions depend on your CUDA/torch build).
2. Implement `app/pipeline.py::_run_trellis` (a sketch is left as a
   docstring there).
3. Set `TRELLIS_USE_REAL_TRELLIS=true` and `TRELLIS_PUBLIC_BASE_URL` to the
   pod's public URL.

## Notes / production TODOs

- Job store is in-memory (`app/jobs.py`) — fine for one process; swap for
  Redis/RQ if you run multiple workers.
- File storage is local disk (`data/`) — swap for S3/R2 if you deploy off a
  single persistent box.
- CORS is wide open for local dev — restrict `allow_origins` in
  `app/main.py` once the web app has a real domain.
