# Trellis_AI

업로드된 PNG를 GLB 메시로 변환하는 FastAPI 잡(job) 서버. `web`(Next.js) 저장소에서 호출한다.

- `POST /jobs` — PNG를 multipart로 업로드(`image` 필드) → 잡을 생성하고 백그라운드에서 변환을 시작, `{id, status}` 반환
- `GET /jobs/{id}` — 잡 상태 폴링: `queued` → `processing` → `done`(`glb_url` 포함) 또는 `failed`(`error` 포함)
- `GET /files/outputs/{id}.glb` — 생성된 메시를 정적 파일로 서빙

## 로컬 개발 (GPU 불필요)

기본값은 `TRELLIS_USE_REAL_TRELLIS=false`라서, 잡이 실제 TRELLIS 대신 플레이스홀더 파이프라인(`app/pipeline.py::_run_placeholder`, 주황색 큐브만 내보냄)을 거친다. GPU 없는 노트북에서도 업로드 → 잡 → three.js 렌더까지 전체 흐름을 빌드/테스트할 수 있다.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

이후 `web` 저장소의 `.env.local`에서 `TRELLIS_SERVICE_URL=http://localhost:8000`을 가리키게 설정한다.

## 실제 TRELLIS 연결하기

실제 추론에는 NVIDIA GPU가 필요하고 Mac에서는 돌릴 수 없다. GPU 머신/파드에서:

1. [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)의 셋업 안내를 따라 의존성을 설치한다(`requirements-trellis.txt`에 대략적인 구성이 적혀 있음 — 정확한 버전은 CUDA/torch 빌드에 따라 다름).
2. `app/pipeline.py::_run_trellis`는 이미 실제 TRELLIS 추론으로 구현돼 있다(파이프라인을 프로세스당 한 번만 로드하는 지연 초기화 싱글턴 방식).
3. `TRELLIS_USE_REAL_TRELLIS=true`로 설정하고, `TRELLIS_PUBLIC_BASE_URL`을 그 머신/파드의 공개 주소로 맞춘다.

## Docker로 실제 TRELLIS 돌리기 (GPU 있는 아무 머신에서나)

매번 손으로 WSL2/conda 셋업을 반복하는 대신, `Dockerfile` + `docker-compose.yml`이 실제 추론 환경을 그대로 패키징해준다:

```bash
git clone https://github.com/HyunWeb/Trellis_AI.git
cd Trellis_AI
docker compose up --build
```

호스트 쪽 전제조건:
- NVIDIA GPU, VRAM 16GB 안팎(TRELLIS-image-large 기준 최대 사용량이 그 정도)
- 최신 NVIDIA 드라이버
- Docker가 GPU를 볼 수 있도록 설정 — Linux: `nvidia-container-toolkit`; Windows: Docker Desktop을 WSL2 백엔드로 사용(최신 NVIDIA 드라이버는 기본적으로 GPU 패스스루를 지원함)

첫 빌드는 CUDA 관련 패키지(flash-attn, nvdiffrast 등)를 소스에서 컴파일하느라 시간이 걸릴 수 있다. 첫 요청 시 HuggingFace에서 TRELLIS-image-large를 내려받는데, 이후엔 named volume에 캐싱돼서 재시작해도 다시 받지 않는다.

이 환경엔 GPU/Docker가 없어서 실제 빌드 테스트는 못 했다 — 원래 WSL2 셋업에서 실제로 통했던 순서를 그대로 옮긴 것이니, 실제 GPU 머신에서 처음 빌드할 때 한두 번은 조정이 필요할 수 있다.

## 참고 / 프로덕션 TODO

- 잡 저장소가 인메모리(`app/jobs.py`)라 프로세스 1개엔 문제없지만, 워커를 여러 개 돌리려면 Redis/RQ 등으로 교체 필요
- 파일 저장이 로컬 디스크(`data/`) — 단일 지속 서버가 아니라면 S3/R2 등으로 교체 필요
- CORS가 로컬 개발용으로 전부 열려있음(`allow_origins=["*"]`) — 실제 도메인이 생기면 `app/main.py`에서 제한 필요
