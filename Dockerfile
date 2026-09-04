# syntax=docker/dockerfile:1
#
# 실제 TRELLIS 추론용 GPU 이미지. WSL2에서 직접 겪은 문제들(conda ToS 동의
# 필요, cuda-toolkit이 끌고 오는 최신 mkl과 torch 심볼 충돌, flash-attn 등
# 빌드-isolation 필요, transformers 5.x가 torch>=2.5 요구)을 그대로 반영해서
# 처음부터 그 문제가 안 생기게 순서를 잡았다.
#
# ⚠️ 이 Dockerfile은 로컬에서 직접 빌드/실행해서 검증한 게 아니다(이 환경엔
# GPU/Docker가 없음) — CUDA/TRELLIS 의존성 설치는 원래도 삽질이 많았던
# 영역이라, 실제 GPU 머신에서 처음 build할 때 버전 충돌 등으로 한두 번은
# 조정이 필요할 수 있다. 문제가 생기면 에러 메시지를 그대로 알려주면 같이
# 고치면 된다.

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# build-essential: flash-attn/nvdiffrast/diffoctreerast/mip-splatting를
# --no-build-isolation으로 소스 빌드하는 데 필요(원래 WSL 셋업 때도 g++
# 없어서 한 번 막혔었음).
# git: TRELLIS 저장소 clone. curl: healthcheck용.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda ──
# conda를 쓰는 이유: TRELLIS 자체가 conda 기반 setup.sh를 제공하고, mkl
# 버전을 conda로 정확히 고정해야 했던 경험 때문에 WSL 셋업과 최대한 동일한
# 경로를 유지한다.
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# conda 채널 이용약관 동의 — 안 하면 이후 install들이 전부 막힘(WSL 셋업 때
# 처음 겪은 문제).
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -y -n trellis python=3.10 \
    && conda clean -afy

# 이후 RUN들은 전부 trellis 환경 안에서 실행되도록(매번 conda run 안 써도 됨).
SHELL ["conda", "run", "-n", "trellis", "/bin/bash", "-c"]

WORKDIR /opt/TRELLIS

# ── TRELLIS 자체 (requirements-trellis.txt가 안내하는 공식 설치 경로) ──
RUN git clone --depth 1 https://github.com/microsoft/TRELLIS.git .

# mkl 버전을 먼저 고정해둔다 — setup.sh가 설치하는 cudatoolkit/pytorch 스택이
# 최신 mkl(2025.x)을 끌고 오면서 torch의 iJIT_NotifyEvent 심볼이 깨지는
# 문제를 WSL에서 실제로 겪었음(사후에 강제 재설치로 고쳤던 걸, 여기선 아예
# 먼저 고정해서 처음부터 문제가 안 생기게 함).
RUN conda install -y -c conda-forge "mkl=2023.1.0" \
    && conda clean -afy

# TRELLIS 공식 setup 스크립트 — requirements-trellis.txt에 적어둔 것과 동일한
# 플래그 조합(우리가 실제로 검증한 조합).
RUN bash ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# transformers 5.x가 torch>=2.5를 요구해서 PyTorch 백엔드가 꺼져버리는 문제를
# 겪어서 고정.
RUN pip install "transformers==4.46.3"

# ── FastAPI 서비스 코드 ──
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# TRELLIS 저장소(/opt/TRELLIS)를 import 경로에 추가 — pip install -e 대신
# PYTHONPATH로 노출(공식 setup.sh가 이미 이 방식을 전제로 함).
ENV PYTHONPATH=/opt/TRELLIS:$PYTHONPATH
ENV TRELLIS_DATA_DIR=/app/data

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "trellis", \
     "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
