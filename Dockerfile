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
# libgl1/libglib2.0-0: trellis 패키지가 import 시점에 open3d를 로드하는데,
# open3d가 libGL.so.1(OpenGL)을 dlopen함 — devel 베이스 이미지엔 이게 없어서
# "OSError: libGL.so.1: cannot open shared object file"로 죽었던 걸 확인하고
# 추가(헤드리스 컨테이너라 실제 GL 렌더링은 안 하지만, 라이브러리 자체는 있어야
# import가 됨).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
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
# --recurse-submodules --shallow-submodules: trellis/representations/mesh/
# flexicubes가 git submodule(MaxtirError/FlexiCubes)인데 --depth 1만 쓰면
# 서브모듈이 아예 안 받아져서 빈 디렉터리로 남고, 나중에 import 시점에
# "No module named 'trellis.representations.mesh.flexicubes.flexicubes'"로
# 죽는 걸 확인해서 추가.
RUN git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/microsoft/TRELLIS.git .

# mkl 버전을 먼저 고정해둔다 — setup.sh가 설치하는 cudatoolkit/pytorch 스택이
# 최신 mkl(2025.x)을 끌고 오면서 torch의 iJIT_NotifyEvent 심볼이 깨지는
# 문제를 WSL에서 실제로 겪었음(사후에 강제 재설치로 고쳤던 걸, 여기선 아예
# 먼저 고정해서 처음부터 문제가 안 생기게 함).
RUN conda install -y -c conda-forge "mkl=2023.1.0" \
    && conda clean -afy

# torch를 먼저 직접 설치한다. setup.sh는 --new-env가 켜져 있을 때만
# `conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8`을
# 실행하는데, 여기선 conda env를 위에서 이미 직접 만들어서(mkl 고정 순서
# 때문에) --new-env 없이 setup.sh를 불렀더니 이 설치가 통째로 스킵되고,
# 그 아래 `python -c "import torch"`부터 조용히 깨지면서 --xformers/
# --flash-attn/--spconv/--kaolin/--nvdiffrast까지 전부 설치가 안 되는
# 채로 빌드가 "성공"해버렸다(GPU 머신에서 실제 배포 후 "No module named
# 'torch'"로 뒤늦게 발견). setup.sh --new-env가 설치했을 것과 동일한
# 버전으로 여기서 미리 깔아둔다.
RUN conda install -y -c pytorch -c nvidia pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 \
    && conda clean -afy

# TRELLIS 공식 setup 스크립트 — --basic만 여기서 돌린다. xformers/spconv/
# kaolin(과 flash-attn/nvdiffrast/diffoctreerast/mipgaussian)은 setup.sh가
# `torch.cuda.is_available()`로 platform을 cuda/cpu 판별해서 그 값에 따라
# 설치를 켜고 끄는데, `docker build` 단계는 (docker-compose.yml의 GPU
# reservation이 런타임에만 적용되고 build엔 GPU가 안 붙어서) 항상
# cuda_is_available()=False라서 PLATFORM이 "cpu"로 잘못 판별되고
# "[XFORMERS] Unsupported platform: cpu" 식으로 전부 조용히 스킵돼버리는 걸
# 확인함(빌드는 에러 없이 "성공"으로 끝나서 눈치채기 어려움). 그래서 아래에서
# CUDA 분기 커맨드를 setup.sh 소스 그대로 베껴서 platform 감지에 기대지 않고
# 직접 실행한다(우리가 위에서 고정한 torch==2.4.0/cu118 기준).
RUN bash ./setup.sh --basic

# xformers/spconv/kaolin — setup.sh의 해당 PYTORCH_VERSION=2.4.0,
# CUDA_VERSION=11.8 분기와 동일한 커맨드.
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install spconv-cu118 \
    && pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# 소스에서 CUDA 확장을 컴파일해야 하는 4개(flash-attn, nvdiffrast,
# diffoctreerast, mip-splatting의 diff-gaussian-rasterization)는 setup.sh가
# --no-build-isolation 없이 pip install을 호출한다 — 최신 pip의 기본 빌드
# 격리 때문에 그 안의 격리된 빌드 환경에서는 이미 설치된 torch가 안 보여서
# "ModuleNotFoundError: No module named 'torch'"로 전부 조용히 실패하고
# (setup.sh에 set -e가 없어서 스크립트 자체는 안 멈추고 빌드가 "성공"으로
# 끝나버림), 실제 배포 후 그제서야 발견됨. --no-build-isolation을 직접 줘서
# 재설치한다(git clone 경로/URL은 setup.sh와 동일).
#
# 4개를 한 RUN에 묶지 않고 패키지별로 분리했다 — 하나로 묶으면 그중 하나만
# 고쳐도 Docker 레이어 캐시가 전부 무효화돼서 이미 성공한(플래시어텐션은 소스
# 컴파일에 2시간 넘게 걸림) 앞 패키지까지 매번 처음부터 다시 빌드해야 했다.
RUN pip install psutil
# psutil: flash-attn의 setup.py가 메타데이터 생성 시점에 병렬 빌드 job 수를
# 정하려고 직접 import한다 — --no-build-isolation을 줘도 이건 원래 env에도
# 안 깔려있어서 "ModuleNotFoundError: No module named 'psutil'"로 따로 막힘.
RUN pip install flash-attn --no-build-isolation

# nvdiffrast/diffoctreerast/mip-splatting은(flash-attn과 달리) torch의 표준
# CUDAExtension 빌드 경로를 타는데, 이게 컴파일 대상 GPU 아키텍처를 자동
# 감지하려고 한다 — 근데 build 단계엔 GPU가 안 보이니(위 xformers/spconv/
# kaolin 문제와 동일한 원인) 감지된 아키텍처 목록이 빈 채로 남아서
# `torch/utils/cpp_extension.py`의 `arch_list[-1] += '+PTX'`가
# "IndexError: list index out of range"로 죽는다. TORCH_CUDA_ARCH_LIST를
# 직접 지정해서 자동 감지에 기대지 않게 한다(Volta~Ampere 범위 + PTX로 그
# 이후 세대도 JIT로 커버 — 실제 배포 GPU 아키텍처를 알면 그것 하나로 좁혀도
# 됨, 빌드가 더 빨라짐).
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

RUN mkdir -p /tmp/extensions \
    && git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast \
    && pip install /tmp/extensions/nvdiffrast --no-build-isolation

RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast \
    && pip install /tmp/extensions/diffoctreerast --no-build-isolation

RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting \
    && pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation \
    && rm -rf /tmp/extensions

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
