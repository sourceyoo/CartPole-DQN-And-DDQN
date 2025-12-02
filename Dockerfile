FROM python:3.10-slim

# 기본 패키지 (렌더링용 opengl 포함)
RUN apt-get update && apt-get install -y \
    python3-opengl \
    && rm -rf /var/lib/apt/lists/*

# pip 및 핵심 패키지 설치
RUN pip install --upgrade pip

# gym 0.25.x는 numpy 1.x와만 호환되므로 2.x 대신 1.x 사용
RUN pip install numpy==1.23.5

# gym 0.25.2 (옛날 API용) + 기타 유틸
RUN pip install "gym==0.25.2" matplotlib tqdm

# CPU 버전 PyTorch (CartPole 수준이면 충분)
RUN pip install torch

# 작업 디렉토리
WORKDIR /app

CMD ["/bin/bash"]

