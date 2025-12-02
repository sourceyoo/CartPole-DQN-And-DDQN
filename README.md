# DQN & DDQN Algorithms for Open-AI gym Cart pole
Implementation for DQN (Deep Q Network) and DDQN (Double Deep Q Networks) algorithms proposed in 

"Mnih, V., Kavukcuoglu, K., Silver, D. *et al.* Human-level control through deep reinforcement learning.                    *Nature* **518,** 529–533 (2015). https://doi.org/10.1038/nature14236"

and

"Hado van Hasselt, Arthur Guez, David Silver. Deep Reinforcement Learning with Double Q-learning https://arxiv.org/abs/1509.06461"

on Open-AI gym Cart Pole environment.

Also a fraction of pole's base distance to center and pole's angle from center were added as a cost in order to encourage model to keep the pole still and in center. Adding this short term cost should help agent to learn avoiding distance from center and increasing angle (which is the final goal) faster. Although removing these costs won't make it impossible for agent to learn, just makes it harder; This means training takes longer and agent's behaviour becomes less predictable and less stable.

Both methods of training create and save policy model in the same manner, therefore model parameters created by either one of training methods can be used for the Run file.



📘 README.md — CartPole DQN / DDQN Reinforcement Learning Project
📌 Overview

본 프로젝트는 OpenAI Gym의 CartPole-v1 환경에서
DQN(Deep Q-Network) 및 DDQN(Double DQN) 알고리즘을 직접 구현하고,
PyTorch 기반으로 학습·시연하는 강화학습 프로젝트입니다.

학습(Training)과 시뮬레이션(Visualization)은
완전히 분리된 환경(Docker / Host venv) 에서 실행되며,
안정적인 재현성과 GUI 렌더링 성능을 모두 확보할 수 있도록 설계되었습니다.

📁 Project Structure
CartPole-DQN-And-DDQN/
│
├── Train_DQN.py               # DQN 학습 코드
├── Train_DDQN.py              # DDQN 학습 코드
├── Model.py                   # 신경망 구조 정의
├── play_dqn.py                # 학습된 DQN 시뮬레이션
├── play_ddqn.py               # 학습된 DDQN 시뮬레이션
│
├── dqn_cartpole.pth           # 학습 완료된 DQN 모델
├── ddqn_cartpole.pth          # 학습 완료된 DDQN 모델
│
└── README.md

🧩 1. 학습 환경 / 시뮬레이션 환경 분리 구조

본 프로젝트는 다음 두 가지 환경으로 나뉘어 실행됩니다.

목적	환경	방식	설명
Training (학습)	Docker	docker run --gpus all ...	GPU 안정 사용, 의존성 고정
Visualization (시뮬레이션)	Host Python venv	source vis_env/bin/activate	실시간 CartPole GUI 렌더링
🟦 Simulation Environment (Host venv)

시뮬레이션은 GUI 렌더링이 필요하므로
Ubuntu Host Python 가상환경에서 실행한다.

# 가상환경 활성화
source vis_env/bin/activate


이 환경에는 다음 패키지가 포함된다:

gymnasium

pygame

torch

numpy

기타 시뮬레이션 관련 패키지

여기서 학습 완료된 .pth 모델을 로드하여 실시간 CartPole 제어를 시연한다.

🐳 Training Environment (Docker)

학습은 Docker 컨테이너 안에서 실행되며 GPU를 안정적으로 활용한다.

docker run -it --gpus all \
  -v $(pwd)/CartPole-DQN-And-DDQN:/app \
  cartpole-dqn-env


이 환경에는 다음 패키지가 고정 버전으로 설치됨:

gym==0.25.2 (DQN/DDQN 코드와 호환)

numpy<2.0

torch (CUDA 지원)

matplotlib, tqdm 등

컨테이너는 호스트의 프로젝트 폴더(/app)를 공유하므로
학습 후 생성된 .pth 모델이 자동으로 호스트에도 동기화된다.

🔧 2. Installation
✔ 2.1 Clone Repository
git clone https://github.com/<user>/<repo>.git
cd CartPole-DQN-And-DDQN

✔ 2.2 Create Simulation venv
python3 -m venv vis_env
source vis_env/bin/activate
pip install --upgrade pip
pip install torch gymnasium pygame numpy

✔ 2.3 Build Docker Image (Training)

Dockerfile 예시:

FROM python:3.10-slim

RUN apt-get update && apt-get install -y python3-opengl
RUN pip install --upgrade pip
RUN pip install "numpy<2.0" gym==0.25.2 torch matplotlib tqdm

WORKDIR /app
CMD ["/bin/bash"]


이미지 빌드:

docker build -t cartpole-dqn-env .

🏋️ 3. Training
▶ DQN 학습
docker run -it --gpus all \
  -v $(pwd)/CartPole-DQN-And-DDQN:/app \
  cartpole-dqn-env

python Train_DQN.py


결과:

dqn_cartpole.pth 생성됨

▶ DDQN 학습
python Train_DDQN.py


결과:

ddqn_cartpole.pth 생성됨
policy_net.pth (best test reward 기준) 생성됨

🎬 4. Simulation (Real-time Visualization)

Host 가상환경 실행:

source vis_env/bin/activate

▶ DQN 시연
python play_dqn.py


실행 화면:

CartPole 환경이 GUI로 표시

저장된 DQN 모델이 자동으로 행동 선택

episode별 reward 출력

▶ DDQN 시연
python play_ddqn.py

📂 5. Model Files
파일명	의미
dqn_cartpole.pth	DQN 학습 최종 모델
ddqn_cartpole.pth	DDQN 학습 최종 모델
policy_net.pth	DDQN 테스트 최고성능 모델(best test reward)