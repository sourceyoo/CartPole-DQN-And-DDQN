# DQN & DDQN Algorithms for Open-AI gym Cart pole
Implementation for DQN (Deep Q Network) and DDQN (Double Deep Q Networks) algorithms proposed in 

"Mnih, V., Kavukcuoglu, K., Silver, D. *et al.* Human-level control through deep reinforcement learning.                    *Nature* **518,** 529–533 (2015). https://doi.org/10.1038/nature14236"

and

"Hado van Hasselt, Arthur Guez, David Silver. Deep Reinforcement Learning with Double Q-learning https://arxiv.org/abs/1509.06461"

on Open-AI gym Cart Pole environment.

Also a fraction of pole's base distance to center and pole's angle from center were added as a cost in order to encourage model to keep the pole still and in center. Adding this short term cost should help agent to learn avoiding distance from center and increasing angle (which is the final goal) faster. Although removing these costs won't make it impossible for agent to learn, just makes it harder; This means training takes longer and agent's behaviour becomes less predictable and less stable.

Both methods of training create and save policy model in the same manner, therefore model parameters created by either one of training methods can be used for the Run file.



📘 CartPole DQN / DDQN Reinforcement Learning Project
📌 Overview

본 프로젝트는 OpenAI Gym CartPole-v1 환경에서
DQN(Deep Q-Network) 및 DDQN(Double DQN) 알고리즘을 직접 구현하고
PyTorch 기반으로 학습(Training) 및 시각화(Simulation) 를 수행하는 강화학습 프로젝트이다.

특히 학습 환경(Docker) 과 시뮬레이션 환경(Host venv) 을 완전히 분리하여

안정적인 의존성

GPU 기반 학습

실시간 GUI 렌더링

을 모두 만족하도록 설계되어 있다.

📁 Project Structure
CartPole-DQN-And-DDQN/
│
├── Train_DQN.py          # DQN 학습 코드
├── Train_DDQN.py         # DDQN 학습 코드
├── Model.py              # 신경망 구조 정의
├── play_dqn.py           # 학습된 DQN 시연
├── play_ddqn.py          # 학습된 DDQN 시연
│
├── dqn_cartpole.pth      # 학습 완료 DQN 모델
├── ddqn_cartpole.pth     # 학습 완료 DDQN 모델
├── policy_net.pth        # DDQN 최고성능(best) 정책 네트워크
│
└── README.md

🧩 1. 학습 환경 / 시뮬레이션 환경 분리 구조

본 프로젝트는 다음 두 환경에서 동작한다.

목적	환경	방식	설명
Training	Docker	docker run ...	GPU 학습 / 의존성 고정
Visualization	Host Python venv	source vis_env/bin/activate	실시간 렌더링, GUI 표시
🟦 Simulation Environment (Host venv)

CartPole GUI 렌더링은 Docker X11 제약을 피하기 위해
Ubuntu Host Python virtualenv에서 실행한다.

가상환경 활성화
source vis_env/bin/activate

설치되는 주요 패키지

gymnasium

torch

numpy

pygame

기타 렌더링 관련 패키지

이 환경에서 학습된 .pth 모델을 로드하여 실시간 게임 플레이 시연을 한다.

🐳 Training Environment (Docker)

학습은 Docker 컨테이너에서 수행하며, GPU를 안정적으로 사용한다.

학습용 컨테이너 실행
docker run -it --gpus all \
  -v $(pwd)/CartPole-DQN-And-DDQN:/app \
  cartpole-dqn-env


Docker 내부에는 다음 패키지가 설치되어 있다:

gym==0.25.2 (DQN/DDQN 코드와 호환)

numpy<2.0

PyTorch (CUDA)

matplotlib, tqdm 등

모델이 저장되는 /app/*.pth 파일은 호스트에도 자동 반영된다.

🔧 2. Installation
✔ 2.1 Clone Repository
git clone https://github.com/<your-id>/CartPole-DQN-And-DDQN.git
cd CartPole-DQN-And-DDQN

✔ 2.2 Create Simulation venv
python3 -m venv vis_env
source vis_env/bin/activate

pip install --upgrade pip
pip install torch gymnasium pygame numpy

✔ 2.3 Build Docker Image (Training)
Dockerfile 예시
FROM python:3.10-slim

RUN apt-get update && apt-get install -y python3-opengl

RUN pip install --upgrade pip
RUN pip install "numpy<2.0" gym==0.25.2 torch matplotlib tqdm

WORKDIR /app
CMD ["/bin/bash"]

이미지 빌드
docker build -t cartpole-dqn-env .

🏋️ 3. Training
▶ DQN 학습
docker run -it --gpus all \
  -v $(pwd)/CartPole-DQN-And-DDQN:/app \
  cartpole-dqn-env


컨테이너 안에서:

python Train_DQN.py


➡ 생성 파일: dqn_cartpole.pth

▶ DDQN 학습

컨테이너 안에서:

python Train_DDQN.py


➡ 생성 파일:

ddqn_cartpole.pth

policy_net.pth (best reward)

🎬 4. Simulation (Real-time Visualization)
호스트 가상환경 실행
source vis_env/bin/activate

▶ DQN 시연
python play_dqn.py

▶ DDQN 시연
python play_ddqn.py

📂 5. Model Files
파일명	설명
dqn_cartpole.pth	DQN 학습 최종 모델
ddqn_cartpole.pth	DDQN 학습 최종 모델
policy_net.pth	DDQN 최고성능 모델(best policy)