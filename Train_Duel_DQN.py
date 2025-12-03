import os
import random
import csv
import gym
import numpy as np
import torch
from torch import nn
from collections import deque
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# [중요] 사용자의 모델 파일 이름에 맞춰 import 하세요.
# Dueling Network 구조가 포함된 모델이어야 합니다.
# ---------------------------------------------------------
from D3QN_Model import Model 

# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------
use_cuda = True
episode_limit = 300
target_update_delay = 10
test_delay = 10
learning_rate = 1e-3  # 학습률
epsilon = 1.0
min_epsilon = 0.1
# epsilon_decay: 스텝 단위로 감소 (약 2500 스텝 동안 0.9 감소)
epsilon_decay = 0.9 / 2.5e3 
gamma = 0.90
memory_len = 10000
train_batch_size = 128
#train_epochs 변수 제거 (DQN은 1회만 업데이트해야 함)

# ---------------------------------------------------------
# Environment & Setup
# ---------------------------------------------------------
env = gym.make("CartPole-v1")
n_features = len(env.observation_space.high)
n_actions = env.action_space.n

memory = deque(maxlen=memory_len)
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()

# Dueling Network Model Setup
policy_net = Model(n_features, n_actions).to(device)
target_net = Model(n_features, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
LOG_DIR = "csv"
MODEL_DIR = "Duel_DDQN_Result"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("image", exist_ok=True)

log_filename = os.path.join(LOG_DIR, "duel_dqn_training_log.csv")
graph_filename = "image/duel_dqn_training_result.png"

def init_log_file():
    with open(log_filename, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "score", "reward", "avg_loss", "avg_q", "epsilon"])

def save_log(ep, score, reward, avg_loss, avg_q, eps):
    with open(log_filename, "a", newline="") as f:
        csv.writer(f).writerow([ep, score, reward, avg_loss, avg_q, eps])

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
def normalize_state(state):
    state[0] /= 2.5
    state[1] /= 2.5
    state[2] /= 0.3
    state[3] /= 0.3

def state_reward(state, env_reward):
    # 커스텀 보상: 중심에 가까울수록, 똑바로 서 있을수록 더 큰 점수
    return env_reward - (abs(state[0]) + abs(state[2])) / 2.5

def get_states_tensor(sample, idx):
    return torch.tensor(np.array([x[idx] for x in sample]), dtype=torch.float32)

def get_action(state, e=min_epsilon):
    if random.random() < e:
        return random.randrange(0, n_actions)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        # Dueling Network를 통해 계산된 Q값 중 최대값 선택
        return policy_net(state).argmax().item()

# ---------------------------------------------------------
# Optimization Logic (Corrected)
# ---------------------------------------------------------
def optimize_model():
    if len(memory) < train_batch_size:
        return 0.0, 0.0
    
    # 1. 메모리에서 배치 샘플링
    batch = random.sample(memory, train_batch_size)

    state = get_states_tensor(batch, 0).to(device)
    next_state = get_states_tensor(batch, 3).to(device)
    
    # 배치 데이터 분리
    actions = torch.tensor([b[1] for b in batch], device=device).unsqueeze(1)
    rewards = torch.tensor([state_reward(ns, r) for ns, r in zip(get_states_tensor(batch, 3), [b[2] for b in batch])],
                           device=device)

    # -------------------------------------------------------
    # Double DQN Logic (Action Selection: Policy / Value: Target)
    # -------------------------------------------------------
    
    # 2. 현재 상태의 Q값 계산 (Gradient 계산 필요)
    # Dueling 구조는 forward 내부에서 V + A 로 자동 계산됨
    q_values = policy_net(state)
    curr_q = q_values.gather(1, actions).squeeze(1)
    
    # 모니터링용 평균 Q값
    avg_q_val = q_values.detach().mean().item()

    # 3. 타겟 Q값 계산 (No Gradient)
    with torch.no_grad():
        # (A) 행동 선택은 Policy Net이 함
        next_actions = policy_net(next_state).argmax(dim=1, keepdim=True)
        # (B) 가치 평가는 Target Net이 함
        next_q_values = target_net(next_state).gather(1, next_actions).squeeze(1)
        
        # 타겟값 = 보상 + 할인율 * 미래가치
        expected_q = rewards + gamma * next_q_values

    # -------------------------------------------------------
    # [수정] 반복문(Epochs) 제거 -> 한 번만 업데이트!
    # -------------------------------------------------------
    optimizer.zero_grad()
    loss = criterion(curr_q, expected_q) # expected_q는 이미 detach 상태
    loss.backward()
    optimizer.step()

    return loss.item(), avg_q_val

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_one_episode(eps):
    state = env.reset()
    if isinstance(state, tuple): state = state[0]
    normalize_state(state)
    done = False
    score = reward_sum = 0
    episode_loss = episode_q = count = 0

    while not done:
        action = get_action(state, eps)
        step = env.step(action)
        
        # Gym 버전에 따른 언패킹 처리
        if len(step) == 5:
            next_state, env_reward, terminated, truncated, _ = step
            done = terminated or truncated
        else:
            next_state, env_reward, done, _ = step

        normalize_state(next_state)
        memory.append((state, action, env_reward, next_state))
        
        state = next_state
        score += env_reward
        reward_sum += state_reward(next_state, env_reward)

        # 학습 수행
        loss, q = optimize_model()
        
        if loss != 0.0:
            episode_loss += loss
            episode_q += q
            count += 1

        # Epsilon Decay (스텝마다 감소)
        eps = max(min_epsilon, eps - epsilon_decay)

    avg_loss = episode_loss / count if count else 0.0
    avg_q = episode_q / count if count else 0.0
    
    return score, reward_sum, avg_loss, avg_q, eps

def test():
    state = env.reset()
    if isinstance(state, tuple): state = state[0]
    normalize_state(state)
    done = False
    score = reward_sum = 0
    while not done:
        # 테스트는 무조건 Greedy (Epsilon=0)
        action = get_action(state, e=0.0)
        step = env.step(action)
        if len(step) == 5:
            state, env_reward, terminated, truncated, _ = step
            done = terminated or truncated
        else:
            state, env_reward, done, _ = step
        normalize_state(state)
        score += env_reward
        reward_sum += state_reward(state, env_reward)
    return score, reward_sum

def save_plots(history):
    plt.figure(figsize=(15, 15))
    keys = [("score", "Score", "blue"), ("reward", "Reward", "magenta"),
            ("avg_q", "Avg Q", "green"), ("loss", "Loss", "orange"), ("epsilon", "Epsilon", "purple")]
    
    for i, (k, title, color) in enumerate(keys, 1):
        plt.subplot(3, 2, i)
        data = history[k]
        plt.plot(data, label=k, color=color, alpha=0.7)
        
        # 이동 평균선 (데이터가 충분할 때만)
        if len(data) >= 10:
            ma = np.convolve(data, np.ones(10)/10, mode="valid")
            plt.plot(range(9, len(data)), ma, label="10-Avg", color="red", linewidth=1.5)
            
        plt.title(title)
        plt.xlabel("Episode")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(graph_filename)
    print(f"Graph saved to {graph_filename}")
    plt.close()

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    global epsilon
    best = -float("inf")
    init_log_file()
    
    history = {k: [] for k in ["score", "reward", "avg_q", "loss", "epsilon"]}

    print("Training Started with Dueling Double DQN (Corrected)...")
    try:
        for ep in range(1, episode_limit + 1):
            score, reward, avg_loss, avg_q, epsilon = train_one_episode(epsilon)
            
            history["score"].append(score)
            history["reward"].append(reward)
            history["avg_q"].append(avg_q)
            history["loss"].append(avg_loss)
            history["epsilon"].append(epsilon)
            
            save_log(ep, score, reward, avg_loss, avg_q, epsilon)
            print(f"Ep {ep}: Score={score:.0f} | Reward={reward:.2f} | Q={avg_q:.2f} | Loss={avg_loss:.4f} | Eps={epsilon:.2f}")

            if ep % target_update_delay == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if ep % test_delay == 0:
                ts, tr = test()
                print(f"--- Test {ep}: Score={ts:.1f} Reward={tr:.2f}")
                if tr > best:
                    best = tr
                    best_path = os.path.join(MODEL_DIR, "duel_dqn_cartpole_best.pth")
                    torch.save(policy_net.state_dict(), best_path)
                    print(f">>> New best model saved! Reward: {best:.2f}")
                    
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    print(f"Final Best Reward: {best}")
    final_path = os.path.join(MODEL_DIR, "duel_dqn_cartpole_final.pth")
    torch.save(policy_net.state_dict(), final_path)
    save_plots(history)

if __name__ == "__main__":
    main()