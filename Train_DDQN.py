import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from Model import Model
import gym
from collections import deque
import random
import numpy as np
import csv      # [추가] CSV 저장을 위해
import matplotlib.pyplot as plt # [추가] 그래프 그리기 위해
import os

# Parameters
use_cuda = True
episode_limit = 300
target_update_delay = 10
test_delay = 10
learning_rate = 0.001
epsilon = 1
min_epsilon = 0.1
epsilon_decay = 0.9 / 2.5e3
gamma = 0.99
memory_len = 10000

env = gym.make('CartPole-v1')
n_features = len(env.observation_space.high)
n_actions = env.action_space.n

memory = deque(maxlen=memory_len)
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()

policy_net = Model(n_features, n_actions).to(device)
target_net = Model(n_features, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(params=policy_net.parameters(), lr=learning_rate)

# --- [추가] 로그 파일 설정 ---
log_filename = "ddqn_training_log.csv"
graph_filename = "ddqn_training_result.png"

def init_log_file():
    """CSV 파일 생성 및 헤더 작성"""
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'score', 'reward', 'avg_loss', 'avg_q', 'epsilon'])

def save_log(episode, score, reward, avg_loss, avg_q, epsilon):
    """한 줄씩 로그 저장"""
    with open(log_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, score, reward, avg_loss, avg_q, epsilon])

# ----------------------------

def get_states_tensor(sample, states_idx):
    batch_data = [x[states_idx] for x in sample]
    return torch.tensor(batch_data, dtype=torch.float32)

def normalize_state(state):
    state[0] /= 2.5
    state[1] /= 2.5
    state[2] /= 0.3
    state[3] /= 0.3

def state_reward(state, env_reward):
    return env_reward - (abs(state[0]) + abs(state[2])) / 2.5

def get_action(state, e=min_epsilon):
    if random.random() < e:
        action = random.randrange(0, n_actions)
    else:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state).argmax().item()
    return action

def fit(model, inputs, labels):
    inputs = inputs.to(device)
    labels = labels.to(device)

    model.train()
    out = model(inputs)
    loss = criterion(out, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    return loss.item()

def optimize_model(train_batch_size=128):
    if len(memory) < train_batch_size:
        return 0.0, 0.0 # [수정] 학습 안 함 -> 0 반환

    train_sample = random.sample(memory, train_batch_size)

    state = get_states_tensor(train_sample, 0)
    next_state = get_states_tensor(train_sample, 3)

    # [수정] 모니터링을 위해 Q값 저장
    q_estimates = policy_net(state.to(device)).detach()
    avg_q_val = q_estimates.mean().item() # 현재 평균 Q값

    next_state_q_estimates = target_net(next_state.to(device)).detach()
    next_actions = policy_net(next_state.to(device)).detach().argmax(dim=1)

    batch = list(zip(*train_sample))
    rewards = torch.tensor([state_reward(s, r) for s, r in zip(next_state, batch[2])], device=device)
    actions = torch.tensor(batch[1], device=device).unsqueeze(1)
    
    next_q_values = next_state_q_estimates.gather(1, next_actions.unsqueeze(1)).squeeze(1)
    expected_q_values = rewards + (gamma * next_q_values)

    q_estimates.scatter_(1, actions, expected_q_values.unsqueeze(1))

    # [수정] Loss 평균 계산
    total_loss = 0
    epochs = 10
    for _ in range(epochs):
        loss = fit(policy_net, state, q_estimates)
        total_loss += loss

    return total_loss / epochs, avg_q_val

def train_one_episode():
    global epsilon
    current_state = env.reset()
    if isinstance(current_state, tuple):
        current_state = current_state[0]
    normalize_state(current_state)
    done = False
    score = 0
    reward = 0
    
    # [수정] 통계 변수
    episode_loss = 0
    episode_q = 0
    optimize_count = 0
    
    while not done:
        action = get_action(current_state, epsilon)
        
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, env_reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, env_reward, done, _ = step_result
            
        normalize_state(next_state)
        memory.append((current_state, action, env_reward, next_state))
        
        current_state = next_state
        score += env_reward
        reward += state_reward(next_state, env_reward)

        # [수정] 반환값 받기
        loss, q_val = optimize_model(100)
        
        if loss != 0.0:
            episode_loss += loss
            episode_q += q_val
            optimize_count += 1

        if epsilon > min_epsilon:
            epsilon -= epsilon_decay

    # [수정] 평균 계산
    avg_loss = episode_loss / optimize_count if optimize_count > 0 else 0
    avg_q = episode_q / optimize_count if optimize_count > 0 else 0

    return score, reward, avg_loss, avg_q

def test():
    state = env.reset()
    if isinstance(state, tuple): state = state[0]
    normalize_state(state)
    done = False
    score = 0
    reward = 0
    while not done:
        action = get_action(state, e=0.0)
        step_result = env.step(action)
        if len(step_result) == 5:
            state, env_reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            state, env_reward, done, _ = step_result
        normalize_state(state)
        score += env_reward
        reward += state_reward(state, env_reward)
    return score, reward

# --- [수정된 그래프 그리기 함수] ---
def save_plots(history):
    # 크기 키움 (가로 15, 세로 15)
    plt.figure(figsize=(15, 15))

    # 1. Score (생존 시간)
    plt.subplot(3, 2, 1)
    plt.plot(history['score'], label='Score', color='blue', alpha=0.6)
    if len(history['score']) >= 10:
        moving_avg = np.convolve(history['score'], np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(history['score'])), moving_avg, label='10-Avg', color='red')
    plt.title('1. Episode Score (Total Steps)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Reward (안정성 점수) - [추가됨!]
    plt.subplot(3, 2, 2)
    plt.plot(history['reward'], label='Reward', color='magenta', alpha=0.6)
    if len(history['reward']) >= 10:
        moving_avg_r = np.convolve(history['reward'], np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(history['reward'])), moving_avg_r, label='10-Avg', color='darkred')
    plt.title('2. Episode Reward (Quality)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Avg Q
    plt.subplot(3, 2, 3)
    plt.plot(history['avg_q'], label='Avg Q', color='green')
    plt.title('3. Average Max Q-Value')
    plt.xlabel('Episode')
    plt.ylabel('Q Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Loss
    plt.subplot(3, 2, 4)
    plt.plot(history['loss'], label='Loss', color='orange')
    plt.title('4. Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Epsilon
    plt.subplot(3, 2, 5)
    plt.plot(history['epsilon'], label='Epsilon', color='purple')
    plt.title('5. Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(graph_filename)
    print(f"Graph saved to {graph_filename}")
    plt.close()

def main():
    best_test_reward = -float('inf')
    init_log_file()
    
    # [수정] reward도 저장
    history = {'score': [], 'reward': [], 'avg_q': [], 'loss': [], 'epsilon': []}

    print(f"Logging to {log_filename}...")

    try:
        for i in range(episode_limit):
            score, reward, avg_loss, avg_q = train_one_episode()
            
            # 리스트에 추가
            history['score'].append(score)
            history['reward'].append(reward) # [추가]
            history['avg_q'].append(avg_q)
            history['loss'].append(avg_loss)
            history['epsilon'].append(epsilon)

            save_log(i+1, score, reward, avg_loss, avg_q, epsilon)

            print(f'Ep {i+1}: Score:{score:.0f} | Reward:{reward:.2f} | Q:{avg_q:.2f} | Loss:{avg_loss:.4f} | Eps:{epsilon:.2f}')

            if i % target_update_delay == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            if (i + 1) % test_delay == 0:
                test_score, test_reward = test()
                print(f'--- Test Ep {i+1}: Score:{test_score:.1f} | Reward:{test_reward:.2f} ---')
                if test_reward > best_test_reward:
                    print('>>> New best record! Saving model...')
                    best_test_reward = test_reward
                    torch.save(policy_net.state_dict(), 'ddqn_cartpole_best.pth')
    
    except KeyboardInterrupt:
        print("Training stopped manually.")

    print(f'Final Best Reward: {best_test_reward}')
    torch.save(policy_net.state_dict(), "ddqn_cartpole_final.pth")
    
    # [추가] 마지막에 그래프 그리기
    save_plots(history)

if __name__ == '__main__':
    main()