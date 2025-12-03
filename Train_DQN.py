import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from Model import Model
import gym
from collections import deque
import random

# Parameters
use_cuda = False
episode_limit = 100
target_update_delay = 10  # update target net every target_update_delay episodes
test_delay = 10
learning_rate = 1e-3
epsilon = 1  # initial epsilon
min_epsilon = 0.1
epsilon_decay = 0.9 / 2.5e3
gamma = 0.99
memory_len = 10000

env = gym.make('CartPole-v1')
n_features = len(env.observation_space.high)
n_actions = env.action_space.n

memory = deque(maxlen=memory_len)
# each memory entry is in form: (state, action, env_reward, next_state)
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
policy_net = Model(n_features, n_actions).to(device)
target_net = Model(n_features, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 옵티마이저를 전역 변수로 선언 (기억 유지!)
optimizer = torch.optim.Adam(params=policy_net.parameters(), lr=learning_rate)


def get_states_tensor(sample, states_idx):
    # 리스트 컴프리헨션으로 데이터만 쏙 뽑아서 한 번에 텐서로 변환
    # sample 구조: [(state, action, reward, next_state), ...]
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
        # explore
        action = random.randrange(0, n_actions)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state_tensor).argmax().item()

    return action


# [수정 3] 쪼개서 학습하던 부분 제거 -> 통째로 한 번만 학습
def fit(model, inputs, labels):
    inputs = inputs.to(device)
    labels = labels.to(device)

    model.train()
    
    # 1. 100개 데이터를 한 번에 예측
    out = model(inputs)
    
    # 2. 오차 계산
    loss = criterion(out, labels)
    
    # 3. 업데이트 (전역 변수 optimizer 사용)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()

    return loss.item()


def optimize_model(train_batch_size=128):
    # 1. 데이터가 부족하면 그냥 패스 (안전장치)
    if len(memory) < train_batch_size:
        return
    
    train_batch_size = min(train_batch_size, len(memory))
    train_sample = random.sample(memory, train_batch_size)

    state = get_states_tensor(train_sample, 0)
    next_state = get_states_tensor(train_sample, 3)

    q_estimates = policy_net(state.to(device)).detach()
    next_state_q_estimates = target_net(next_state.to(device)).detach()

    for i in range(len(train_sample)):
        q_estimates[i][train_sample[i][1]] = (
            state_reward(next_state[i], train_sample[i][2])
            + gamma * next_state_q_estimates[i].max()
        )

    # 준비된 데이터(state, q_estimates)를 가지고 10번 반복 학습합니다.
    # 예전 코드(5개씩 20번)와 비슷한 학습량을 확보하면서도 훨씬 안정적입니다.
    
    epochs = 10  # 10번 정도 반복 복습
    for _ in range(epochs):
        fit(policy_net, state, q_estimates)


def train_one_episode():
    global epsilon
    current_state = env.reset()
    if isinstance(current_state, tuple):
        current_state = current_state[0]
    normalize_state(current_state)
    done = False
    score = 0
    reward = 0
    while not done:
        action = get_action(current_state, epsilon)
        next_state, env_reward, done, _ = env.step(action)
        normalize_state(next_state)
        memory.append((current_state, action, env_reward, next_state))
        current_state = next_state
        score += env_reward
        reward += state_reward(next_state, env_reward)

        optimize_model(100)

        epsilon -= epsilon_decay

    return score, reward


def test():
    state = env.reset()
    if isinstance(state, tuple): state = state[0]
    normalize_state(state)
    done = False
    score = 0
    reward = 0
    while not done:
        #테스트 때는 무조건 실력으로만 (epsilon=0)
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


def main():
    best_test_reward = 0

    for i in range(episode_limit):
        score, reward = train_one_episode()

        print(f'Episode {i + 1}: score: {score} - reward: {reward}')

        if i % target_update_delay == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()

        if (i + 1) % test_delay == 0:
            test_score, test_reward = test()
            print(f'Test Episode {i + 1}: test score: {test_score} - test reward: {test_reward}')
            if test_reward > best_test_reward:
                print('New best test reward. Saving model')
                best_test_reward = test_reward
                torch.save(policy_net.state_dict(), 'policy_net.pth')

    if episode_limit % test_delay != 0:
        test_score, test_reward = test()
        print(f'Test Episode {episode_limit}: test score: {test_score} - test reward: {test_reward}')
        if test_reward > best_test_reward:
            print('New best test reward. Saving model')
            best_test_reward = test_reward
            torch.save(policy_net.state_dict(), 'policy_net.pth')

    print(f'best test reward: {best_test_reward}')

    # ---- 학습된 모델 저장 (요청한 블록 추가) ----
    save_path = "dqn_cartpole.pth"
    torch.save(policy_net.state_dict(), save_path)
    print(f"Saved trained DQN model to {save_path}")


if __name__ == '__main__':
    main()

