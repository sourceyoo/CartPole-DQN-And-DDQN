import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from Model import Model
import gym
from collections import deque
import random

# Parameters
use_cuda = False
episode_limit = 300
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

optimizer = torch.optim.Adam(params=policy_net.parameters(), lr=learning_rate)


# 고속 벡터 연산으로 변경 (for문 제거)
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
        # explore
        action = random.randrange(0, n_actions)
    else:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # 차원 추가
        with torch.no_grad(): # 예측 시에는 그래디언트 계산 끔 (속도 향상)
            action = policy_net(state).argmax().item()

    return action


def fit(model, inputs, labels):
    inputs = inputs.to(device)
    labels = labels.to(device)

    model.train()
    
    # 예측
    out = model(inputs)
    loss = criterion(out, labels)
    
    # 업데이트 (전역 변수 optimizer 사용)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    return loss.item()


def optimize_model(train_batch_size=128):
    if len(memory) < train_batch_size:
        return

    train_sample = random.sample(memory, train_batch_size)

    state = get_states_tensor(train_sample, 0)
    next_state = get_states_tensor(train_sample, 3)

    # 1. 현재 상태의 예측값 미리 계산 (q_estimates)
    # detach() 중요: 정답지를 만드는 재료이므로 기울기 계산에서 제외
    q_estimates = policy_net(state.to(device)).detach()
    next_state_q_estimates = target_net(next_state.to(device)).detach()

    # 2. DDQN 로직: 행동 선택 (Policy Net)
    next_actions = policy_net(next_state.to(device)).detach().argmax(dim=1)

    # 3. 정답 계산 (Vectorization)
    batch = list(zip(*train_sample))
    rewards = torch.tensor([state_reward(s, r) for s, r in zip(next_state, batch[2])], device=device)
    actions = torch.tensor(batch[1], device=device).unsqueeze(1)
    
    # 4. Target Value 계산
    next_q_values = next_state_q_estimates.gather(1, next_actions.unsqueeze(1)).squeeze(1)
    expected_q_values = rewards + (gamma * next_q_values)

    # 5. q_estimates 업데이트 (여기가 수정됨!)
    # 아까처럼 q_targets를 새로 만들지 않고, 위에서 만든 q_estimates에 바로 덮어씌웁니다.
    q_estimates.scatter_(1, actions, expected_q_values.unsqueeze(1))

    # 6. 학습
    epochs = 10
    for _ in range(epochs):
        # 완성된 정답지(q_estimates)를 넣고 학습
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
    normalize_state(state)
    done = False
    score = 0
    reward = 0
    while not done:
        # [수정 5] 테스트는 실력으로만 (epsilon=0)
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

    # ---- 학습된 DDQN 모델 저장 (추가한 부분) ----
    save_path = "ddqn_cartpole.pth"  # 원하면 "ddqn_cartpole.pth"로 바꿔도 됨
    torch.save(policy_net.state_dict(), save_path)
    print(f"Saved trained DDQN model to {save_path}")


if __name__ == '__main__':
    main()

