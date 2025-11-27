import ddqn as dqn
import py2048
import numpy as np
import torch
from collections import deque
import time
import os

print("=" * 80)
print("2048 A3C Full Training Test")
print("=" * 80)

step = 0
height = 4
width = 4
board = py2048.GameBoard(height=height, width=width)

# 상태 크기: 16 (기본) + 1 (빈 칸 비율) + 2 (최대 타일 위치) + 2 (monotonicity) = 21
agent = dqn.DQNAgent(state_size = 21, action_size = 4)
agent.discount_factor = 0.99  # 할인 계수 증가

# 테스트용으로 적절한 스텝 수 설정 (실제 학습은 더 길게)
n_training_moves = 200000  # 더 많은 학습으로 2048 달성 목표
max_episodes = 1000  # 최대 에피소드 수 증가

def calculateReward(state_raw, next_state_raw, action, done, board):
    """
    대폭 개선된 보상 함수 - 2048 달성을 위한 전략적 보상:
    1. 빈 칸 증가 (타일 병합 시 큰 보상)
    2. 점수 증가 (스케일 조정)
    3. 최대 타일 값 증가 (큰 보상)
    4. Monotonicity (단조성) - 큰 타일이 모서리에 있을수록 좋음
    5. Smoothness (부드러움) - 인접 타일의 차이가 작을수록 좋음
    6. 게임 종료 페널티
    """
    reward = 0.0
    
    # state_raw와 next_state_raw는 지수 값 (0, 1, 2, ...)이므로 2^값으로 변환
    current_values = 2 ** state_raw
    next_values = 2 ** next_state_raw
    
    # 1. 빈 칸 증가 보상 (타일 병합 시 - 매우 중요!)
    empty_diff = (next_state_raw == 0).sum() - (state_raw == 0).sum()
    reward += empty_diff * 1.0  # 보상 증가
    
    # 2. 점수 증가 보상 (타일 값의 합)
    current_sum = current_values.sum()
    next_sum = next_values.sum()
    score_increase = next_sum - current_sum
    reward += score_increase * 0.0001  # 정규화된 점수 증가
    
    # 3. 최대 타일 값 증가 보상 (매우 중요!)
    current_max = state_raw.max() if len(state_raw) > 0 else 0
    next_max = next_state_raw.max() if len(next_state_raw) > 0 else 0
    if next_max > current_max:
        # 최대 타일이 클수록 더 큰 보상 (지수적 증가)
        reward += (next_max - current_max) * (2.0 ** (next_max - 10))  # 1024 이상에서 큰 보상
    
    # 4. Monotonicity (단조성) - 큰 타일이 모서리/가장자리에 있을수록 좋음
    exp_board = board.exponentiate()
    if exp_board.max() > 0:
        # 최대 타일의 위치 확인
        max_pos = np.unravel_index(exp_board.argmax(), exp_board.shape)
        # 모서리에 있으면 보상
        if max_pos[0] in [0, 3] and max_pos[1] in [0, 3]:
            reward += 0.5
        # 가장자리에 있으면 작은 보상
        elif max_pos[0] in [0, 3] or max_pos[1] in [0, 3]:
            reward += 0.2
        
        # Monotonicity 점수 계산 (행/열이 정렬되어 있는지)
        for i in range(4):
            row = exp_board[i, :]
            col = exp_board[:, i]
            # 행이 감소하는 방향으로 정렬되어 있으면 보상
            if np.all(row[:-1] >= row[1:]) or np.all(row[1:] >= row[:-1]):
                reward += 0.1
            # 열이 감소하는 방향으로 정렬되어 있으면 보상
            if np.all(col[:-1] >= col[1:]) or np.all(col[1:] >= col[:-1]):
                reward += 0.1
    
    # 5. Smoothness (부드러움) - 인접 타일의 차이가 작을수록 좋음
    exp_board = board.exponentiate()
    smoothness_penalty = 0.0
    for i in range(4):
        for j in range(4):
            if exp_board[i, j] > 0:
                # 인접 타일과의 차이 계산
                neighbors = []
                if i > 0:
                    neighbors.append(exp_board[i-1, j])
                if i < 3:
                    neighbors.append(exp_board[i+1, j])
                if j > 0:
                    neighbors.append(exp_board[i, j-1])
                if j < 3:
                    neighbors.append(exp_board[i, j+1])
                
                for neighbor in neighbors:
                    if neighbor > 0:
                        diff = abs(exp_board[i, j] - neighbor)
                        # 차이가 작을수록 좋음 (로그 스케일)
                        if diff > 0:
                            smoothness_penalty += np.log2(diff + 1) * 0.01
    
    reward -= smoothness_penalty
    
    # 6. 게임 종료 페널티
    if done:
        reward -= 50.0  # 게임 종료에 큰 페널티
    
    # 7. 보드가 가득 찰 위험도 (빈 칸이 적을수록 페널티)
    empty_count = (next_state_raw == 0).sum()
    if empty_count <= 2:
        reward -= 1.0  # 위험 상태 페널티
    
    return reward

def calculateScore(board,exp = False):
    if exp:
        board = board.exponentiate()
    else:
        board = board.board
    return board.flatten().sum()

smoothness = 1500

scores = deque(maxlen = 4000)
expScores = deque(maxlen = 4000)

print(f"\nDevice: {agent.device}")
print(f"Training moves: {n_training_moves}")
print(f"Max episodes: {max_episodes}")
print(f"Results will be saved to: {agent.results_dir}/")
print("-" * 80)

model_loaded = agent.load_model("./save_model/py2048_a3c")
if not model_loaded:
    print("\n[INFO] Starting training with a new model")
    print("       (Previous model incompatible due to network structure changes)")

start_time = time.time()
episode_count = 0

print("\nStarting training...")
print("-" * 80)

for i in range(n_training_moves):
    # 상태 정규화 개선: 로그 스케일 + 보드 구조 특징 추가
    state_raw = board.board.flatten()
    state_log = np.log2(state_raw + 1) / 12.0  # 최대값 2^12 = 4096을 고려한 정규화
    
    # 보드 구조 특징 추가 (monotonicity, smoothness 등)
    exp_board = board.exponentiate()
    features = []
    
    # 1. 기본 로그 스케일 상태
    features.extend(state_log)
    
    # 2. 빈 칸 비율
    empty_ratio = (state_raw == 0).sum() / 16.0
    features.append(empty_ratio)
    
    # 3. 최대 타일 위치 (정규화)
    if exp_board.max() > 0:
        max_pos = np.unravel_index(exp_board.argmax(), exp_board.shape)
        features.append(max_pos[0] / 3.0)
        features.append(max_pos[1] / 3.0)
    else:
        features.extend([0.0, 0.0])
    
    # 4. 행/열 monotonicity 점수
    row_mono = 0.0
    col_mono = 0.0
    for i in range(4):
        row = exp_board[i, :]
        col = exp_board[:, i]
        if np.all(row[:-1] >= row[1:]) or np.all(row[1:] >= row[:-1]):
            row_mono += 0.25
        if np.all(col[:-1] >= col[1:]) or np.all(col[1:] >= col[:-1]):
            col_mono += 0.25
    features.append(row_mono)
    features.append(col_mono)
    
    state = np.array(features, dtype=np.float32)
    
    action = agent.get_action(state)
    board.performAction(action)
    
    next_state_raw = board.board.flatten()
    next_state_log = np.log2(next_state_raw + 1) / 12.0
    
    # 다음 상태의 특징도 동일하게 계산
    next_exp_board = board.exponentiate()
    next_features = []
    next_features.extend(next_state_log)
    next_empty_ratio = (next_state_raw == 0).sum() / 16.0
    next_features.append(next_empty_ratio)
    if next_exp_board.max() > 0:
        next_max_pos = np.unravel_index(next_exp_board.argmax(), next_exp_board.shape)
        next_features.append(next_max_pos[0] / 3.0)
        next_features.append(next_max_pos[1] / 3.0)
    else:
        next_features.extend([0.0, 0.0])
    next_row_mono = 0.0
    next_col_mono = 0.0
    for i in range(4):
        row = next_exp_board[i, :]
        col = next_exp_board[:, i]
        if np.all(row[:-1] >= row[1:]) or np.all(row[1:] >= row[:-1]):
            next_row_mono += 0.25
        if np.all(col[:-1] >= col[1:]) or np.all(col[1:] >= col[:-1]):
            next_col_mono += 0.25
    next_features.append(next_row_mono)
    next_features.append(next_col_mono)
    next_state = np.array(next_features, dtype=np.float32)
    
    done = int(board.gameOver)
    reward = calculateReward(state_raw, next_state_raw, action, done, board)
    
    agent.memory(state, action, reward)
    
    # Calculate max probability for current state
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    agent.actor_critic.eval()
    with torch.no_grad():
        policy, _ = agent.actor_critic(state_tensor)
    agent.actor_critic.train()
    agent.avg_p_max += torch.max(policy).cpu().item()
    
    step += 1
    agent.t += 1
    
    # 주기적으로 모델 저장
    if i % 5000 == 0 and i > 0:
        agent.save_model("./save_model/py2048_a3c")
        elapsed = time.time() - start_time
        print(f"\n[Step {i}] Model saved. Elapsed time: {elapsed:.1f}s")
        if len(scores) > 0:
            print(f"  Recent avg score: {np.mean(list(scores)[-min(50, len(scores)):]):.2f}")
    
    if agent.t >= agent.t_max or done:
        agent.train_episode(done)
        agent.update_localmodel()
        agent.t = 0
    
    if done:
        episode_count += 1
        MaxQ = (agent.avg_p_max / float(step)) if step > 0 else 0
        current_score = calculateScore(board)
        scores.append(current_score)
        expScores.append(calculateScore(board,exp=True))
        MaxNumber = 2 ** board.Max_number()
        
        # 주기적으로 상세 정보 출력
        if episode_count % 10 == 0 or episode_count <= 5:
            avg_score = np.mean(list(scores)[-min(smoothness, len(scores)):])
            avg_exp = np.mean(list(expScores)[-min(smoothness, len(expScores)):])
            elapsed = time.time() - start_time
            print(f"\n[Episode {episode_count}] Step {i}")
            print(f"  Score: {current_score:.1f} | Avg Score: {avg_score:.1f}")
            print(f"  ExpScore: {calculateScore(board,exp=True):.1f} | Avg ExpScore: {avg_exp:.1f}")
            print(f"  MaxNumber: {MaxNumber} {'*** 2048 ACHIEVED! ***' if MaxNumber >= 2048 else ''}")
            if MaxNumber >= 4096:
                print(f"  *** AMAZING! 4096 ACHIEVED! ***")
            print(f"  MaxProb: {MaxQ:.4f}")
            print(f"  Time: {elapsed:.1f}s")
            print("-" * 80)
        
        # 2048 달성 시 특별 메시지
        if MaxNumber >= 2048 and episode_count % 10 != 0:
            print(f"\n*** MILESTONE: 2048 achieved in episode {episode_count}! ***")
        
        if MaxNumber >= 4096:
            print(f"\n*** AMAZING: 4096 achieved in episode {episode_count}! ***")
        
        board.reset()
        agent.save_learning_result(
            np.mean(list(scores)[-min(smoothness, len(scores)):]) if len(scores) > 0 else 0, 
            MaxNumber, 
            MaxQ
        )
        step = 0
        agent.avg_p_max = 0
        
        # 최대 에피소드 수 도달 시 종료
        if episode_count >= max_episodes:
            print(f"\n✓ Reached maximum episodes ({max_episodes})")
            break

total_time = time.time() - start_time

print("\n" + "=" * 80)
print("Training completed!")
print("=" * 80)
print(f"Total steps: {i+1}")
print(f"Total episodes: {episode_count}")
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"Average time per episode: {total_time/episode_count:.2f}s" if episode_count > 0 else "N/A")

if len(scores) > 0:
    print(f"\nFinal Statistics:")
    print(f"  Average score: {np.mean(scores):.2f}")
    print(f"  Best score: {max(scores):.2f}")
    print(f"  Average exp score: {np.mean(expScores):.2f}")
    print(f"  Best exp score: {max(expScores):.2f}")
    if len(scores) >= 10:
        print(f"  Recent 10 avg: {np.mean(list(scores)[-10:]):.2f}")
    
    # 최고 달성 타일 확인 (agent의 step_list에서)
    try:
        max_tile = max(agent._DQNAgent__step_list) if len(agent._DQNAgent__step_list) > 0 else 0
        print(f"\n  Maximum tile achieved: {max_tile:.0f}")
        if max_tile >= 2048:
            print(f"  *** SUCCESS: 2048 tile achieved! ***")
        if max_tile >= 4096:
            print(f"  *** EXCELLENT: 4096 tile achieved! ***")
        if max_tile >= 8192:
            print(f"  *** OUTSTANDING: 8192 tile achieved! ***")
    except:
        pass

print(f"\nResults saved to:")
print(f"  - CSV: {agent.csv_file}")
print(f"  - Image: {os.path.join(agent.results_dir, 'learning_curve.png')}")
print(f"  - HTML: {os.path.join(agent.results_dir, 'learning.html')}")
print("=" * 80)

