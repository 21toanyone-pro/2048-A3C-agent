# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import plotly as py
import plotly.graph_objs as go
try:
    from plotly.subplots import make_subplots
except ImportError:
    from plotly import tools
    make_subplots = tools.make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


class ActorCritic(nn.Module):
    """개선된 Actor-Critic 네트워크 (더 깊고 넓은 구조)"""
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        # 더 큰 네트워크로 성능 향상
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        policy = self.actor(features)
        value = self.critic(features)
        return policy, value


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.discount_factor = 0.99    # discount rate
        self.no_op_steps = 30

        # optimizer parameters (학습률 스케줄링을 위해 초기값 설정)
        self.actor_lr = 0.0005  # 초기 학습률
        self.critic_lr = 0.0005
        self.initial_lr = 0.0005
        self.threads = 7
        self.episode_count = 0

        # create model for actor and critic network
        self.actor_critic = ActorCritic(state_size, action_size).to(self.device)
        self.local_actor_critic = ActorCritic(state_size, action_size).to(self.device)
        
        # optimizer - 전체 네트워크를 하나의 optimizer로 학습 (학습률 스케줄링 포함)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.actor_lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # update local model
        self.update_localmodel()

        self.episode = 1
        self.__print_period = 100
        self.__save_graph_period = 100
        self.__episode_list = []
        self.__score_list = []
        self.__step_list = []
        self.__Max_reward = []
        self.t_max = 100  # 더 긴 시퀀스 학습 (장기 전략)
        self.t = 0
        self.states = []
        self.rewards = []
        self.actions = []

        self.avg_p_max = 0
        
        # 결과 저장을 위한 디렉토리 생성
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # CSV 파일 초기화
        self.csv_file = os.path.join(self.results_dir, "training_results.csv")
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Score', 'MaxNumber', 'MaxProb', 'Timestamp'])

    def load_model(self, name):
        """모델 로드"""
        try:
            actor_path = name + "_actor.pth"
            critic_path = name + "_critic.pth"
            
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                # 전체 모델 로드 (weights_only=False는 명시적으로 설정하여 경고 제거)
                state_dict = torch.load(actor_path, map_location=self.device, weights_only=False)
                # 네트워크 구조가 일치하는지 확인
                try:
                    self.actor_critic.load_state_dict(state_dict, strict=True)
                    print(f"[OK] Model loaded successfully from {actor_path}")
                    return True
                except RuntimeError as e:
                    print(f"[WARNING] Model structure mismatch:")
                    print(f"  {str(e)[:200]}...")  # 에러 메시지 일부만 출력
                    print("[INFO] Starting with a new model (network structure has changed)")
                    return False
            else:
                print(f"[INFO] Model files not found: {actor_path} or {critic_path}")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("[INFO] Starting with a new model")
            return False

    def save_model(self, name):
        """모델 저장"""
        try:
            actor_path = name + "_actor.pth"
            # 전체 모델 저장 (Actor와 Critic이 같은 네트워크에 있으므로 하나만 저장)
            torch.save(self.actor_critic.state_dict(), actor_path)
            # 호환성을 위해 critic도 저장 (같은 내용)
            torch.save(self.actor_critic.state_dict(), name + "_critic.pth")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def save_learning_result(self, score, maxnumber, Max_prob):
        self.episode += 1
        self.episode_count += 1
        # 학습률 스케줄링 업데이트
        if self.episode_count % 100 == 0:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.episode_count % 1000 == 0:
                print(f"Learning rate: {current_lr:.6f}")
        if (self.episode % self.__print_period == 0):
            print("episode:{0:>8}".format(self.episode), " score:{0:>5}".format(score),
                  "MaxNumber:{0:9}".format(maxnumber), " step:{0:>5}".format(Max_prob))
        # 정보를 저장한다.
        self.__episode_list.append(self.episode)
        self.__score_list.append(score)
        self.__step_list.append(maxnumber)
        self.__Max_reward.append(Max_prob)
        
        # CSV에 기록
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([self.episode, score, maxnumber, Max_prob, timestamp])
        
        # 그래프를 그린다 (Plotly HTML)
        if (self.episode % self.__save_graph_period == 0):
            trace1 = go.Scatter(x=self.__episode_list, y=self.__score_list, name='score')
            trace2 = go.Scatter(x=self.__episode_list, y=self.__step_list, name='MaxNumber')
            trace3 = go.Scatter(x=self.__episode_list, y=self.__Max_reward, name='Max_prob')

            fig = make_subplots(rows=3, cols=2, print_grid=False)
            fig.append_trace(trace1, 1, 1)
            fig.append_trace(trace2, 1, 2)
            fig.append_trace(trace3, 2, 1)

            fig['layout']['xaxis1'].update(title='episode')
            fig['layout']['xaxis2'].update(title='episode')
            fig['layout']['xaxis3'].update(title='episode')

            fig['layout']['yaxis1'].update(title='score')
            fig['layout']['yaxis2'].update(title='MaxNumber')
            fig['layout']['yaxis3'].update(title='Max_prob')

            fig['layout'].update(title='Learning Result')
            html_path = os.path.join(self.results_dir, "learning.html")
            py.offline.plot(fig, html_path, auto_open=False)
            
            # Matplotlib으로 이미지 저장
            self._save_plot_image()
    
    def _save_plot_image(self):
        """학습 곡선을 이미지로 저장"""
        if len(self.__episode_list) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('2048 A3C Learning Results', fontsize=16, fontweight='bold')
        
        # Score 그래프
        axes[0, 0].plot(self.__episode_list, self.__score_list, 'b-', linewidth=1.5, alpha=0.7)
        axes[0, 0].set_title('Score', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True, alpha=0.3)
        if len(self.__score_list) > 0:
            mean_score = np.mean(self.__score_list)
            max_score = np.max(self.__score_list)
            max_idx = np.argmax(self.__score_list)
            max_episode = self.__episode_list[max_idx]
            
            axes[0, 0].axhline(y=mean_score, color='r', linestyle='--', 
                              label=f'Mean: {mean_score:.2f}')
            axes[0, 0].axhline(y=max_score, color='g', linestyle='--', 
                              label=f'Max: {max_score:.2f} (Ep {max_episode})')
            axes[0, 0].plot(max_episode, max_score, 'go', markersize=10, 
                          label=f'Best: {max_score:.2f}')
            axes[0, 0].legend()
        
        # MaxNumber 그래프
        axes[0, 1].plot(self.__episode_list, self.__step_list, 'g-', linewidth=1.5, alpha=0.7)
        axes[0, 1].set_title('Max Number Achieved', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Max Number (2^n)')
        axes[0, 1].grid(True, alpha=0.3)
        if len(self.__step_list) > 0:
            mean_max = np.mean(self.__step_list)
            max_max = np.max(self.__step_list)
            max_max_idx = np.argmax(self.__step_list)
            max_max_episode = self.__episode_list[max_max_idx]
            
            axes[0, 1].axhline(y=mean_max, color='r', linestyle='--', 
                              label=f'Mean: {mean_max:.2f}')
            axes[0, 1].axhline(y=max_max, color='g', linestyle='--', 
                              label=f'Max: {max_max:.0f} (Ep {max_max_episode})')
            axes[0, 1].plot(max_max_episode, max_max, 'go', markersize=10, 
                          label=f'Best: {max_max:.0f}')
            axes[0, 1].legend()
        
        # Max Probability 그래프
        axes[1, 0].plot(self.__episode_list, self.__Max_reward, 'm-', linewidth=1.5, alpha=0.7)
        axes[1, 0].set_title('Max Probability', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Max Prob')
        axes[1, 0].grid(True, alpha=0.3)
        if len(self.__Max_reward) > 0:
            axes[1, 0].axhline(y=np.mean(self.__Max_reward), color='r', linestyle='--', 
                              label=f'Mean: {np.mean(self.__Max_reward):.4f}')
            axes[1, 0].legend()
        
        # Score 이동 평균
        if len(self.__score_list) > 10:
            window = min(50, len(self.__score_list) // 10)
            moving_avg = np.convolve(self.__score_list, np.ones(window)/window, mode='valid')
            episode_avg = self.__episode_list[window-1:]
            axes[1, 1].plot(self.__episode_list, self.__score_list, 'b-', alpha=0.3, label='Raw')
            axes[1, 1].plot(episode_avg, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
            axes[1, 1].set_title('Score with Moving Average', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Need more data\nfor moving average', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Score with Moving Average', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        image_path = os.path.join(self.results_dir, "learning_curve.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning curve saved to {image_path}")

    def discount_rewards(self, rewards, done=True):
        """보상 할인 계산"""
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done and len(self.states) > 0:
            state_tensor = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = self.actor_critic(state_tensor)
                running_add = value.cpu().numpy()[0, 0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def memory(self, state, action, reward):
        """상태, 행동, 보상 저장"""
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def train_episode(self, done):
        """에피소드별 학습"""
        if len(self.states) == 0:
            return
            
        discounted_rewards = self.discount_rewards(self.rewards, done)
        
        # NumPy 배열을 PyTorch 텐서로 변환
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.actions)).to(self.device)
        discounted_rewards_tensor = torch.FloatTensor(discounted_rewards).to(self.device)
        
        # Forward pass
        policy, values = self.actor_critic(states_tensor)
        # values는 [batch_size, 1] 형태이므로 마지막 차원만 제거하여 [batch_size]로 만듦
        values = values.squeeze(-1)  # 마지막 차원만 제거 (더 안전)
        
        # Advantages 계산
        advantages = discounted_rewards_tensor - values.detach()
        
        # Actor loss (policy gradient with entropy)
        action_probs = torch.sum(policy * actions_tensor, dim=1)
        log_probs = torch.log(action_probs + 1e-10)
        actor_loss = -torch.sum(log_probs * advantages)
        
        # Entropy regularization (exploration 증가)
        entropy = -torch.sum(policy * torch.log(policy + 1e-10), dim=1)
        entropy_loss = -torch.sum(entropy)
        
        total_actor_loss = actor_loss + 0.1 * entropy_loss  # 엔트로피 계수 증가 (더 많은 탐험)
        
        # Critic loss - 차원 일치 확인 및 수정
        # values: [batch_size], discounted_rewards_tensor: [batch_size]로 맞춤
        if values.dim() == 0:  # 스칼라인 경우 (batch_size=1일 때)
            values = values.unsqueeze(0)
        if discounted_rewards_tensor.dim() == 0:
            discounted_rewards_tensor = discounted_rewards_tensor.unsqueeze(0)
        if values.dim() != discounted_rewards_tensor.dim():
            # 차원이 다르면 맞춤
            if values.dim() < discounted_rewards_tensor.dim():
                values = values.unsqueeze(-1)
            else:
                discounted_rewards_tensor = discounted_rewards_tensor.unsqueeze(-1)
        
        # 둘 다 같은 차원이 되었으므로 squeeze하여 1D로 만듦
        values = values.squeeze()
        discounted_rewards_tensor = discounted_rewards_tensor.squeeze()
        
        critic_loss = nn.MSELoss()(values, discounted_rewards_tensor)
        
        # Total loss (두 loss를 합쳐서 한 번에 backward)
        total_loss = total_actor_loss + critic_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 메모리 초기화
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        """정책에 따라 행동 선택"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor_critic.eval()
        with torch.no_grad():
            policy, _ = self.actor_critic(state_tensor)
        self.actor_critic.train()
        policy_np = policy.cpu().numpy()[0]
        return np.random.choice(self.action_size, 1, p=policy_np)[0]

    def build_localmodel(self):
        """로컬 모델 생성 (현재는 사용하지 않지만 호환성을 위해 유지)"""
        self.local_actor_critic = ActorCritic(self.state_size, self.action_size).to(self.device)
        self.update_localmodel()
        return self.local_actor_critic, self.local_actor_critic

    def update_localmodel(self):
        """로컬 모델 업데이트"""
        self.local_actor_critic.load_state_dict(self.actor_critic.state_dict())
