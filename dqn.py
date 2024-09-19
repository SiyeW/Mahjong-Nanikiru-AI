import os
import random
import time
from collections import deque
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import server

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 输入维度：[4行 * 34列]的手牌状态数据
        self.conv_shupai = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=(0, 1)),  # 第一层卷积
            nn.ReLU(),
            # nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(1, 3), stride=1, padding=(0, 1)),  # 第二层卷积，通道数增加
            # nn.ReLU(),
            # nn.Conv2d(in_channels=6, out_channels=9, kernel_size=(1, 3), stride=1, padding=(0, 1)),  # 第三层卷积，最终输出9个通道
            # nn.ReLU(),
            nn.Flatten()
        ).to(self.device)

        # 归一化层
        # self.norm_shupai = nn.BatchNorm2d(1)
        # self.norm_zipai = nn.LayerNorm(7)
        self.norm_shoupai = nn.BatchNorm2d(1).to(self.device)
        self.norm_extra = nn.LayerNorm(4).to(self.device)

        # 2或3层全连接层，将卷积输出与后面的额外数据拼接一起
        self.fc = nn.Sequential(
            nn.Linear(2*34*16+4, 1024),  # 手牌卷积特征9*9*3 + 三个向听 + 余牌数
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim)  # 最后输出的维度为action_dim
        ).to(self.device)

    def forward(self, state):
        
        # 将state拆分为数牌、字牌、向听余牌数三个部分
        state = state.reshape(-1, self.state_dim)
        
        # shupai_state = state[:, :27].reshape(-1, 1, 3, 9)
        # zipai_state = state[:, 27:34].reshape(-1, 7)
        shoupai_state = state[:, :34*4].reshape(-1, 1, 4, 34)
        extra_state = state[:, 34*4:].reshape(-1, 4) # 后面的额外状态

        # 对手牌部分进行归一化
        # shupai_state = self.norm_shupai(shupai_state)
        # zipai_state = self.norm_zipai(zipai_state)
        shoupai_state = self.norm_shoupai(shoupai_state)
        extra_state = self.norm_extra(extra_state)

        # 对手牌进行卷积
        shupai_features = self.conv_shupai(shoupai_state) 
        # shape (-1, 1, 4, 34) -> (-1, 1, 2, 34) -> (-1, 2*34*16)
        shupai_features = shupai_features.reshape(-1, 2*34*16) 
        
        # 将卷积特征和额外状态拼接起来
        combined = torch.cat([shupai_features, extra_state], dim=1)

        # 通过全连接层
        output = self.fc(combined)
        return output

class Agent():
    def __init__(self, state_dim, action_dim, memory_size=10000, batch_size=64, gamma=0.99, lr=5e-6, weight_decay=1e-10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_fn = nn.MSELoss().to(self.device)
        self.steps = 0
        self.writer = SummaryWriter()
        
        # 加载模型
        self.load_model()
        
    def select_action(self, state, eps):
        # state 是一个包含手牌和其他状态信息的向量
        mask = torch.FloatTensor(state[:self.action_dim]).to(self.device)  # state 中非0的部分表示可行动作

        if random.random() < eps:
            env.training_log_text('\n*')
            random_action = random.randint(0, self.action_dim - 1)
            # 屏蔽非法弃牌，通过mask判断合法性
            while mask[random_action] == 0:  # 判断合法性
                random_action = random.randint(0, self.action_dim - 1)
            return random_action
        else:
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state).reshape(-1)
                
                # 将非法动作的Q值设为-inf
                q_values[mask == 0] = float("-inf") # mask == 0 表示非法动作
                # 使用 softmax 函数将 Q 值转换为概率分布，公式 pi = exp(Q/t) / ∑exp(Q/t)
                t = 0.1 # 温度系数
                prob_dist = torch.softmax(q_values / t, dim=0)

                # 输出概率分布中最大值对应的动作
                # action = torch.multinomial(prob_dist, 1).item()
                
                # 使用 torch.sort() 按照从大到小的顺序排序
                sorted_prob, sorted_indices = torch.sort(prob_dist, descending=True)

                # sorted_indices[0] 是最大值的索引，依此类推
                # print(f"Sorted probabilities: {sorted_prob}")
                # print(f"Sorted indices: {sorted_indices}")
                # print(prob_dist)
                
                # 输出日志 max(q_values)
                env.training_log_text('\n')
                for i in range(5):
                    prob = sorted_prob[i]
                    indices =  sorted_indices[i].item()
                    env.training_log_text(f'{prob:.2f}{env.shoupai_pu_to_text((indices*4,), True)} ')
                env.training_log_text("|  ")
                
                # action = q_values.argmax().item()
                # 输出概率分布中最大值对应的动作
                action = torch.multinomial(prob_dist, 1).item()
            return action
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state_batch).max(1)[0]
        # expected_q_values = (1-self.gamma) * reward_batch + self.gamma * next_q_values * (1 - done_batch)
        expected_q_values = self.gamma * reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = self.loss_fn(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        self.writer.add_scalar("Loss", loss.item(), self.steps)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_model(self):
        # 创建model目录，如果不存在的话
        if not os.path.exists('./model/'):
            os.makedirs('./model/')
        
        # 保存模型，文件名为时间戳
        model_path = f"./model/{time.strftime('%m%d%H%M')}_{self.steps}.pth"
        torch.save(self.policy_net.state_dict(), model_path)
        print(f"Model saved at {model_path}")

    def load_model(self):
        # 查找 './model/' 目录中是否有带 "=" 的模型文件
        # model_files = [f for f in os.listdir('./model/') if f.startswith("=")]
        model_files = [f for f in os.listdir('./model/') if '=' in f]
        if model_files:
            model_path = os.path.join('./model/', model_files[0])
            self.policy_net.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        else:
            print("No pre-trained model found, starting fresh.")

# def train_dqn(env, agent, eps_start=1, eps_end=0.1, eps_decay=0.999, max_episodes=10000, max_steps=18):
def train_dqn(env, agent, eps_start=0.6, eps_end=0.4, eps_decay=0.998, max_episodes=10000, max_steps=18):
    eps = eps_start
    for episode in tqdm(range(max_episodes)):
        if episode % 100 == 0:
            env.training_log = True
        else:
            env.training_log = False
        state = env.reset()
        for step in range(max_steps):
            action = agent.select_action(state, eps)

            next_state, reward, done = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.train()
            if done:
                break
        agent.update_target()
        eps = max(eps * eps_decay, eps_end)
    else:
        agent.save_model()
        
if __name__ == "__main__":
    env = server.Gaming()
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = Agent(state_dim, action_dim)
    
    try:
        train_dqn(env, agent)
    except KeyboardInterrupt:
        print("Training interrupted, saving model...")
        agent.save_model()