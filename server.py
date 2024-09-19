# Hosting the nanikiru game
# receive: what to cut
# sent: pais, shanten, remaining pais, remaining rounds
# init => reset => step => ... => step => reset => ...

import random
from typing import *
from shanten import *

class Gaming():
    def __init__(self):
        self.state_dim = 4*34+4
        self.action_dim = 34
        self.shanten_calculator = ShantenCalculator()
        self.training_log: bool = False
        
    def reset(self) -> list[int]:
        # 局数，东1~南4对应0~7，无本场
        self.ju = 0
        # 东家0南家1西家2北家3
        self.side = 0
        # 自摸即结束
        # self.done = False
        # 所有牌共34种136枚，配牌52枚，王牌14枚，可供东家摸牌70枚17巡。
        self.paishan = self.generate_paishan()
        self.paishan_array = 0 # 0~69
        # 配牌
        self.shoupai: list = self.generate_peipai(self.side)
        self.zimopai: int = None
        # # 摸打过程
        self.mopai()
        # 返回当前张量状态
        state = self.calc_state()
        return state
        
    @staticmethod
    def generate_paishan() -> tuple[int, ...]:
        # 生成初始列表，每4个数字表示一门牌，赤牌为最小的数字
        numbers = [i for i in range(136)]
        # 打乱列表
        random.seed()
        random.shuffle(numbers)
        return tuple(numbers)
    
    def generate_peipai(self, side: int) -> list[int]:
        # (0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,1,2,3)
        peipai = []
        for _ in range(3):
            peipai += self.paishan[self.paishan_array: self.paishan_array+4]
            self.paishan_array += 4*4
        peipai.append(self.paishan[self.paishan_array+side])
        self.paishan_array += 4
        peipai.sort() # 手牌必须排序
        return peipai
    
    def mopai(self) -> bool:
        # False: game end
        self.prev_shoupai = self.shoupai # 记录上一步手牌，用于计算向听
        
        self.paishan_array += 3 # 他家摸牌计数
        remaining_pai = 136 - 14 - self.paishan_array
        if remaining_pai <= 0:
            return False
        new_pai: int = self.paishan[self.paishan_array]
        self.paishan_array += 1 # 自家摸牌计数
        self.shoupai = self.shoupai + [new_pai]
        self.shoupai.sort() # 手牌必须排序
        return True
        
    @staticmethod
    def shoupai_pu_to_net(shoupai) -> list[int]:
        # 手牌数据 牌谱格式->张量格式
        shoupai_net: list = [0]*34
        for cell in shoupai:
            shoupai_net[cell//4] += 1 #对应牌的数量+1
        return shoupai_net
    @staticmethod
    def shoupai_pu_to_onehot(shoupai) -> list[int]: #shape: [4, 34]
        # 手牌数据 牌谱格式->独热格式
        shoupai_onehot: list[list[int]] = [[0]*34]*4
        for cell in shoupai:
            for i in range(4):
                if not(shoupai_onehot[i][cell//4]): #对应牌的数量n维位置无牌
                    shoupai_onehot[i][cell//4] = 1
                    break
                else:
                    continue
            # else:
            #     raise ValueError # 一门牌有第五张
        return shoupai_onehot
    @staticmethod
    def shoupai_pu_to_text(shoupai, visualize: bool=False) -> str:
        # 手牌数据 牌谱格式->文本格式
        shoupai_net = Gaming.shoupai_pu_to_net(shoupai)
        i = 0
        if not(visualize): # 使用常规记牌
            shoupai_text = '|' # 开头字符用于后续修正某花色无牌情况
            while i < 9:
                shoupai_text = shoupai_text + str(i+1)*shoupai_net[i]
                i+=1
            shoupai_text = shoupai_text + 'm'
            while i < 18:
                shoupai_text = shoupai_text + str(i+1-9)*shoupai_net[i]
                i+=1
            shoupai_text = shoupai_text + 'p'
            while i < 27:
                shoupai_text = shoupai_text + str(i+1-18)*shoupai_net[i]
                i+=1
            shoupai_text = shoupai_text + 's'
            while i < 34:
                shoupai_text = shoupai_text + str(i+1-27)*shoupai_net[i]
                i+=1
            shoupai_text = shoupai_text + 'z'
            # 修正某花色无牌情况
            shoupai_text = shoupai_text.replace('sz', 's')
            shoupai_text = shoupai_text.replace('ps', 'p')
            shoupai_text = shoupai_text.replace('mp', 'm')
            shoupai_text = shoupai_text.replace('|m', '|')
            shoupai_text = shoupai_text.replace('|', '')
        else: # 使用可视化
            characters = '🀇🀈🀉🀊🀋🀌🀍🀎🀏🀙🀚🀛🀜🀝🀞🀟🀠🀡🀐🀑🀒🀓🀔🀕🀖🀗🀘🀀🀁🀂🀃🀆🀅🀄'
            shoupai_text = ''
            while i < 34:
                shoupai_text = shoupai_text + characters[i]*shoupai_net[i]
                i+=1
        return shoupai_text
    
    def calc_state(self) -> list[int]:
        # [3,3,0,3,3,0,1,0,1, #摸牌后的手牌，例11122244455579m
        #  0,0,0,0,0,0,0,0,0,
        #  0,0,0,0,0,0,0,0,0,
        #  0,0,0,0,0,0,0,
        #  0,3,11, # 三种向听
        #  69 # 余牌数
        #  ]
        # 日志
        self.training_log_shoupai(self.shoupai)
        # 手牌数据 牌谱格式->张量格式
        shoupai_net = self.shoupai_pu_to_net(self.shoupai)
        # 手牌数据 牌谱格式->独热格式
        shoupai_onehot = self.shoupai_pu_to_onehot(self.shoupai)
        # 手牌数据 牌谱格式->文本格式
        shoupai_text = self.shoupai_pu_to_text(self.shoupai)
        prev_shoupai_text = self.shoupai_pu_to_text(self.prev_shoupai)
        # 摸牌前向听
        prev_shoupai_parsed = self.shanten_calculator.parse_hand(prev_shoupai_text)
        prev_shanten_mianzi: int = self.shanten_calculator.shanten(prev_shoupai_parsed, 1)[0][0]
        prev_shanten_qidui: int = self.shanten_calculator.shanten(prev_shoupai_parsed, 2)[0][0]
        prev_shanten_guoshi: int = self.shanten_calculator.shanten(prev_shoupai_parsed, 4)[0][0]
        shanten: list = [prev_shanten_mianzi, prev_shanten_qidui, prev_shanten_guoshi]
        # 余牌数
        remaining_pai: int = 136 - 14 - self.paishan_array
        # 组合成state
        state: list[int] = [item for sublist in shoupai_onehot for item in sublist] + shanten + [remaining_pai]
        return state
        
    def step(self, action: int) -> tuple[list, float, bool]:
        # return next_state, reward, done
        
        # 日志
        self.training_log_action(action)
        
        # 行为不合法
        if not(self.shoupai_pu_to_net(self.shoupai)[action]):
            reward = -10.0
            done = False # 重下一次
            next_state = self.calc_state()
            
            self.training_log_reward(reward)
            return (next_state, reward, done)
        
        # 准备撤销打牌
        shoupai_backup = self.shoupai.copy()
        prev_shoupai_backup = self.prev_shoupai.copy()
        
        # 弃牌，摸牌，下一步手牌
        qipai = action
        for i in range(4):
            if qipai*4+i in self.shoupai:
                self.shoupai.remove(qipai*4+i)
                break
        
        (this_shanten_mianzi, _), (this_jinzhang_mianzi, _) = self.shanten_calculator.shanten(self.shoupai_pu_to_net(self.prev_shoupai), 1)
        (this_shanten, _), (this_jinzhang, _) = self.shanten_calculator.shanten(self.shoupai_pu_to_net(self.prev_shoupai), 1)
        
        if self.mopai(): # 有余牌
            # 打牌后向听，下一步状态
            (next_shanten_mianzi, _), (next_jinzhang_mianzi, _) = self.shanten_calculator.shanten(self.shoupai_pu_to_net(self.prev_shoupai), 1)
            (next_shanten, _), (next_jinzhang, _) = self.shanten_calculator.shanten(self.shoupai_pu_to_net(self.prev_shoupai), 1)
            
            next_state = self.calc_state()
            # 计算奖励
            reward = 0
            if next_shanten == 0: # 自摸
                reward += 3.0
                done = True
            elif next_shanten < this_shanten: # 进向
                reward += 0.1
                if next_shanten_mianzi < this_shanten_mianzi:
                    reward += 0.1 # 面子手进向额外加分
                done = False
            elif next_shanten == this_shanten: # 不变
                reward += 0.00
                # 进张
                if next_jinzhang_mianzi > this_jinzhang_mianzi:
                    reward += 0.005
                elif next_jinzhang_mianzi == this_jinzhang_mianzi:
                    reward += 0.00
                else:
                    reward += -0.01
                    self.shoupai = shoupai_backup # 撤销打牌
                    self.prev_shoupai = prev_shoupai_backup # 撤销打牌
                done = False
            elif next_shanten > this_shanten: # 退向
                reward += -0.3
                done = False
                self.shoupai = shoupai_backup # 撤销打牌
                self.prev_shoupai = prev_shoupai_backup # 撤销打牌
                
            self.training_log_text(f"{this_shanten},{this_jinzhang} -> {next_shanten},{next_jinzhang} | {reward:6.3f}")
        
        else: # 荒牌流局
            next_state = self.calc_state()
            reward = 0.5 if this_shanten <= 2 else 0.0 # 一向听以内正分
            done = True
            
            self.training_log_reward(reward)
        
        return (next_state, reward, done)
    
    def training_log_shoupai(self, shoupai) -> None:
        if self.training_log:
            text = Gaming.shoupai_pu_to_text(shoupai, True)
            print(text, end=' ')
    def training_log_action(self, action) -> None:
        if self.training_log:
            text = Gaming.shoupai_pu_to_text((action*4,), True)
            print(text)
    def training_log_reward(self, reward) -> None:
        if self.training_log:
            print(f'{reward:.3f}')
    def training_log_text(self, text) -> None:
        if self.training_log:
            print(text, end='')

if __name__ == '__main__':
    gaming = Gaming()
    gaming.training_log = True
    gaming.reset()
    # print(gaming.reset())
    try:
        while True:
            decision = input()
            if decision[1] == 'm':
                action = int(decision[0])-1
            elif decision[1] == 'p':
                action = int(decision[0])-1+9
            elif decision[1] == 's':
                action = int(decision[0])-1+18
            elif decision[1] == 'z':
                action = int(decision[0])-1+27
            # print(gaming.step(action))
            gaming.step(action)
    except KeyboardInterrupt:
        pass