# Hosting the nanikiru game
# receive: what to cut
# sent: pais, shanten, remaining pais, remaining rounds
# init => reset => step => ... => step => reset => ...

import random
from typing import *
from shanten import *

class Gaming():
    def __init__(self):
        self.state_dim = 34 + 3 + 1
        self.action_dim = 34
        self.shanten_calculator = ShantenCalculator()
        
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
        return self.calc_state()
        
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
    def shoupai_pu_to_text(shoupai) -> str:
        # 手牌数据 牌谱格式->文本格式
        shoupai_net = Gaming.shoupai_pu_to_net(shoupai)
        shoupai_text = '|' # 开头字符用于后续修正某花色无牌情况
        i = 0
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
        return shoupai_text
    
    def calc_state(self) -> list[int]:
        # [3,3,0,3,3,0,1,0,1, #摸牌后的手牌，例11122244455579m
        #  0,0,0,0,0,0,0,0,0,
        #  0,0,0,0,0,0,0,0,0,
        #  0,0,0,0,0,0,0,
        #  0,3,11, # 三种向听
        #  69 # 余牌数
        #  ]
        # 手牌数据 牌谱格式->张量格式
        shoupai_net = self.shoupai_pu_to_net(self.shoupai)
        # 手牌数据 牌谱格式->文本格式
        shoupai_text = self.shoupai_pu_to_text(self.shoupai)
        # 向听
        shoupai_parsed = self.shanten_calculator.parse_hand(shoupai_text)
        shanten_mianzi: int = self.shanten_calculator.shanten(shoupai_parsed, 1)[0]
        shanten_qidui: int = self.shanten_calculator.shanten(shoupai_parsed, 2)[0]
        shanten_guoshi: int = self.shanten_calculator.shanten(shoupai_parsed, 4)[0]
        shanten: list = [shanten_mianzi, shanten_qidui, shanten_guoshi]
        # 余牌数
        remaining_pai: int = 136 - 14 - self.paishan_array
        # 组合
        state: list = shoupai_net + shanten + [remaining_pai]
        self.log(shoupai_text)
        return state
        
    def step(self, action: int) -> tuple[list, float, bool]:
        # return next_state, reward, done
        this_shanten = self.shanten_calculator.shanten(self.shoupai_pu_to_net(self.shoupai), 7)[0]
        
        # 弃牌，摸牌，下一步手牌
        qipai = action
        for i in range(4):
            if qipai*4+i in self.shoupai:
                self.shoupai.remove(qipai*4+i)
                break
            
        if self.mopai(): # 有余牌
            # 向听，下一步状态
            next_shanten = self.shanten_calculator.shanten(self.shoupai_pu_to_net(self.shoupai), 7)[0]
            
            next_state = self.calc_state()
            # 计算奖励
            if next_shanten == 0: # 自摸
                reward = 2.0
                done = True
            elif next_shanten < this_shanten: # 进向
                reward = 0.5
                done = False
            elif next_shanten == this_shanten: # 不变
                reward = 0.0
                done = False
            elif next_shanten > this_shanten: # 退向
                reward = -0.5
                done = False
        else: # 荒牌流局
            next_state = self.calc_state()
            reward = 1.0 - this_shanten / 2 # 一向听以内正分
            done = True
        
        return (next_state, reward, done)
    def log(self, text: str) -> None:
        print(text)

if __name__ == '__main__':
    gaming = Gaming()
    print(gaming.reset())
    # print(gaming.shoupai)
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
            print(gaming.step(action))
    except KeyboardInterrupt:
        pass