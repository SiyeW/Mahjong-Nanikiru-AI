# Hosting the nanikiru game
# receive: what to cut
# sent: pais, shanten, remaining pais, remaining rounds

import random
from typing import *
from shanten import *

class Gaming():
    def __init__(self):
        # 局数，东1~南4对应0~7，无本场
        self.ju = 0
        # 东家0南家1西家2北家3
        self.side = 0
        # 所有牌共34种136枚，配牌52枚，王牌14枚，可供东家摸牌17巡。
        self.paishan = self.generate_paishan()
        self.paishan_array = 0
        # 配牌
        self.shoupai = self.generate_peipai(self.side)
        # 摸打过程
        self.moda()
    def reset(self):
        self.ju = 0
        self.side = 0
        self.paishan = self.generate_paishan()
        self.paishan_array = 0
        self.peipai = self.generate_peipai(self.side)
        self.moda()
    @staticmethod
    def generate_paishan() -> Tuple[int, ...]:
        # 生成初始列表，每4个数字表示一门牌，赤牌为最小的数字
        numbers = [i for i in range(136)]
        # 打乱列表
        random.seed()
        random.shuffle(numbers)
        return tuple(numbers)
    def generate_peipai(self, side: int) -> Tuple[int, ...]:
        # (0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,1,2,3)
        peipai = []
        for _ in range(3):
            peipai += self.paishan[self.paishan_array: self.paishan_array+4]
            self.paishan_array += 4*4
        peipai.append(self.paishan[self.paishan_array+side])
        self.paishan_array += 4
        peipai.sort()
        return tuple(peipai)
    def mopai(self):
        pass
    def dapai(self):
        pass
    def moda(self):
        pass
    def zimo(self):
        pass
    
if __name__ == '__main__':
    gaming = Gaming()
    print(gaming.peipai)