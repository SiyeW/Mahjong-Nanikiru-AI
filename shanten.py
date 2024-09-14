# 麻将向听数计算器，用于计算给定手牌的向听数。

# 类的主要功能包括：
# - 解析手牌字符串，将其转换为内部表示的手牌格式。
# - 计算不同的向听数，包括面子手、七对子、国士无双。
# - 根据手牌数量确定计算向听数所用的m值。

# 类方法：
# - `parse_hand(hand_str: str) -> Tuple[int, ...]`:
#     解析手牌字符串，将形如"29m167p168s334z"的字符串转换为内部格式的手牌元组。
    
# 实例方法：
# - `__init__()`:
#     初始化 `ShantenCalculator` 实例，并读取组合向听表数据。
    
# - `calc_lh(hand: List[int], m: int) -> int`:
#     计算面子手向听数。
    
# - `calc_sp(hand: List[int]) -> int`:
#     计算七对子向听数。
    
# - `calc_to(hand: List[int]) -> int`:
#     计算国士无双向听数。
    
# - `calc(hand: List[int], mode: int, m: int) -> Tuple[int, int]`:
#     根据模式计算不同的向听数。
    
# - `shanten(hand: List[int], mode: int = 7) -> Tuple[int, int]`:
#     计算手牌的总向听数，默认模式为7（计算所有类型的向听数）。
#     1 计算面子手 2 计算七对 4 计算国士

# 类属性：
# - `INDEX_S_FILE`:
#     数牌组合向听表文件路径。
    
# - `INDEX_H_FILE`:
#     字牌组合向听表文件路径。

import csv
from typing import List, Tuple

class ShantenCalculator:
    INDEX_S_FILE: str = "./index_s.csv"  # 数牌组合向听表
    INDEX_H_FILE: str = "./index_h.csv"  # 字牌组合向听表
    # 所有牌共34种136枚

    def __init__(self):
        self.mp1: List[List[int]] = self.read_csv(self.INDEX_S_FILE)
        self.mp2: List[List[int]] = self.read_csv(self.INDEX_H_FILE)
        
    @staticmethod
    def parse_hand(hand_str: str) -> Tuple[int, ...]:
        # 初始化34张牌的元组，每一张牌的数量从0开始
        hand = [0] * 34
        
        # 花色的结束标识符
        suits = 'mpsz'
        suit_offsets = {
            'm': 0,   # 万子偏移量
            'p': 9,   # 饼子偏移量
            's': 18,  # 索子偏移量
            'z': 27   # 字牌偏移量
        }
        
        # 临时存储当前的牌
        current_tiles = ''
        
        # 遍历输入字符串，解析出各个花色的牌
        for char in hand_str:
            if char in suits:
                # 将前面一段数字解释为某个花色的牌
                if current_tiles:
                    # 将当前解析出的牌按对应花色添加到hand元组中
                    offset = suit_offsets[char]
                    # 将每张牌的数量添加到hand中
                    for tile in current_tiles:
                        tile_num = int(tile)
                        if char == 'z':
                            # 字牌范围为1-7（东、南、西、北、中、发、白），对应索引27-33
                            hand[offset + tile_num - 1] += 1
                        else:
                            # 万、饼、索的范围为1-9，对应不同的偏移量
                            hand[offset + tile_num - 1] += 1
                # 清空临时存储，准备处理下一段牌
                current_tiles = ''
            else:
                current_tiles += char

        return list(hand)

    @staticmethod
    def add1(lhs: List[int], rhs: List[int], m: int) -> List[int]:
        lhs = lhs.copy()  # 创建lhs的副本，避免修改原始lhs
        for j in range(m + 5, 4, -1):
            sht = min(lhs[j] + rhs[0], lhs[0] + rhs[j])
            for k in range(5, j):
                sht = min(sht, lhs[k] + rhs[j - k], lhs[j - k] + rhs[k])
            lhs[j] = sht

        for j in range(m, -1, -1):
            sht = lhs[j] + rhs[0]
            for k in range(j):
                sht = min(sht, lhs[k] + rhs[j - k])
            lhs[j] = sht

        return lhs

    @staticmethod
    def add2(lhs: List[int], rhs: List[int], m: int) -> List[int]:
        lhs = lhs.copy()  # 创建lhs的副本，避免修改原始lhs
        j = m + 5
        sht = min(lhs[j] + rhs[0], lhs[0] + rhs[j])
        for k in range(5, j):
            sht = min(sht, lhs[k] + rhs[j - k], lhs[j - k] + rhs[k])
        lhs[j] = sht
        return lhs

    @staticmethod
    def read_csv(file: str) -> List[List[int]]:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            return [[int(cell) for cell in row if cell] for row in reader]

    @staticmethod
    def accum(v: List[int], start: int, end: int, base: int) -> int:
        for i in range(start, end):
            base = 5 * base + v[i]
        return base

    def calc_lh(self, hand: List[int], m: int) -> int:
        # 计算面子手向听数
        ret = self.mp1[self.accum(hand, 1, 9, hand[0])]
        ret = self.add1(ret, self.mp1[self.accum(hand, 10, 18, hand[9])], m)
        ret = self.add1(ret, self.mp1[self.accum(hand, 19, 27, hand[18])], m)
        ret = self.add2(ret, self.mp2[self.accum(hand, 28, 34, hand[27])], m)
        return ret[5 + m]

    def calc_sp(self, hand: List[int]) -> int:
        # 计算七对子向听数
        pair = sum(1 for i in range(34) if hand[i] >= 2)
        kind = sum(1 for i in range(34) if hand[i] > 0)
        return 7 - pair + (7 - kind if kind < 7 else 0)

    def calc_to(self, hand: List[int]) -> int:
        # 计算国士无双向听数
        pair = sum(1 for i in [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33] if hand[i] >= 2)
        kind = sum(1 for i in [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33] if hand[i] > 0)
        return 14 - kind - (1 if pair > 0 else 0)

    def calc(self, hand: List[int], mode: int) -> Tuple[int, int]:
        # 根据手牌数量确定对应的m，手牌数量异常时m按4处理
        hand_number = sum(hand)
        m_dict = {1:0, 2: 0, 4:1, 5: 1, 7:2, 8: 2, 10:3, 11: 3, 13:4, 14: 4}
        m = m_dict.get(hand_number, 4)
        
        # 根据模式计算不同向听数
        ret = [1024, 0]

        if mode & 1:
            sht = self.calc_lh(hand, m)
            if sht < ret[0]:
                ret = [sht, 1]
            elif sht == ret[0]:
                ret[1] |= 1

        if (mode & 2) and m == 4:
            sht = self.calc_sp(hand)
            if sht < ret[0]:
                ret = [sht, 2]
            elif sht == ret[0]:
                ret[1] |= 2

        if (mode & 4) and m == 4:
            sht = self.calc_to(hand)
            if sht < ret[0]:
                ret = [sht, 4]
            elif sht == ret[0]:
                ret[1] |= 4
                
        return tuple(ret)

    def shanten(self, hand: List[int], mode: int = 7) -> Tuple[tuple[int, int], tuple[int, list[int]]|tuple[None]]:
        # 计算手牌向听数
        shanten = self.calc(hand, mode)
        # 计算手牌进张
        jinzhang = self.jinzhang(hand) if sum(hand) == 13 else ()
        
        return shanten, jinzhang
    
    def jinzhang(self, hand: list[int], mode: int = 7) -> tuple[int, list[int]]:
        hand_number = sum(hand)
        if hand_number not in [1, 4, 7, 10, 13]:
            return -1, [] # 手牌数量不对
        jinzhang_tile = []
        shanten, _ = self.calc(hand, mode)
        for tile in range(34):
            next_hand = hand.copy()
            if hand[tile] < 4:
                next_hand[tile] += 1
                next_shanten, _ = self.calc(next_hand, mode)
                if next_shanten < shanten:
                    # print(next_shanten, shanten, tile)
                    jinzhang_tile.append(tile)
            else:
                continue
        # 计算进张枚数
        jinzhang_number = 0
        for tile in jinzhang_tile:
            jinzhang_number += 4 - hand[tile]
        return jinzhang_number, jinzhang_tile
            

# # Example 1：
# calculator = ShantenCalculator()
# result = calculator.shanten([
#     2, 2, 2, 0, 0, 0, 0, 0, 0,  # Manzu
#     0, 0, 0, 2, 2, 2, 0, 0, 0,  # Pinzu
#     0, 0, 0, 0, 0, 0, 1, 0, 0,  # Souzu
#     0, 0, 0, 0, 0, 0, 0  # Jihai
# ])
# print(result)
# # (向听数+1，对应牌型)

# # Example 2:
calculator = ShantenCalculator()
hand_parsed = ShantenCalculator.parse_hand("1122334456789mp")
print(hand_parsed)
result = calculator.shanten(hand_parsed)
print(result)

# calculator = ShantenCalculator()
# result = calculator.shanten((1, 0, 3, 0, 1, 0, 0, 0, 2, 1, 0, 1, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
# print(result)
# result = calculator.shanten((1, 0, 3, 0, 1, 0, 0, 0, 2, 1, 0, 1, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
# print(result)