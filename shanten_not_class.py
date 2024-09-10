import csv

INDEX_S_FILE = "./index_s.csv"
INDEX_H_FILE = "./index_h.csv"

mp1 = None
mp2 = None

K = 34

#####################################
#  便利函数组
#####################################

def min_val(a, b):
    return b if a > b else a

def min3(a, b, c):
    return min_val(a, min_val(b, c))

def add1(lhs, rhs, m):
    for j in range(m + 5, 4, -1):
        sht = min_val(lhs[j] + rhs[0], lhs[0] + rhs[j])
        for k in range(5, j):
            sht = min3(sht, lhs[k] + rhs[j - k], lhs[j - k] + rhs[k])
        lhs[j] = sht

    for j in range(m, -1, -1):
        sht = lhs[j] + rhs[0]
        for k in range(j):
            sht = min_val(sht, lhs[k] + rhs[j - k])
        lhs[j] = sht

    return lhs

def add2(lhs, rhs, m):
    j = m + 5
    sht = min_val(lhs[j] + rhs[0], lhs[0] + rhs[j])
    for k in range(5, j):
        sht = min3(sht, lhs[k] + rhs[j - k], lhs[j - k] + rhs[k])
    lhs[j] = sht
    return lhs

def read_csv(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        mp = [[int(cell) for cell in row if cell] for row in reader]
    return mp

def accum(v, start, end, base):
    ret = base
    for i in range(start, end):
        ret = 5 * ret + v[i]
    return ret

####################################
#  主处理
####################################

# 初始化
def initialize():
    global mp1, mp2
    mp1 = read_csv(INDEX_S_FILE)
    mp2 = read_csv(INDEX_H_FILE)

# 通常的面子手
def calc_lh(t, m):
    ret = mp1[accum(t, 1, 9, t[0])]
    ret = add1(ret, mp1[accum(t, 10, 18, t[9])], m)
    ret = add1(ret, mp1[accum(t, 19, 27, t[18])], m)
    ret = add2(ret, mp2[accum(t, 28, 34, t[27])], m)
    return ret[5 + m]

# 七对子
def calc_sp(t):
    pair = 0
    kind = 0
    for i in range(K):
        if t[i] > 0:
            kind += 1
            if t[i] >= 2:
                pair += 1
    return 7 - pair + (7 - kind if kind < 7 else 0)

# 国士无双
def calc_to(t):
    pair = 0
    kind = 0
    for i in [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]:
        if t[i] > 0:
            kind += 1
            if t[i] >= 2:
                pair += 1
    return 14 - kind - (1 if pair > 0 else 0)

# 向听数计算主函数
def calc(hand, mode, m):
    ret = [1024, 0]

    if mode & 1:
        sht = calc_lh(hand, m)
        if sht < ret[0]:
            ret = [sht, 1]
        elif sht == ret[0]:
            ret[1] |= 1

    if (mode & 2) and m == 4:
        sht = calc_sp(hand)
        if sht < ret[0]:
            ret = [sht, 2]
        elif sht == ret[0]:
            ret[1] |= 2

    if (mode & 4) and m == 4:
        sht = calc_to(hand)
        if sht < ret[0]:
            ret = [sht, 4]
        elif sht == ret[0]:
            ret[1] |= 4

    return ret

def shanten(hand, mode = 7, m = 4):
    # Initializing every time will extremely lower the speed.
    # initialize()
    return calc(hand, mode, m)

# Example 1:
# Speed is about 1250/s.

# import tqdm
# initialize()
# for _ in tqdm.trange(100000):
#     shanten((
#     2,2,2,0,0,0,0,0,0, #Manzu
#     0,0,0,2,2,2,0,0,0, #Pinzu
#     0,0,0,0,0,0,1,0,0, #Souzu
#     0,0,0,0,0,0,0 #Jihai
#     ))

# Example 2:

# initialize()
# print(shanten((
#     0, 0, 0, 0, 2, 1, 0, 0, 1,  # Manzu
#     0, 0, 1, 0, 2, 1, 2, 0, 0,  # Pinzu
#     0, 0, 1, 0, 0, 0, 1, 0, 0,  # Souzu
#     0, 0, 0, 1, 0, 0, 1  # Jihai
# ), 1))

# Example 3:

# initialize()
# print(shanten((
#     0, 1, 0, 0, 0, 0, 0, 0, 0,  # Manzu
#     1, 0, 0, 0, 0, 1, 0, 0, 0,  # Pinzu
#     0, 0, 0, 0, 1, 0, 0, 0, 0,  # Souzu
#     0, 1, 0, 0, 0, 0, 0  # Jihai
# ), m = 1))

# Example 4:

# initialize()
# print(shanten((
#     0, 1, 0, 0, 0, 0, 0, 0, 1,  # Manzu
#     1, 0, 0, 0, 0, 1, 1, 0, 0,  # Pinzu
#     1, 0, 0, 0, 0, 1, 0, 1, 0,  # Souzu
#     0, 0, 2, 1, 0, 0, 0  # Jihai
# ), m = 3))

# Example 5:

# initialize()
# print(shanten((
#     0, 0, 0, 0, 0, 0, 0, 0, 0,  # Manzu
#     0, 0, 0, 0, 0, 0, 0, 0, 0,  # Pinzu
#     0, 0, 0, 0, 0, 0, 0, 0, 0,  # Souzu
#     0, 0, 1, 1, 0, 0, 0  # Jihai
# ), m = 0))