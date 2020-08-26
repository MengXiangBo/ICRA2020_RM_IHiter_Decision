import numpy as np

class TP:
    TrainedTeam = 'Blue'
    ActionDim = (8 * 2) * (8 * 2)
    StateDim = (4 * 6 + 6)

class EP:
    """
        Environment Parameter
    """
    RobotnameList = ['Blue0', 'Blue1', 'Red0', 'Red1']
    TeamnameList = {'Blue', 'Red'}
    XMax = 8080
    YMax = 4480
    Border = 360
    TrainTeam = 'Blue'
    MaxBlood = 2000
    StartAmmo = 50
    MaxAmmo = 500
    MaxHeat = 240
    StepTime = .1
    BloodLost = 100

class RP:
    side = 450
    StepLength = 100
    DegreeMAxStep = 45
    MaxBlood = 2000
    StartAmmo = 50
    MaxAmmo = 500
    MaxHeat = 240
    Border = np.array([[side,side],[-side,side],[-side,-side],[side,-side]])/2


class BT:
    # Buff的枚举类型
    BlueRecover = 0
    RedRecover = 1
    BlueAmmo = 2
    RedAmmo = 3
    NoShoot = 4
    NoMove = 5


class MC:
    """
        Move Code
    """
    Up = 0
    Down = 1
    Left = 2
    Right = 3
    UpLeft = 4
    UpRight = 5
    DownLeft = 6
    DownRight = 7


class AC:
    """
        Attack Code
        机器人攻击代号
    """
    NoShoot = 0
    Shoot = 1


class MR:
    """
        Move Result
    """
    ForbiddenBuff = 'ForbiddenBuff'
    WrongBuff = 'WrongBuff'
    RightBuff = 'RightBuff'
    MoveWell = 'MoveWell'
    Crash = 'Crash'


class AR:
    """
        Attack Result
        机器人的攻击情况代号
    """
    NoAttack = 'NoAttack' # 机器人自主决策不射击
    OverHeatNoShoot = 'OverHeatNoShoot' # 温度过高无法射击
    BuffNoShoot = 'BuffNoShoot' # 有buff无法射击
    NoAmmoNoShoot = 'NoAmmoNoShoot' # 弹药不足无法射击
    WrongAttack = 'WrongAttack' # 目标已经死亡无法攻击
    Missing = 'Missing' # 无法射击到
    Hit = 'Hit' # 攻击成功
