import numpy as np
from IHiterEnv.parameter import *
from collections import OrderedDict

class Rectangle():
    """ 
        The basic type of the obstacle, domain, border and robot model
    """
    def __init__(self, points):
        self.points = points
        self.center = np.sum(self.points, 0) / 4
        [xmax, ymax], [xmin, ymin] = \
            np.max(self.points, 0), np.min(self.points, 0)
        self.border = np.array([[xmax, ymax], 
            [xmax, ymin], [xmin, ymin], [xmin, ymax]])
        self.size = np.array([xmax-xmin, ymax-ymin])

    def __mul__(self, other):
        """
            rewrite the __mul__ and return if these two 
            rectangles are intersected
        """
        if (np.abs(self.center-other.center) >
                     (self.size + other.size)/2).any():
            return False
        else:
            return True


class Obstacle(Rectangle):
    """
        The basic type of the obstacle including 
        robot and static obstacle
    """
    def __init__(self, points):
        self.isObstacle = True
        super().__init__(points)

    def isLineIntersect(self, point1, point2):
        def cross(array1, array2):
            return array1[0] * array2[1] - array2[0] * array1[1]
        def isLineIntersectLine(point11, point12, point21, point22):
            result1 = cross(point11-point21,point12-point21) \
                * cross(point11-point22,point12-point22)
            result2 = cross(point21-point11,point22-point11) \
                * cross(point21-point12,point22-point12)
            if result1 < 0 and result2 < 0:
                return True
            else:
                return False
        if isLineIntersectLine(point1, point2, 
                    self.points[0], self.points[1]) or \
            isLineIntersectLine(point1, point2, 
                    self.points[1], self.points[2]) or \
            isLineIntersectLine(point1, point2, 
                    self.points[2], self.points[3]) or \
            isLineIntersectLine(point1, point2, 
                    self.points[3], self.points[0]):
            return True
        

class Box(Rectangle):
    """"
        the basic type of the box including start place, 
        buff domain and map border
    """
    def __init__(self, BoxPoints):
        self.isObstacle = False
        [x, y, w, h, t] = BoxPoints
        self.out_points = np.array([[x+w/2, y+h/2], [x-w/2, y+h/2], 
                                [x-w/2, y-h/2], [x+w/2, y-h/2]])
        self.interal_points = np.array([[x+w/2-t, y+h/2-t], [x-w/2+t, y+h/2-t],
                                [x-w/2+t, y-h/2+t], [x+w/2-t, y-h/2+t]])
        super().__init__(self.out_points)


class StartPlace(Box):
    """
        the type for startplace in the four corners of map
    """
    def __init__(self, BoxPoints):
        super().__init__(BoxPoints)


class MapBorder(Box):
    """
        the type for the map border
    """
    def __init__(self):
        super().__init__([EP.XMax/2, EP.YMax/2, 
                EP.XMax+2*EP.Border, EP.YMax+2*EP.Border, EP.Border])
        self.isObstacle = True
        self.border = self.interal_points
        self.size = np.array([EP.XMax, EP.YMax])


    def __mul__(self, rectangle):
        """
            rewrite the function to realize the insect
        """
        if (np.abs(self.center-rectangle.center) < 
                np.abs(self.size-rectangle.size)/2).all():
            return False
        else:
            return True


class Domain(Box):
    
    def __init__(self, BoxPoints):
        super().__init__(BoxPoints)

    def ChangeAttribute(self, BuffType):
        self.buff = BuffType


class Buff():

    def __init__(self, BuffType, robot=None):
        '''
            根据输入的信息创建buff
            加成类buff需要队伍和机器人的引用
            惩罚类buff只需要机器人的引用
        '''
        self.BuffType = BuffType
        self.Robotname = robot.name        
        # 禁止类buff生成
        if BuffType == BT.NoShoot or BuffType == BT.NoMove:
            self.timer = 10
        # 加成类buff的team是受益方的队伍
        elif BuffType == BT.BlueRecover or BuffType == BT.BlueAmmo:
            self.team_name = 'Blue'
        elif BuffType == BT.RedRecover or BuffType == BT.RedAmmo:
            self.team_name = 'Red'

    def TimeFly(self, time):
        '''
            每一步结束后减去时间，如果buff时间结束，则输出True
        '''
        self.timer -= time
        if self.timer == 0:
            return True


class BuffDoaminSet(OrderedDict):
    
    def __init__(self, env_map, TeamDict):
        super().__init__()
        self.TeamDict = TeamDict
        #初始化buff区域的分配
        self.BuffDistribute = np.zeros(6)
        self['F1'] = env_map['F1']
        self['F2'] = env_map['F2']
        self['F3'] = env_map['F3']
        self['F4'] = env_map['F4']
        self['F5'] = env_map['F5']
        self['F6'] = env_map['F6']
        # 记录在一个刷新周期内产生的Buff
        self.NoMoveBuffDict = dict()
        self.NoShootBuffDict = dict()
        self.AttributeBuffDict = dict()
        # 计数刷新
        self.BuffRefresh()

    def BuffRefresh(self):
        '''
            每一个30s都进行一次刷新，需要进行以下内容：

            1. 去除所有的buff

            2. 刷新加成惩罚区的分布
        '''
        # 生成Buff的分布
        self.BuffBasket = [[BT.BlueRecover, BT.RedRecover], 
                            [BT.BlueAmmo, BT.RedAmmo], 
                            [BT.NoShoot, BT.NoMove]] 
        for index in range(3):
            BuffPair = self.BuffBasket.pop(np.random.randint(0, len(self.BuffBasket)))
            self.BuffDistribute[index] = BuffPair.pop(np.random.randint(0, 2))
            self.BuffDistribute[index+3] = BuffPair[0]
        # 将buff的分布载入到加成惩罚区域
        index = 0
        for item in self.values():
            item.ChangeAttribute(self.BuffDistribute[index])
            index += 1
        # 更新时间
        self.timer = 30
        # 将所有buff字典去掉去掉
        self.AttributeBuffDict.clear()
        self.NoMoveBuffDict.clear()
        self.NoShootBuffDict.clear()
        return self.BuffDistribute

    def StepBuffRefresh(self, Time):
        '''
            1. 对禁止类的buff进行每步时间上的刷新，如果buff时间消失则去掉buff。
            
            2. 同时刷新buff分配的计时。

            @ time：刷新时间
        '''
        self.timer -= Time
        for robot in self.NoMoveBuffDict.keys():
            if self.NoMoveBuffDict[robot].TimeFly(Time):
                self.NoMoveBuffDict.pop(robot)
        for robot in self.NoShootBuffDict.keys():
            if self.NoShootBuffDict[robot].TimeFly(Time):
                self.NoShootBuffDict.pop(robot)

    def CheckNewBuff(self, robot, domain):
        '''
            输入机器人的坐标，计算出机器人会不会触发buff

            如果激发的话讲保存在相应的字典中，并执行buff

            机器人的名字保存在key中，Buff的引用保存在value里面

            @ Robot：机器人的引用
        '''
        if domain.buff == BT.NoShoot and robot.name not in self.NoShootBuffDict:
            self.NoShootBuffDict[robot.name] = Buff(domain.buff, robot)
            return 
        # NoMove
        if domain.buff == BT.NoMove and robot.name not in self.NoMoveBuffDict:
            self.NoMoveBuffDict[robot.name] = Buff(domain.buff, robot)
            return
        # 加成类buff，并检查是不是重复的
        if self.isNewAttributeBuff(domain.buff):
            self.AttributeBuffDict[domain.buff] = Buff(domain.buff, robot)
            if domain.buff == BT.BlueAmmo:
                self.TeamDict['Blue'].FullTeamAddAmmo()
            if domain.buff == BT.RedAmmo:
                self.TeamDict['Red'].FullTeamAddAmmo()
            if domain.buff == BT.BlueRecover:
                self.TeamDict['Blue'].FullTeamRecover()
            if domain.buff == BT.RedRecover:
                self.TeamDict['Red'].FullTeamRecover()
            return

    def isNewAttributeBuff(self, BuffType):
        '''
            输入Attribute Buff的类型，查看有没有相同的buff
        '''
        for buff in self.AttributeBuffDict.values():
            if buff.BuffType == BuffType:
                return False
        return True

    def GetNShootTime(self, robot):
        # 输入机器人的引用，输出禁止射击buff剩余的时间
        if robot.name in self.NoShootBuffDict:
            return self.NoShootBuffDict[robot.name].timer
        return None

    def GetNMoveTime(self, robot):
        # 输入机器人的引用，输出禁止移动buff剩余的时间
        if robot.name in self.NoMoveBuffDict:
            return self.NoMoveBuffDict[robot.name].timer
        return None
