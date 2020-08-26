import numpy as np
from math import atan2, pi
from collections import OrderedDict
from IHiterEnv.map_element import Obstacle
from IHiterEnv.parameter import *

class Robot(Obstacle):
    '''' Robot type

    The Robot type contain the basic robot attributes and action functions

    param: 
    
    team_name -> str: the name of the team

    number -> str: the robot number in the team
    '''
    def __init__(self, team_name = None, number = None):
        self.team_name = team_name
        self.name = team_name + str(number)
        self.number = number
        self.enemy_team = None

        self.gun = 0
        self.blood, self.ammo, self.heat = RP.MaxBlood, 0, 0
        super().__init__(np.zeros(2) + RP.Border)
        self.attack_point = None

    def SetPosition(self, center):
        self.center = np.copy(center)
        self._updata_points()

    def _updata_points(self):
        self.points = self.center + self.border
        if self.gun <= -180:
            self.gun += 360
        else:
            self.gun -= 360

    def Move(self, action):
        self.last_center = np.copy(self.center)
        self.center += action.move
        self.gun = action.gun
        self._updata_points()

    def MoveGun(self, action):
        self.gun = action.gun
        self._updata_points()

    def StepBack(self):
        self.center = np.copy(self.last_center)
        self._updata_points()

    def isInShootRange(self, defender):
        def LineAngle(point1, point2):
            line = point2 - point1
            return atan2(line[1], line[0]) / pi * 180
        relative_angle = LineAngle(self.center, defender.center)
        if self.gun < 135 and self.gun > -135:
            if relative_angle > self.gun + 45 or relative_angle < self.gun - 45:
                return False
        elif self.gun > 135:
            if relative_angle > self.gun - 315 and relative_angle < self.gun - 45:
                return False
        elif self.gun < -135:   
            if relative_angle < self.gun + 315 and relative_angle > self.gun + 45:
                return False
        return True

    def isAlive(self):
        if self.blood > 0:
            return True
        else:
            return False

    def HeatRecover(self):
        self.heat = max(self.heat-12, 0)

    def Shoot(self):
        if self.heat >= RP.MaxHeat:
            self.heat = max(self.heat-12, 0)
            self.HeatRecover()
            return AR.OverHeatNoShoot
        elif self.ammo <= 0:
            self.heat = max(self.heat-12, 0)
            self.HeatRecover()
            return AR.NoAmmoNoShoot
        else:
            self.ammo -= 1
            self.heat = min(self.heat + 20, RP.MaxHeat)
            return True

    def BloodLost(self):
        self.blood -= EP.BloodLost


class Team():
    """
        The type of the 
    """
    def __init__(self, name, MemberNumber):
        self.name = name
        self.RobotDict = OrderedDict()
        for index in range(MemberNumber):
            Robotname = name + str(index)
            self.RobotDict[Robotname] = Robot(self.name, index)
        self.RestartEpisode()

    def RestartEpisode(self):
        for robot in self.RobotDict.values():
            robot.blood = RP.MaxBlood
            robot.ammo = RP.StartAmmo
            robot.heat = 0

    def FullTeamAddAmmo(self):
        for robot in self.RobotDict.values():
            if robot.isAlive():
                robot.ammo = min(robot.ammo + 100, RP.MaxAmmo)
        return True

    def FullTeamRecover(self):
        for robot in self.RobotDict.values():
            if robot.isAlive():
                robot.blood = min(robot.blood + 200, RP.MaxBlood)
        return True

    def GetLivingNum(self):
        result = 0
        for robot in self.RobotDict.values():
            if robot.isAlive():
                result += 1
        return result


class State():
    def __init__(self, agent):
        self.agent = agent
        self.state = np.zeros(6)

    def OutputState(self):
        self.state[0] = self.agent.center[0] / EP.XMax
        self.state[1] = self.agent.center[1] / EP.YMax
        self.state[2] = self.agent.gun / 180
        self.state[3] = self.agent.blood / EP.MaxBlood
        self.state[4] = self.agent.ammo / EP.MaxAmmo
        self.state[5] = self.agent.heat / EP.MaxHeat
        return self.state


class JointState:
    def __init__(self, RobotDict, BuffDistribute):
        self.RobotStateDict = OrderedDict()
        self.RobotDict = RobotDict
        for robot in RobotDict.values():
            self.RobotStateDict[robot.name] = State(robot)
        self.BuffDistribute = BuffDistribute
        self.joint_state = np.zeros(TP.StateDim)

    def OutputState(self, Trainteam_name):
        if Trainteam_name == 'Blue':
            self.joint_state = np.concatenate((self.RobotStateDict['Blue0'].OutputState(), \
                            self.RobotStateDict['Blue1'].OutputState(), \
                            self.RobotStateDict['Red0'].OutputState(), \
                            self.RobotStateDict['Red1'].OutputState(), \
                            self.BuffDistribute))
        if Trainteam_name == 'Red':
            self.joint_state = np.concatenate((self.RobotStateDict['Red0'].OutputState(), \
                            self.RobotStateDict['Red1'].OutputState(), \
                            self.RobotStateDict['Blue0'].OutputState(), \
                            self.RobotStateDict['Blue1'].OutputState(), \
                            self.BuffDistribute))
        return self.joint_state


class Action():

    def __init__(self, MoveCode, AttackCode):
        if MoveCode is MC.Up:
            self.move = np.array([0, RP.StepLength])
            self.gun = 90
        elif MoveCode is MC.Down:
            self.move = np.array([0, -RP.StepLength])
            self.gun = -90
        elif MoveCode is MC.Left:
            self.move = np.array([-RP.StepLength, 0])
            self.gun = 180
        elif MoveCode is MC.Right:
            self.move = np.array([RP.StepLength, 0])
            self.gun = 0
        elif MoveCode is MC.UpLeft:
            self.move = np.array([-RP.StepLength, RP.StepLength])
            self.gun = 135
        elif MoveCode is MC.UpRight:
            self.move = np.array([RP.StepLength, RP.StepLength])
            self.gun = 45
        elif MoveCode is MC.DownLeft:
            self.move = np.array([-RP.StepLength, -RP.StepLength])
            self.gun = -135
        elif MoveCode is MC.DownRight:
            self.move = np.array([RP.StepLength, -RP.StepLength])
            self.gun = -45
        self.shoot = AttackCode


class TeamAction():

    def __init__(self):
        self.ActionList = list()
        for a1 in range(2):
            for a2 in range(2):
                for m1 in range(8):
                    for m2 in range(8):
                        self.ActionList.append([Action(m1, a1), Action(m2, a2)])
        self.action_dict = OrderedDict()

    def ActionGen(self, output1, output2):

        self.action_dict['Blue0'] = self.ActionList[np.argmax(output1)][0]
        self.action_dict['Blue1'] = self.ActionList[np.argmax(output1)][1]
        self.action_dict['Red0'] = self.ActionList[np.argmax(output2)][0]
        self.action_dict['Red1'] = self.ActionList[np.argmax(output2)][1]
        return self.action_dict