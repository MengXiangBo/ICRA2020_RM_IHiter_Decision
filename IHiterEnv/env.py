import numpy as np
import time
from collections import OrderedDict
from IHiterEnv.display import Viewer
from IHiterEnv.parameter import *
from IHiterEnv.map_element import *
from IHiterEnv.agent import *
from IHiterEnv.policy import RandomPolicy

class ObstacleSet(OrderedDict):

    def __init__(self, RobotDict):
        """
            the type of map, containing the basic element in the map
        """
        self.RobotDict = RobotDict
        self['border'] = MapBorder()
        # the initializationof obstacle
        self['B1'] = Obstacle(np.array(
                [[0, 3480], [1000, 3480], [1000, 3280], [0, 3280]]))
        self['B2'] = Obstacle(np.array(
            [[1500, 2340], [2300, 2340], [2300, 2140], [1500, 2140]]))
        self['B3'] = Obstacle(np.array(
            [[1500, 1000], [1750, 1000], [1750, 0], [1500, 0]]))
        self['B4'] = Obstacle(np.array(
            [[3540, 3545], [4540, 3545], [4540, 3345], [3540, 3345]]))
        self['B5'] = Obstacle(np.array(
            [[4040, 2416], [4216, 2240], [4040, 2064], [3864, 2240]]))
        self['B6'] = Obstacle(np.array(
            [[3540, 1135], [4540, 1135], [4540, 935], [3540, 935]]))
        self['B7'] = Obstacle(np.array(
            [[6380, 4480], [6580, 4480], [6580, 3480], [6380, 3480]]))
        self['B8'] = Obstacle(np.array(
            [[5780, 2340], [6580, 2340], [6580, 2140], [5780, 2140]]))
        self['B9'] = Obstacle(np.array(
            [[7080, 1200], [8080, 1200], [8080, 1000], [7080, 1000]]))
        # the initialization of buff domain
        self['F1'] = Domain(np.array([500, 2790, 540, 480, 40]))
        self['F2'] = Domain(np.array([1900, 1650, 540, 480, 40]))
        self['F3'] = Domain(np.array([4040, 4035, 540, 480, 40]))
        self['F4'] = Domain(np.array([7580, 1690, 540, 480, 40]))
        self['F5'] = Domain(np.array([6180, 2830, 540, 480, 40]))
        self['F6'] = Domain(np.array([4040, 445, 540, 480, 40]))
        # the initialization of start place
        self['C1'] = StartPlace(np.array([500, 3980, 1000, 1000, 40]))
        self['C2'] = StartPlace(np.array([500, 500, 1000, 1000, 40]))
        self['C3'] = StartPlace(np.array([7580, 500, 1000, 1000, 40]))
        self['C4'] = StartPlace(np.array([7580, 3980, 1000, 1000, 40]))

    def CanShoot(self, attacker, defender):
        """
            Figure out if defender is in the shoot range
            of the attacker

            param:

            attacker -> Robot: the refernence of the attacking robot

            defender -> Robot: the reference of the aimed robot

            retrun:
            
            the bool value of the answer
        """
        for item in self.values():
            if isinstance(item, Obstacle) and \
            item.isLineIntersect(attacker.center, defender.center):
                return False
            if isinstance(item, Robot) and \
                item is not attacker and \
                    item is not defender:
                if item.isLineIntersect(attacker.center, 
                                        defender.center):
                    return False
        return True


class ICRA_Env:
    '''
        This type was made as the core class of env
    '''

    def __init__(self):
        self.TeamDict = OrderedDict([
            ('Blue', Team('Blue', 2)), ('Red', Team('Red', 2))])
        self.RobotDict = OrderedDict()
        for team in self.TeamDict.values():
            for robot in team.RobotDict.values():
                self.RobotDict[robot.name] = robot
                if robot.team_name == 'Blue':
                    robot.enemy_team = self.TeamDict['Red']
                if robot.team_name == 'Red':
                    robot.enemy_team = self.TeamDict['Blue']
        self.map = ObstacleSet(self.RobotDict)
        self.buff_set = BuffDoaminSet(self.map, self.TeamDict)
        self.state = JointState(self.RobotDict, 
                            self.buff_set.BuffDistribute)
        self.robot_logger = RobotLogger()
        self.team_action = TeamAction()
        self.Red = RandomPolicy()

    def MoveAction(self, robot, action):
        '''
            the move action for robot

            param:

            robot -> Robot: the reference of the moving robot

            action -> Action: the robot action

            retrun: the moving result
        '''
        if not robot.isAlive():
            return None
        if robot.name in self.buff_set.NoMoveBuffDict:
            robot.MoveGun(action)
            return MR.ForbiddenBuff
        robot.Move(action)
        for item in self.map.values():
            if isinstance(item, Domain) and item * robot:
                self.buff_set.CheckNewBuff(robot, item)
                if item.buff is BT.NoMove or \
                        item.buff is BT.NoShoot:
                    return MR.ForbiddenBuff
                elif item.buff is BT.BlueAmmo or \
                        item.buff is BT.BlueAmmo:
                    if robot.team_name == 'Red':
                        return MR.WrongBuff
                    else:
                        return MR.RightBuff
                elif item.buff is BT.RedAmmo or \
                    item.buff is BT.RedAmmo:
                    if robot.team_name == 'Blue':
                        return MR.WrongBuff
                    else:
                        return MR.RightBuff
            if item.isObstacle and item * robot:
                robot.StepBack()
                return MR.Crash
        for another_robot in self.RobotDict.values():
            if another_robot is robot:
                continue
            elif another_robot * robot:
                robot.StepBack()
                return MR.Crash
        return MR.MoveWell

    def AttackAction(self, attacker, action):
        '''
            the shoot action made by the attacker

            blood lost are operated automaticly

            param:

            attacker -> robot: the reference of the attacking robot

            action -> the action of the robot
            
            return: the attack result
        '''
        attacker.attack_point = None
        if not attacker.isAlive():
            return None
        if action.shoot == AC.NoShoot:
            attacker.HeatRecover()
            return AR.NoAttack
        if attacker.name in self.buff_set:
            attacker.HeatRecover()
            return AR.BuffNoShoot
        NoShootReason = attacker.Shoot()
        if NoShootReason is not True:
            return NoShootReason
        results = list()
        for defender in attacker.enemy_team.RobotDict.values():
            if not defender.isAlive():
                results.append(AR.WrongAttack)
                continue
            elif attacker.isInShootRange(defender) \
                and self.map.CanShoot(attacker, defender):
                defender.BloodLost()
                results.append(AR.Hit)
                attacker.attack_point = defender.center
                break
            else:
                results.append(AR.Missing)
        if AR.Hit in results:
            return AR.Hit
        elif AR.WrongAttack in results:
            return AR.WrongAttack
        else:
            return AR.Missing

    def step(self, action):
        '''
            input the raw action, and output the next state

            param:

            action -> ndarray of (256): the raw array of the robot
        '''
        action_dict = self.team_action.ActionGen(action, 
                self.Red.React())
        for robot in self.RobotDict.values():
            self.robot_logger[robot.name] = [
                self.MoveAction(robot, action_dict[robot.name]), \
                self.AttackAction(robot, action_dict[robot.name])]
        self.buff_set.StepBuffRefresh(EP.StepTime)
        self.StepRefreshAndLog()
        return self.state.OutputState(TP.TrainedTeam), \
            self.robot_logger.reward['Blue'], self.isDone, dict()

    def StepRefreshAndLog(self):
        
        for team in self.TeamDict.values():
            if team.GetLivingNum() == 0:
                self.isDone = True
                self.Winner = 'Blue' if team.name is 'Red' else 'Red'
                break
        if self.buff_set.timer <= 0:
            self.buff_set.BuffRefresh()
        self.EpisodeStep += 1
        self.robot_logger.CaculateReward(self.Winner)

    def reset(self):
        '''
            the reset function
        '''
        self.isDone = False
        self.Winner = None
        self.EpisodeStep = 0
        self.RobotDict['Blue0'].SetPosition(
                            self.map['C1'].center)
        self.RobotDict['Blue1'].SetPosition(
                            self.map['C2'].center)
        self.RobotDict['Red0'].SetPosition(
                            self.map['C3'].center)
        self.RobotDict['Red1'].SetPosition(
                            self.map['C4'].center)
        for team in self.TeamDict.values():
            team.RestartEpisode()
        self.buff_set.BuffRefresh()
        return self.state.OutputState('Blue')

    def render(self, mode='human'):
        '''
            the render function
        '''
        if not hasattr(self, 'viewer'):
            self.viewer = Viewer(self.map)
        else:
            self.viewer.Render()

    def close(self):
        pass


class RobotLogger(dict):
    """
        The basic type to log the robot's observable and 
        unobservable state 
    """

    def __init__(self):
        super().__init__()
        self.reward = Reward()

    def CaculateReward(self, winner=None):
        self.reward.ClearReward()
        for robot_name in EP.RobotnameList:
            if robot_name == 'Red0' or robot_name == 'Red1':
                team_name = 'Red'
            else:
                team_name = 'Blue'
            if self[robot_name][0] == MR.MoveWell:
                self.reward[team_name] -= .1
            elif self[robot_name][0] == MR.RightBuff:
                self.reward[team_name] += 5
            elif self[robot_name][0] is None:
                self.reward[team_name] -= .1
            else:
                self.reward[team_name] -= 10
            if self[robot_name][1] == AR.NoAttack or \
                 self[robot_name][1] == AR.BuffNoShoot:
                self.reward[team_name] -= .1
            elif self[robot_name][1] == AR.Hit:
                self.reward[team_name] += 10
            elif self[robot_name][0] is None:
                self.reward[team_name] -= .1
            else:
                self.reward[team_name] -= 10
        if winner is not None:
            self.reward[winner] += 100
        return self.reward


class Reward(dict):
    """
        The basic type of the reward of different team
    """
    def __init__(self):
        super().__init__()
        self.ClearReward()
        
    def ClearReward(self):
        for name in EP.TeamnameList:
            self[name] = 0