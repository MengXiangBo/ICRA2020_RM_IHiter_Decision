import pyglet
from pyglet.window import Window
import numpy as np
from IHiterEnv.parameter import *
from IHiterEnv.map_element import *

Black = (0, 0, 0)
White = (255, 255, 255)
Red = (255, 0, 0)
Green = (0, 255, 0)
Blue = (0, 0, 255)
Purple = (125, 38, 205)
Grey = (105, 105, 105)


class RobotDisplay():

    def __init__(self, robot):
        '''
            @ robot：机器人的引用
        '''
        self.RobotBatch = pyglet.graphics.Batch()
        self.robot = robot
        if robot.name == 'Blue0':
            self.color = Blue
            self.y = 520 + 60 * 3
            label_color = (0, 0, 255, 255)
        elif robot.name == 'Blue1':
            self.color = Blue
            self.y = 520 + 60 * 2
            label_color = (0, 0, 255, 255)
        elif robot.name == 'Red0':
            self.color = Red
            self.y = 520 + 60 * 1
            label_color = (255, 0, 0, 255)
        elif robot.name == 'Red1':
            self.color = Red
            self.y = 520
            label_color = (255, 0, 0, 255)
        pyglet.text.Label(text=robot.name, bold=True, font_size=20, x=30,
                            y=self.y + 20, color=label_color, batch=self.RobotBatch)
        self.RobotVertex = self.RobotBatch.add(4, pyglet.gl.GL_QUADS, None,
                        ('v2f', np.zeros(8)),
                        ('c3B', self.color * 4,))
        # 射击的射线显示
        self.shootlinebatch = pyglet.graphics.Batch()
        self.shootlinevertex = self.shootlinebatch.add(2, pyglet.gl.GL_LINES, None,
                        ('v2f', np.zeros(4)),
                        ('c3B', self.color * 2,))
        self.BloodBatch, self.BloodVertex = self.DisplayBox('Blood')
        self.AmmoBatch, self.AmmoVertex = self.DisplayBox('Ammo')
        self.HeatBatch, self.HeatVertex = self.DisplayBox('Heat')


    def DisplayBox(self, DisplayType):
        Batch = pyglet.graphics.Batch()
        r = 2
        y = self.y + 35
        if DisplayType == 'Blood':
            x = 100 + 30
            self.blood_text = pyglet.text.Label(text=DisplayType + ' : ' + str(self.robot.blood),
                            bold=True, font_size=15, x=x, y=y, color=(0, 0, 0, 255), batch=Batch)
            percent = self.robot.blood / RP.MaxBlood
        elif DisplayType == 'Ammo':
            x = 100 + 260 * 1 + 30
            self.ammo_text = pyglet.text.Label(text=DisplayType + ' : ' + str(self.robot.ammo),
                            bold=True, font_size=15, x=x, y=y, color=(0, 0, 0, 255), batch=Batch)
            percent = self.robot.ammo / RP.MaxAmmo
        elif DisplayType == 'Heat':
            x = 100 + 260 * 2 + 30
            self.heat_text = pyglet.text.Label(text=DisplayType + ' : ' + str(self.robot.heat),
                            bold=True, font_size=15, x=x, y=y, color=(0, 0, 0, 255), batch=Batch)
            percent = self.robot.heat / RP.MaxHeat

        y = self.y + 10
        # 最外面的灰框
        Batch.add(4, pyglet.gl.GL_QUADS, None,
            ('v2f', (x, y, x + 200, y, x + 200, y + 15, x, y + 15)),
            ('c3B', Grey * 4,))
        # 中间填充的白条
        Batch.add(4, pyglet.gl.GL_QUADS, None,
            ('v2f', (x + r, y + r, x + 200 -r, y + r, x + 200 - r, y + 15 - r, x + r, y + 15 - r)),
            ('c3B',  White * 4,))
        # 中间的数字计量的颜色条
        BarVertex = Batch.add(4, pyglet.gl.GL_QUADS, None,
            ('v2f', (x + r, y + r, x - r + 200 * percent, y + r, x - r + 200 * percent, y + 15 - r, x + r, y + 15 - r)),
            ('c3B', Red * 4,))

        return Batch, BarVertex


    def draw(self):
        self.RobotVertex.vertices = ((self.robot.points+EP.Border)/10).reshape(8)
        r, y, x = 2, self.y + 10, 100 + 30
        percent = self.robot.blood / RP.MaxBlood
        self.BloodVertex.vertices = (x + r, y + r, x - r + 200 * percent, y + r, x - r + 200 * percent, y + 15 -r, x + r, y + 15 - r)
        self.blood_text.text = 'Blood : ' + str(self.robot.blood)

        x = 100 + 260 * 1 + 30
        percent = self.robot.ammo / RP.MaxAmmo
        self.AmmoVertex.vertices = (x + r, y + r, x - r + 200 * percent, y + r, x - r + 200 * percent, y + 15 -r, x + r, y + 15 - r)
        self.ammo_text.text = 'Ammo : ' + str(self.robot.ammo)

        x = 100 + 260 * 2 + 30
        percent = self.robot.heat / RP.MaxHeat
        self.HeatVertex.vertices = (x + r, y + r, x - r + 200 * percent, y + r, x - r + 200 * percent, y + 15 -r, x + r, y + 15 - r)
        self.heat_text.text = 'Heat : ' + str(self.robot.heat)

        attack_point = self.robot.attack_point
        if attack_point is None:
            self.shootlinevertex.vertices = np.zeros(4)
        else:
            attack_point = (attack_point + EP.Border) / 10
            self.shootlinevertex.vertices = np.concatenate(((self.robot.center+EP.Border)/10, attack_point))

        self.RobotBatch.draw()
        self.BloodBatch.draw()
        self.AmmoBatch.draw()
        self.HeatBatch.draw()


class DomainDisplay():
    def __init__(self, domain):
        self.domain = domain
        self.Batch = pyglet.graphics.Batch()

        self.Batch.add(4, pyglet.gl.GL_QUADS, None,
                        ('v2f', ((self.domain.out_points+EP.Border)/10).reshape(8)),
                        ('c3B', Grey * 4,))
        self.Batch.add(4, pyglet.gl.GL_QUADS, None,
                        ('v2f', ((self.domain.interal_points+EP.Border)/10).reshape(8)),
                        ('c3B', White * 4,))
        self.LabelBatch = pyglet.graphics.Batch()
        self.label = pyglet.text.Label(text='B', bold=True, font_size=30, 
                        x=(self.domain.center[0]+EP.Border)/10 - 15, 
                        y=(self.domain.center[1]+EP.Border)/10 - 15, 
                        color=(0, 0, 0, 0), batch=self.LabelBatch)

    def draw(self):
        if self.domain.buff == BT.NoMove:
            self.label.color, self.label.text = (0, 0, 0, 255), 'M'
        if self.domain.buff == BT.NoShoot:
            self.label.color, self.label.text = (0, 0, 0, 255), 'S'
        if self.domain.buff == BT.RedRecover:
            self.label.color, self.label.text = (255, 0, 0, 255), 'R'
        if self.domain.buff == BT.RedAmmo:
            self.label.color, self.label.text = (255, 0, 0, 255), 'A'
        if self.domain.buff == BT.BlueRecover:
            self.label.color, self.label.text = (0, 0, 255, 255),  'R'
        if self.domain.buff == BT.BlueAmmo:
            self.label.color, self.label.text = (0, 0, 255, 255), 'A'
        self.Batch.draw()
        self.LabelBatch.draw()


class Viewer(Window):

    def __init__(self, env_map):
        '''
            在初始化阶段需要输入的数据有：

            @ RobotDict：所有机器人引用的字典
            
            @ Obstacles：所有障碍物的numpy，包含每个障碍物四个点的位置

            @ StartPlaces：启动区的numpy，用block数据型表示

            @ Domains：buff区的引用
        '''
        super().__init__(width=int((EP.XMax+2*EP.Border)/10), height=760, resizable=False, 
                        caption='ICRA', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        # pyglet坐标系 左下角为（0,0）右上角为（width,height）
        self.map = env_map
        # 初始化边框
        self.ObstacleBatch = pyglet.graphics.Batch()
        self.DomainsDrawDict = dict()
        for name in self.map:
            # 初始化障碍物的block
            if isinstance(self.map[name], Obstacle):
                self.ObstacleBatch.add(4, pyglet.gl.GL_QUADS, None,
                                ('v2f', ((self.map[name].points+EP.Border)/10).reshape(8)),
                                ('c3B', Black * 4,))
            # 初始化启动区
            if isinstance(self.map[name], StartPlace) or isinstance(self.map[name], MapBorder):
                if name is 'C1' or name is 'C2':
                    color = Blue
                elif name is 'border':
                    color = Black
                else:
                    color = Red
                self.ObstacleBatch.add(4, pyglet.gl.GL_QUADS, None,
                                    ('v2f', ((self.map[name].out_points+EP.Border)/10).reshape(8)),
                                    ('c3B', color * 4,))
                self.ObstacleBatch.add(4, pyglet.gl.GL_QUADS, None,
                                    ('v2f', ((self.map[name].interal_points + EP.Border)/10).reshape(8)),
                                    ('c3B', White * 4,))
            # 初始化加成区的
            if isinstance(self.map[name], Domain): 
                self.DomainsDrawDict[name] = DomainDisplay(self.map[name])
        # 初始化机器人的block
        self.RobotsDrawDict = dict()
        for robot in self.map.RobotDict.values():
            self.RobotsDrawDict[robot.name] = RobotDisplay(robot)

    def Render(self):
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        # self.on_draw()  # link with the on_draw method
        self.flip()


    def on_draw(self):
        self.clear()
        self.ObstacleBatch.draw()
        for domaindisplay in self.DomainsDrawDict.values():
            domaindisplay.draw()
        for robotdisplay in self.RobotsDrawDict.values():
            robotdisplay.draw()
        for robotdisplay in self.RobotsDrawDict.values():
            robotdisplay.shootlinebatch.draw()
