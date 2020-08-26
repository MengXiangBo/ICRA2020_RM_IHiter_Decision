from IHiterEnv.env import ICRA_Env
from DuelingDQN import DuelingDQN
import time
import tensorflow as tf
from IHiterEnv.policy import *
from IHiterEnv.agent import *

MaxEpisode = 2000
MaxEpisodeSteps = 500

train_file = './train_data/'

RLBrain = DuelingDQN(train_dir=train_file)
env = ICRA_Env()
Random = RandomPolicy()
team_action = TeamAction()

def run_n_episode(n):
    print(RLBrain.checkpoints_dir)
    checkpoint_file = RLBrain.checkpoints_dir + 'params1200'
    print('===================== checkpoint_file : ',
        checkpoint_file)
    RLBrain.saver.restore(RLBrain.sess, 
        checkpoint_file)
    print('============================================')
    blue_win_times, red_win_times = 0, 0
    for episode in range(1, n + 1):
        Episode_steps, total_reward = 0, 0
        obs= env.reset()
        while True:
            env.render()
            Blue_Action = RLBrain.EvalDicision(obs)
            next_state, StepReward, isGameOver, _ = env.step(Blue_Action)
            obs = next_state
            Episode_steps += 1
            total_reward += StepReward
            if isGameOver:
                break
        print(' ')
        print('Episode : ', episode,'winner : ', env.Winner)
        print('Episode steps : ', Episode_steps, "average reward", 
            total_reward/Episode_steps)
        if env.Winner == 'Blue':
            blue_win_times += 1
        elif env.Winner == 'Red':
            red_win_times += 1
    print('\nBlue win rate : ', blue_win_times/n, 'Red win rate : ', 
        red_win_times/n, '%')

if __name__ == "__main__":
    run_n_episode(10)