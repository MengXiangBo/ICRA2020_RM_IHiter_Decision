from IHiterEnv.env import ICRA_Env
from DuelingDQN import DuelingDQN
import numpy as np
import os
from IHiterEnv.parameter import TP
from IHiterEnv.policy import RandomPolicy
from IHiterEnv.agent import TeamAction

MaxEpisode = 2000
MaxEpisodeSteps = 100000

if not os.path.exists(os.path.abspath('.') \
    + '/train_data'):
    train_file = os.path.abspath('.') \
    + '/train_data' + '/'
    os.mkdir(train_file)

RLBrain = DuelingDQN(train_dir=train_file)
env = ICRA_Env()
team_action = TeamAction()

with open(train_file + 'param.txt', 'w') as f:
    f.writelines(["learning rate : " + str(RLBrain.LearningRate) + '\n',
        "ReplaceTargetIter", str(RLBrain.ReplaceTargetIter) + '\n',
        "BatchSize", str(RLBrain.BatchSize) + '\n',
        "Epsilon", str(RLBrain.Epsilon) + '\n',
        "EpsilonMin : " + str(RLBrain.EpsilonMin) + '\n',
        "Gamma", str(RLBrain.Gamma)])

def run_n_episode(n):
    print('============================================')
    blue_win_times, red_win_times = 0, 0
    for episode in range(1, n + 1):
        Episode_steps, total_reward = 0, 0
        obs= env.reset()
        for _ in range(MaxEpisodeSteps):
            Blue_Action = RLBrain.EvalDicision(obs)
            next_state, StepReward, isGameOver, _ = env.step(Blue_Action)
            obs = next_state
            Episode_steps += 1
            total_reward += StepReward
            if isGameOver:
                break
        print(' ')
        print('Episode : ', episode,'winner : ', env.Winner)
        print('Episode steps : ', Episode_steps, "average reward", total_reward/Episode_steps)
        if env.Winner == 'Blue':
            blue_win_times += 1
        elif env.Winner == 'Red':
            red_win_times += 1
    print('\nBlue win rate : ', blue_win_times/n, 'Red win rate : ', red_win_times/n, '%')

def train():
    blue_win_times, red_win_times = 0, 0
    for episode in range(1, MaxEpisode + 1):
        Episode_steps, total_reward = 0, 0
        obs= env.reset()
        for _ in range(MaxEpisodeSteps):
            Blue_Action = RLBrain.TrainDicision(obs)
            next_state, StepReward, isGameOver, _ = env.step(Blue_Action)
            RLBrain.StoreTransition(obs, Blue_Action, StepReward, next_state)
            RLBrain.Learn()
            obs = next_state
            Episode_steps += 1
            total_reward += StepReward
            if isGameOver:
                break
        print('-----------------------------------')
        print('Episode : ', episode,'winner : ', env.Winner)
        print('Episode steps : ', Episode_steps, "average reward", total_reward/Episode_steps)
        if env.Winner == 'Blue':
            blue_win_times += 1
        elif env.Winner == 'Red':
            red_win_times += 1

        if episode % 100 == 0:
            run_n_episode(10)

    print("the end")
    RLBrain.Save("final")


if __name__ == "__main__":
    train()
    # run_n_episode(10)