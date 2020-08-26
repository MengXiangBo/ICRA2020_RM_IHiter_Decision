from IHiterEnv.env import *


if __name__ == "__main__":
    env = ICRA_Env()
    blue = RandomPolicy()
    ob = env.reset()

    while True:
        env.render()
        act = blue.React(ob)
        state, reward, isDone, _ = env.step(act)
        if env.EpisodeStep % 10000 == 0:
            print(env.reset())
        if isDone:

            # print("============================================")
            env.reset()
        # print(a)
        # time.sleep(0.01)
        # print("here")