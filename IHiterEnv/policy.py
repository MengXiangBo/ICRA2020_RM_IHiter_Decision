from IHiterEnv.parameter import TP
import numpy as np

class Policy:

    def __init__(self):
        pass


class RandomPolicy(Policy):

    def React(self, joint_state=None):
        return np.random.randn(TP.ActionDim)