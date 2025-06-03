#!/usr/bin/env python3
# Minimal LinUCB contextual bandit for pH-RL intervention selection.
import numpy as np
from typing import Tuple

class LinUCBBandit:
    def __init__(self, n_actions:int, n_features:int, alpha:float=1.0):
        self.n_a, self.n_f, self.alpha = n_actions, n_features, alpha
        self.A = [np.eye(n_features) for _ in range(n_actions)]
        self.b = [np.zeros((n_features,1)) for _ in range(n_actions)]

    def _theta(self,a:int)->np.ndarray:
        return np.linalg.solve(self.A[a], self.b[a])

    def select(self, ctx:np.ndarray)->int:
        x = ctx.reshape(-1,1)
        p = [ (self._theta(a).T @ x + self.alpha*np.sqrt(x.T @ np.linalg.inv(self.A[a]) @ x))[0,0]
              for a in range(self.n_a) ]
        return int(np.argmax(p))

    def update(self, ctx:np.ndarray, action:int, reward:float):
        x = ctx.reshape(-1,1)
        self.A[action] += x @ x.T
        self.b[action] += reward * x

def bandit_step(bandit:LinUCBBandit, features:dict, reward_fn)->Tuple[int,float]:
    ctx = np.array(list(features.values()), dtype=float)
    a  = bandit.select(ctx)
    r  = reward_fn(a)
    bandit.update(ctx, a, r)
    return a, r

__all__=["LinUCBBandit","bandit_step"]
