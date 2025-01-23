import os, sys
import numpy as np

class SYS:
    def __init__(self):
        pass

    def resample(self, particles, weights):
        """
        Systematic resampling for Particle Filter
        :param particles: N-array of particle state vectores of the form [x,y,vx,vy,Hx,Hy,s]
        :param weights: N-array of particle weights
        :return: Resampled particle state vectors
        """
        N = particles.shape[1]
        cdf = np.cumsum(weights)
        r = np.random.rand() / N
        res_particles = np.zeros(particles.shape)

        for i in range(N):
            idx = np.min(np.where(cdf >= r)[0])
            res_particles[:, i] = particles[:, idx]
            r = r + 1 / N

        res_weights = 1 / N * np.ones(weights.shape)

        return res_particles, res_weights