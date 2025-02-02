import os, sys
import numpy as np
import cv2

from PedestrianTracking.ObservationModel.ObsModel import ObservModel

"""
Observation Model based on color histograms
"""

class ClrHisto(ObservModel):
    def __init__(self,args):
        super(ClrHisto,self).__init__(args)

    def initialize(self, state_vectors, frame):
        """

        :param state_vectors: set of state vectors 7*N
        :param frame: Current frame
        :return:
        """
        N = state_vectors.shape[1]
        #iterate over particles and calculate color histogram for bounding box
        for n in range(N):
            particle_state = state_vectors[:,n]

            self.xl = max(int(particle_state[0] + 0.1 * particle_state[4]), 0)
            self.yl = max(int(particle_state[1] + 0.1 * particle_state[5]), 0)
            self.xr = min(int(particle_state[0] + 0.9 * particle_state[4] + 1), frame.shape[1])
            self.yr = min(int(particle_state[1] + 0.9 * particle_state[5] + 1), frame.shape[0])

            target = frame[self.yl:self.yr, self.xl:self.xr, :]

            #Convert target image to HSV-frame for more robust tracking
            col_switch = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

            # Calculate color histgram of target
            bins = cv2.calcHist([col_switch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            bins = bins.flatten()

            # normalize histogram
            if np.sum(bins) != 0:
                bins = 1 / np.sum(bins) * bins

            if n == 0:
                p = bins
            else:
                p = np.vstack((p, bins))

        return p

    def distance(self, phi_t, phi_c):
        """
        Calculates distance between measurement vectors of target and observation
        :param phi_t: feature vector of target 1*512
        :param phi_c: feature vector of particles N*512
        :return:
        """
        N = phi_c.shape[0]
        d = np.array([])
        for n in range(N):
            d = np.append(d, np.array([cv2.compareHist(phi_t, phi_c[n, :], cv2.HISTCMP_BHATTACHARYYA)]), axis=0)

        return d

    def likelihood(self, phi_t, phi_c, mu, sigma):
        """
        Computes distribution of weights based on distance between measurement vectors of target and obervation
        :param phi_t: vector of target
        :param phi_c: vector of particles
        :return: weights of particles
        """
        d = self.distance(phi_t, phi_c)

        mu_vec = mu*np.ones(d.shape)
        sigma_vec = sigma*np.ones(d.shape)

        weights = np.array(list(map(self.normpdf, d, mu_vec, sigma_vec)))

        if np.sum(weights, axis=0) >= 1e-14:
            weights = 1 / np.sum(weights, axis=0) * weights
        else:
            weights = 1 / self.args.N * np.ones(self.args.N)

        return weights


