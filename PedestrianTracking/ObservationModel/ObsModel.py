import os, sys
import numpy as np

class ObservModel:
    def __init__(self,args):
        self.args = args

    def normpdf(selfself,x,mu,sigma):
        """
        :param x: Variable of PDF
        :param mu: Mean
        :param sigma: Variance
        :return:
        """
        return 1/(sigma*(2*np.pi)**0.5)*np.exp(-1*(x-mu)**2/(2*sigma**2))
