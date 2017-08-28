# -*- coding: utf-8 -*-

import numpy as np
def fast_moving_average(x,N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]