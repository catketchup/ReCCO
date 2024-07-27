import numpy as np
from scipy.interpolate import interp1d
import time
import redshifts
from math import lgamma
import loginterp
import common as c
import copy
from scipy.linalg import cholesky
from scipy import linalg
import healpy

import config as conf
import estim
from estim import estimator
data_lmax = 4000

nside = 20
lmax = 4000

estimator = estimator(data_lmax, conf)


Cls = estimator.load_theory_Cl('vr', 'vr')
vmap, vlm = estimator.get_maps_and_alms(['vr'], nside, lmax)
# print(type(Cls))
print(Cls.shape)
