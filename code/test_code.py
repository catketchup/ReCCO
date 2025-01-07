from sympy import *
from sympy.tensor.array.expressions import ArrayTensorProduct
import numpy as np
import my_remote_spectra as rs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import param
import config as config
import kszpsz_config as kszpsz_config
import pdb; pdb.set_trace()

Omega_b = config.Omega_b
Omega_c = config.Omega_c
w = config.w
wa = config.wa
Omega_K = config.Omega_K
h = config.h
As= config.As
ns = config.ns
tau = config.tau

breakpoint()
rs.z_re(Omega_b, h, tau)
