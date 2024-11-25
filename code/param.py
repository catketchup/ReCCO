import math
import config

# speed of light
c = 3.e5 #kms^-1

# Hubble parameter, kms^-1Mpc^-1
# observable universe radius,14260 Mpc
# the boundary position of the two bubbles, Mpc

z_c = 0
r_H = c/config.H0 # Mpc
direction = 'z'

# initial potential from bubble collision psi_i parametried by A and B, Eq.2.2
# A = 1.e-4 # 1-sigma limit
# B = 0

A =  0# 1-sigma limit
B = 1.e-4

curvature_ini_Class = 1
R_nu = 0
