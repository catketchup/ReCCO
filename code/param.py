import math

# speed of light
#kms^-1
c = 3.e5
# Hubble parameter, kms^-1Mpc^-1
H0 = 68
# observable universe radius,14260 Mpc
r_o = 15000
# the boundary position of the two bubbles, Mpc
x_c = 0
r_H = c/H0
# initial potential from bubble collision psi_i parametried by A and B, Eq.2.2
A = 1.e-4 # 1-sigma limit
B = 0.0

# A = 0
# B = 1.e-4 # 1-sigma limit

# A = 1
# B = 1
# need a resolution of the 3-d grid?
d = 1000
N = int(2*r_o/d) +1

curvature_ini_Class = 1
R_nu = 0
