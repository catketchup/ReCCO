import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from math import pi

class Evolve():
    def __init__(self, Omega_b, Omega_c, w, wa, Omega_K, h):
        self.Omega_b = Omega_b
        self.Omega_c = Omega_c
        self.w = w
        self.wa = wa
        self.Omega_K = Omega_K
        self.h = h

    # def 

def ThreeDEvolve(G_itp, input_g_field_k, kk):
    # here kk should be in 1/Mpc
    kk[np.where(kk==0)] = kk.mean()
    output_field_k = G_itp(kk)*input_g_field_k/kk

    return np.fft.ifftn(output_field_k)

def ThreeDEvolve_test(G_function, input_g_field_k, kk):
    kk[np.where(kk==0)] = kk.mean()
    # here kk should be in 1/Mpc
    output_field_k = G_function*input_g_field_k/kk

    return np.fft.ifftn(output_field_k)

def ThreeDEvolve_component(G_function, input_field_k, kk_i, kk):
    kk[np.where(kk==0)] = kk.mean()
    # here kk should be in 1/Mpc
    output_field_k = G_function*input_field_k*kk_i/kk

    return np.fft.ifftn(output_field_k)

def ThreeDField_on_TwoDSurface(x_grid, y_grid, z_grid, field3D, x, y, z):

    return RegularGridInterpolator((x_grid, y_grid, z_grid), field3D, method='nearest')((y, x, z))

def TwoDVecField_Radial(field_x1, field_x2, field_x3, theta, phi):

    return field_x1*np.sin(theta)*np.cos(phi) + field_x2*np.sin(theta)*np.sin(phi) + field_x3*np.cos(theta)
