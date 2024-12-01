import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from math import pi
import my_remote_spectra as rs


class Evolve():
    def __init__(self, kk, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
        self.kk = kk
        self.kk[np.where(kk==0)] = kk.mean()
        self.ze = ze
        self.Omega_b = Omega_b
        self.Omega_c = Omega_c
        self.w = w
        self.wa = wa
        self.Omega_K = Omega_K
        self.h = h
        self.T = rs.T(kk, Omega_b, Omega_c, w, wa, Omega_K, h)


    def ThreeD_Evolve(self, G_function_k, input_field_k):
        output_field_k = G_function_k*input_field_k

        return np.fft.ifftn(output_field_k)


    def RDF_component_Evolve_SW(self, psi_i_k, kk_component):
        "RDF for remote dipole field"
        G_function_k = self.T*rs.G_SW_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)*kk_component/self.kk

        return self.ThreeD_Evolve(G_function_k, psi_i_k)


    def RDF_component_Evolve_localDopp(self, psi_i_k, kk_component):
        "RDF for remote dipole field"
        G_function_k = self.T*rs.G_localDopp_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)*kk_component/self.kk

        return self.ThreeD_Evolve(G_function_k, psi_i_k)


    def RDF_gradient_Evolve_localDopp(self, psi_i_k):
        "RDF for remote dipole field"
        G_function_k = self.T*rs.G_localDopp_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)/self.kk

        return self.ThreeD_Evolve(G_function_k, psi_i_k)


    def RDF_component_Evolve_Dopp(self, psi_i_k, kk_component):
        "RDF for remote dipole field"
        G_function_k = self.T*rs.G_Dopp_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)*kk_component/self.kk

        return self.ThreeD_Evolve(G_function_k, psi_i_k)


    def RDF_component_Evolve_ISW(self, psi_i_k, kk_component):
        "RDF for remote dipole field"
        G_function_k = self.T*rs.G_ISW_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)*kk_component/self.kk

        return self.ThreeD_Evolve(G_function_k, psi_i_k)




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

def ThreeDField_on_TwoDSurface(x_grid, y_grid, z_grid, fieldThreeD, x, y, z):

    return RegularGridInterpolator((x_grid, y_grid, z_grid), fieldThreeD, method='nearest')((y, x, z))

def TwoDVecField_Radial(field_x1, field_x2, field_x3, theta, phi):

    return field_x1*np.sin(theta)*np.cos(phi) + field_x2*np.sin(theta)*np.sin(phi) + field_x3*np.cos(theta)
