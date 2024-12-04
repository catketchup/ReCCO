import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from math import pi
import my_remote_spectra as rs
import importlib
importlib.reload(rs)

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


    def ThreeD_Evolve(self, G_k, input_field_k):
        output_field_k = G_k*input_field_k
        return np.fft.ifftn(output_field_k)


    def RDF_Evolve(self, name, input_field_k, kk_component, itp='False', ks='None'):
        if itp =='True':
            kk = ks
        else:
            kk = self.kk

        T = rs.T(kk, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)

        if name=='SW':
            G_k = T*rs.G_SW_ksz(kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
        elif name=='localDopp':
            G_k = T*rs.G_localDopp_ksz(kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
        elif name=='Dopp':
            G_k = T*rs.G_Dopp_ksz(kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
        elif name=='ISW':
            G_k = T*rs.G_ISW_ksz(kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)


        if itp =='True':
            G_k = interp1d(ks, G_k)(self.kk)


        return np.array([self.ThreeD_Evolve(G_k*kk_component[0]/self.kk,input_field_k), self.ThreeD_Evolve(G_k*kk_component[1]/self.kk,input_field_k), self.ThreeD_Evolve(G_k*kk_component[2]/self.kk,input_field_k)])

    def RDF_g_Evolve(self, name, input_g_field_k):

        if name=='SW':
            G_k = self.T*rs.G_SW_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
        elif name=='localDopp':
            G_k = self.T*rs.G_localDopp_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
        elif name=='Dopp':
            G_k = self.T*rs.G_Dopp_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
        elif name=='ISW':
            G_k = self.T*rs.G_ISW_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
        elif name=='total':
            G_k = self.T*rs.G_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)

        return np.array([self.ThreeD_Evolve(G_k/self.kk, input_g_field_k[0]), self.ThreeD_Evolve(G_k/self.kk, input_g_field_k[1]), self.ThreeD_Evolve(G_k/self.kk, input_g_field_k[2])])

    

    def RQF_g_Evolve(self, name, input_g_field_k):

        if name=='SW':
            G_k = self.T*rs.G_SW_psz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
        elif name=='localDopp':
            G_k = self.T*rs.G_localDopp_psz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
        elif name=='Dopp':
            G_k = self.T*rs.G_Dopp_psz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
        elif name=='ISW':
            G_k = self.T*rs.G_ISW_psz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
        elif name=='total':
            G_k = self.T*rs.G_psz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)

        return np.array([self.ThreeD_Evolve(G_k/self.kk, input_g_field_k[0]), self.ThreeD_Evolve(G_k/self.kk, input_g_field_k[1]), self.ThreeD_Evolve(G_k/self.kk, input_g_field_k[2])])



    # def RDF_g_Evolve(self, name, input_g_field_k, itp='False', ks='None'):
    #     if itp =='True':
    #         kk = ks
    #     else:
    #         kk = self.kk

    #     T = rs.T(kk, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)

    #     if name=='SW':
    #         G_k = T*rs.G_SW_ksz(kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
    #     elif name=='localDopp':
    #         G_k = T*rs.G_localDopp_ksz(kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
    #     elif name=='Dopp':
    #         G_k = T*rs.G_Dopp_ksz(kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
    #     elif name=='ISW':
    #         G_k = T*rs.G_ISW_ksz(kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)
    #     elif name=='total':
    #         G_k = T*rs.G_ksz(kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)

    #     if itp =='True':
    #         G_k = interp1d(ks, G_k)(self.kk)

    #     return np.array([self.ThreeD_Evolve(G_k/self.kk, input_g_field_k[0]), self.ThreeD_Evolve(G_k/self.kk, input_g_field_k[1]), self.ThreeD_Evolve(G_k/self.kk, input_g_field_k[2])])







    def ThreeDVec_Projection(self, grid, RDF_3d, points):
        return np.array([RegularGridInterpolator(grid, RDF_3d[0])(points), RegularGridInterpolator(grid, RDF_3d[1])(points), RegularGridInterpolator(grid, RDF_3d[2])(points)])

    def ThreeDVec_Projection_Radial(self, grid, RDF_3d, points, angle):
        theta = angle[0]
        phi = angle[1]
        RDF_vec = self.ThreeDVec_Projection(grid, RDF_3d, points)

        return RDF_vec[0]*np.sin(theta)*np.cos(phi) + RDF_vec[1]*np.sin(theta)*np.sin(phi) + RDF_vec[2]*np.cos(theta)



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




    # def RDF_component_Evolve_SW(self, psi_i_k, kk_component):
    #     "RDF for remote dipole field"
    #     G_k = self.T*rs.G_SW_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)*kk_component/self.kk

    #     return self.ThreeD_Evolve(G_k, psi_i_k)

    # def RDF_gradient_Evolve_SW(self, g_psi_i_k):
    #     "RDF for remote dipole field"
    #     G_k = self.T*rs.G_SW_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)/self.kk

    #     return self.ThreeD_Evolve(G_k, g_psi_i_k)


    # def RDF_component_Evolve_localDopp(self, psi_i_k, kk_component):
    #     "RDF for remote dipole field"
    #     G_k = self.T*rs.G_localDopp_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)*kk_component/self.kk

    #     return self.ThreeD_Evolve(G_k, psi_i_k)


    # def RDF_gradient_Evolve_localDopp(self, g_psi_i_k):
    #     "RDF for remote dipole field"
    #     G_k = self.T*rs.G_localDopp_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)/self.kk

    #     return self.ThreeD_Evolve(G_k, g_psi_i_k)


    # def RDF_component_Evolve_Dopp(self, psi_i_k, kk_component):
    #     "RDF for remote dipole field"
    #     G_k = self.T*rs.G_Dopp_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)*kk_component/self.kk

    #     return self.ThreeD_Evolve(G_k, psi_i_k)


    # def RDF_component_Evolve_ISW(self, psi_i_k, kk_component):
    #     "RDF for remote dipole field"
    #     G_k = self.T*rs.G_ISW_ksz(self.kk, self.ze, self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)*kk_component/self.kk

    #     return self.ThreeD_Evolve(G_k, psi_i_k)
