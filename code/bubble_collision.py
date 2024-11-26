import numpy as np
from math import pi
import my_remote_spectra as rs
import config as config
import kszpsz_config
import scipy.integrate as integrate

class BubbleCollision_Veff():
    def __init__(self, A, B, Z_c, Z_e, Omega_b, Omega_c, w, wa, Omega_K, h):
        self.A = A
        self.B = B
        self.Z_c = Z_c
        self.Z_e = Z_e
        self.Omega_b = Omega_b
        self.Omega_c = Omega_c
        self.w = w
        self.wa = wa
        self.Omega_K = Omega_K
        self.h = h
        self.chi_c = rs.chifromz(Z_c)
        self.chi_e = rs.chifromz(Z_e)
        self.chi_edec = rs.chifromz(config.zdec) - rs.chifromz(Z_e)
        self.r_H = 3.e5/(h*100)
        self.Dpsi_e = rs.Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+Z_e))
        self.Dpsi_dec = rs.Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+config.zdec))
        self.Dv_e = rs.Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+Z_e))
        self.Dv_dec = rs.Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+config.zdec))


    def Cos_theta_c(self, theta_e):
        if np.isscalar(theta_e):
            theta_e = np.array([theta_e])

        cos_theta_c = (self.chi_c -self.chi_e*np.cos(theta_e))/self.chi_edec
        cos_theta_c[np.where(abs(cos_theta_c)>=1)] = 1

        return cos_theta_c


    def Delta_cos_theta_c_n(self, theta_e, n):
        return 1 - self.Cos_theta_c(theta_e)**n


    def Approx_Veff_SW_radial_a(self, a, theta_e):
        A = self.A
        B = self.B

        chi_edec = rs.chifromz(1/a-1) - rs.chifromz(self.Z_e)
        cos_theta_e = np.cos(theta_e)

        # ensure chi_edec[-1] is not zero
        chi_edec[-1] = 1e-5
        cos_theta_c = (self.chi_c -self.chi_e*cos_theta_e)/chi_edec

        cos_theta_c[np.where(abs(cos_theta_c)>=1)] = 1

        return 3/2*cos_theta_e*(A/self.r_H*((self.chi_e*cos_theta_e-self.chi_c)*1/2*self.Delta_cos_theta_c_n(theta_e, 2)+1/3*self.chi_edec*self.Delta_cos_theta_c_n(theta_e, 3)) + B/(self.r_H**2)*((self.chi_e*cos_theta_e-self.chi_c)**2*1/2*self.Delta_cos_theta_c_n(theta_e, 2) + (self.chi_e*cos_theta_e-self.chi_c)*self.chi_edec*2/3*self.Delta_cos_theta_c_n(theta_e, 3) + 1/4*self.chi_edec**2*self.Delta_cos_theta_c_n(theta_e, 4)))



    def Approx_Veff_SW_radial(self, theta_e):
        # need to select the range of Z_e and theta_e
        A = self.A
        B = self.B
        cos_theta_c = self.Cos_theta_c(theta_e)
        cos_theta_e = np.cos(theta_e)

        return (2*self.Dpsi_dec-3/2)*3/2*cos_theta_e*(A/self.r_H*((self.chi_e*cos_theta_e-self.chi_c)*1/2*self.Delta_cos_theta_c_n(theta_e, 2)+1/3*self.chi_edec*self.Delta_cos_theta_c_n(theta_e, 3)) + B/(self.r_H**2)*((self.chi_e*cos_theta_e-self.chi_c)**2*1/2*self.Delta_cos_theta_c_n(theta_e, 2) + (self.chi_e*cos_theta_e-self.chi_c)*self.chi_edec*2/3*self.Delta_cos_theta_c_n(theta_e, 3) + 1/4*self.chi_edec**2*self.Delta_cos_theta_c_n(theta_e, 4)))


    def Approx_Veff_localDopp_radial(self, theta_e):
        A = self.A
        B = self.B
        cos_theta_e = np.cos(theta_e)
        step_array = np.ones_like(theta_e)
        step_array[np.where(cos_theta_e<=(self.chi_c/self.chi_e))]=0

        return -self.Dv_e/(self.r_H)*(A + 2*B/(self.r_H)*(self.chi_e*cos_theta_e - self.chi_c))*cos_theta_e*step_array


    def Approx_Veff_decDopp_radial(self, theta_e):
        A = self.A
        B = self.B
        cos_theta_c = self.Cos_theta_c(theta_e)
        cos_theta_e = np.cos(theta_e)

        return 3/2*cos_theta_e*self.Dv_dec*(1/3*A/self.r_H* self.Delta_cos_theta_c_n(theta_e, 3) +  2/3*B/(self.r_H**2)*(self.chi_e*cos_theta_e-self.chi_c)* self.Delta_cos_theta_c_n(theta_e, 3) + 1/2*B/(self.r_H**2)*self.chi_edec*self.Delta_cos_theta_c_n(theta_e, 4))


    def Approx_Veff_ISW_radial(self, theta_e):
        a = np.logspace(np.log10(config.adec), np.log10(rs.az(self.Z_e)), kszpsz_config.transfer_integrand_sampling)
        chi_a = rs.Chia_inter(self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)(a)

        Veff_ISW_radial = []

        # need to select the range of Z_e, theta_e and a, which have been accounted for in Approx_Veff_SW_radial_a(a, theta_e_i)
        for theta_e_i in theta_e:
            integrand = rs.derv_Dpsi_inter(self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)(a)*self.Approx_Veff_SW_radial_a(a, theta_e_i)

            Veff_ISW_radial.append(2*integrate.simps(integrand, a))

        return np.array(Veff_ISW_radial)

    def Approx_Veff_radial(self, theta_e):
        return self.Approx_Veff_SW_radial(theta_e) + self.Approx_Veff_ISW_radial(theta_e) + self.Approx_Veff_decDopp_radial(theta_e) + self.Approx_Veff_localDopp_radial(theta_e)


    def Approx_RQF_SW(self, theta_e):
        A = self.A
        B = self.B
        cos_theta_c = self.Cos_theta_c(theta_e)
        cos_theta_e = np.cos(theta_e)

        return (2*self.Dpsi_dec-3/2)*5/8*(3*cos_theta_e**2-1)*(A/self.r_H*(3/4*self.chi_edec*self.Delta_cos_theta_c_n(theta_e, 4) + (self.chi_e*cos_theta_e-self.chi_c)*self.Delta_cos_theta_c_n(theta_e, 3) -1/2*self.chi_edec*self.Delta_cos_theta_c_n(theta_e, 2) - (self.chi_e*cos_theta_e -self.chi_c)* self.Delta_cos_theta_c_n(theta_e, 1)) + B/(self.r_H**2)*(3/5*self.chi_edec**2*self.Delta_cos_theta_c_n(theta_e, 5) + 3/2*self.chi_edec*(self.chi_e*cos_theta_e-self.chi_c)*self.Delta_cos_theta_c_n(theta_e, 4) + 1/3*(-self.chi_edec**2 + 3*(self.chi_e*cos_theta_e-self.chi_c)**2)*self.Delta_cos_theta_c_n(theta_e,3) - self.chi_edec*(self.chi_e*cos_theta_e-self.chi_c)*self.Delta_cos_theta_c_n(theta_e, 2) - (self.chi_e*cos_theta_e-self.chi_c)**2))

    def Approx_RQF_SW_a(self, a, theta_e):

        A = self.A
        B = self.B

        chi_edec = rs.chifromz(1/a-1) - rs.chifromz(self.Z_e)
        cos_theta_e = np.cos(theta_e)

        # ensure chi_edec[-1] is not zero
        chi_edec[-1] = 1e-5
        cos_theta_c = (self.chi_c -self.chi_e*cos_theta_e)/chi_edec

        cos_theta_c[np.where(abs(cos_theta_c)>=1)] = 1

        return 5/8*(3*cos_theta_e**2-1)*(A/self.r_H*(3/4*self.chi_edec*self.Delta_cos_theta_c_n(theta_e, 4) + (self.chi_e*cos_theta_e-self.chi_c)*self.Delta_cos_theta_c_n(theta_e, 3) -1/2*self.chi_edec*self.Delta_cos_theta_c_n(theta_e, 2) - (self.chi_e*cos_theta_e -self.chi_c)* self.Delta_cos_theta_c_n(theta_e, 1)) + B/(self.r_H**2)*(3/5*self.chi_edec**2*self.Delta_cos_theta_c_n(theta_e, 5) + 3/2*self.chi_edec*(self.chi_e*cos_theta_e-self.chi_c)*self.Delta_cos_theta_c_n(theta_e, 4) + 1/3*(-self.chi_edec**2 + 3*(self.chi_e*cos_theta_e-self.chi_c)**2)*self.Delta_cos_theta_c_n(theta_e,3) - self.chi_edec*(self.chi_e*cos_theta_e-self.chi_c)*self.Delta_cos_theta_c_n(theta_e, 2) - (self.chi_e*cos_theta_e-self.chi_c)**2))



    def Approx_RQF_decDopp(self, theta_e):
        A = self.A
        B = self.B
        cos_theta_c = self.Cos_theta_c(theta_e)
        cos_theta_e = np.cos(theta_e)

        return self.Dv_dec* 5/8*(3*cos_theta_e**2-1)*(A/self.r_H*(3/4*self.Delta_cos_theta_c_n(theta_e, 4) - 1/2*self.Delta_cos_theta_c_n(theta_e, 2)) + 2*B/(self.r_H**2)*((3/5*self.chi_edec*self.Delta_cos_theta_c_n(theta_e, 5)) + 3/4*(self.chi_e*cos_theta_e-self.chi_c)*self.Delta_cos_theta_c_n(theta_e, 4) -1/3*self.chi_edec*self.Delta_cos_theta_c_n(theta_e, 3) - 1/2*(self.chi_e*cos_theta_e-self.chi_c)*self.Delta_cos_theta_c_n(theta_e, 2)))


    def Approx_RQF_ISW(self, theta_e):
        a = np.logspace(np.log10(config.adec), np.log10(rs.az(self.Z_e)), kszpsz_config.transfer_integrand_sampling)
        chi_a = rs.Chia_inter(self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)(a)

        RQF_ISW = []

        for theta_e_i in theta_e:
            integrand = rs.derv_Dpsi_inter(self.Omega_b, self.Omega_c, self.w, self.wa, self.Omega_K, self.h)(a)*self.Approx_RQF_SW_a(a, theta_e_i)

            RQF_ISW.append(2*integrate.simps(integrand, a))

        return np.array(RQF_ISW)

    def Approx_RQF(self, theta_e):
        return self.Approx_RQF_SW(theta_e) + self.Approx_RQF_decDopp(theta_e) + self.Approx_RQF_ISW(theta_e)
