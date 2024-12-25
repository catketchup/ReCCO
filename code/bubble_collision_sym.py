from sympy import *
import numpy as np
from math import pi
import my_remote_spectra as rs
import config as config
import kszpsz_config
import scipy.integrate as integrate

import importlib
importlib.reload(kszpsz_config)

class BubbleCollision_RF_sym():
    "symbolic Bubble-collision induced remote fields (RF) including remote dipole field (RDF) and remote quadrupole field (RQF)"
    theta_e = Symbol('theta_e')
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
        self.Delta_chi_dec = rs.chifromz(config.zdec) - rs.chifromz(Z_e)
        self.r_H = 3.e5/(h*100)
        self.Dpsi_e = rs.Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+Z_e))
        self.Dpsi_dec = rs.Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+config.zdec))
        self.Dv_e = rs.Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+Z_e))
        self.Dv_dec = rs.Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+config.zdec))

    def check_remote_region(self, theta_e_1d=None):
        if not np.any(theta_e_1d):
            theta_e_1d = np.arange(0,181,1)

        d1 = self.chi_e*np.cos(theta_e_1d) + self.Delta_chi_dec - self.chi_c
        d2 = self.chi_e*np.cos(theta_e_1d) - self.Delta_chi_dec - self.chi_c

        self.remote_region = [theta_e_1d[np.where((d1>0)&(d2<0))], theta_e_1d[np.where((d2>0))], theta_e_1d[np.where((d1<0))]]
        self.remote_region_exist = Array([np.any(self.region[0]), np.any(self.region[1]), np.any(self.region[2])])

        return self.region

    def check_local_region(self, theta_e_1d=None):
        if not np.any(theta_e_1d):
            theta_e_1d = np.arange(0,181,1)
        self.local_region = [theta_e_1d[np.where(self.chi_e*np.cos(theta_e_1d)>self.chi_c)], theta_e_1d[np.where(self.chi_e*np.cos(theta_e_1d)<self.chi_c)]]

        return self.local_region

    # test calling symbolic function
    def Cos_theta_c(self):
        str_Cos_theta_c = ['(chi_c - chi_e*cos(theta_e))/(Delta_chi_dec)', -1, 1]
        return Array(sympify(str_Cos_theta_c, evaluate=False))

    def Delta_cos_theta_c_n(self, n):
        str_Delta_cos_theta_c_n = []
        for i in range(3):
            str_Delta_cos_theta_c_n.append(f'1 - ({self.Cos_theta_c()[i]})**{n}')
        return Array(sympify(str_Delta_cos_theta_c_n, evaluate=False))

    def Y_s2l2m0(self):
        str_Y_s2l2m0 = '(3/4)*sqrt(5/(6*pi))*sin(theta_e)**2'
        return sympify(str_Y_s2l2m0, evaluate=False)

    def Y_l1m0(self):
        str_Y_l2m0 = '(1/2)*sqrt(3/(pi))*cos(theta_e)'
        return sympify(str_Y_l2m0, evaluate=False)

    def RDF_eff_SW(self):
        str_RDF_eff_SW = []
        for i in range(3):
            str_RDF_eff_SW.append(f'(2*D_psi_dec-3/2)*3/2*cos(theta_e)*(A/r_H*((chi_e*cos(theta_e)-chi_c)*\
            1/2*({self.Delta_cos_theta_c_n(2)[i]})+1/3*Delta_chi_dec*({self.Delta_cos_theta_c_n(3)[i]})) + \
            B/(r_H**2)*((chi_e*cos(theta_e)-chi_c)**2*1/2*({self.Delta_cos_theta_c_n(2)[i]}) + \
            (chi_e*cos(theta_e)-chi_c)*Delta_chi_dec*2/3*({self.Delta_cos_theta_c_n(3)[i]}) + \
            1/4*Delta_chi_dec**2*({self.Delta_cos_theta_c_n(4)[i]})))')

        return Array(sympify(str_RDF_eff_SW, evaluate=True))

    def RDF_eff_decDopp(self):
        str_RDF_eff_decDopp = []
        for i in range(3):
            str_RDF_eff_decDopp.append(f'(3/2)*cos(theta_e)*D_v_dec*(1/3*A/r_H*({self.Delta_cos_theta_c_n(3)[i]}) + \
            2/3*B/(r_H**2)*(chi_e*cos(theta_e)-chi_c)* ({self.Delta_cos_theta_c_n(3)[i]}) + \
            1/2*B/(r_H**2)*chi_edec*({self.Delta_cos_theta_c_n(4)[i]}))')

        return Array(sympify(str_RDF_eff_decDopp, evaluate=True))

    def RDF_eff_localDopp(self):
        str_RDF_eff_localDopp = []
        str_RDF_eff_localDopp.append(f'D_v_e/(r_H)*cos(theta_e)*(A + 2*B/(r_H)*(chi_e*cos(theta_e) - chi_c))')
        str_RDF_eff_localDopp.append(0)

        return Array(sympify(str_RDF_eff_localDopp, evaluate=True))


    def RQF_eff_SW(self):
        str_RQF_eff_SW = []
        for i in range(3):
            str_RQF_eff_SW.append(f'(2* D_psi_dec - 3/2)*5*sqrt(6)/16*sin(theta_e)**2*\
            (A/r_H*(3/4*Delta_chi_dec*({self.Delta_cos_theta_c_n(4)[i]})\
            + (chi_e*cos(theta_e)-chi_c)*({self.Delta_cos_theta_c_n(3)[i]}) \
            -1/2*Delta_chi_dec*({self.Delta_cos_theta_c_n(2)[i]}) - \
            (chi_e*cos(theta_e) - chi_c)* ({self.Delta_cos_theta_c_n(1)[i]})) + \
            B/(r_H**2)*(3/5*Delta_chi_dec**2*({self.Delta_cos_theta_c_n(5)[i]}) + \
            3/2*Delta_chi_dec*(chi_e*cos(theta_e)-chi_c)*({self.Delta_cos_theta_c_n(4)[i]}) + \
            1/3*(-Delta_chi_dec**2 + \
            3*(chi_e*cos(theta_e)-chi_c)**2)*({self.Delta_cos_theta_c_n(3)[i]}) - \
            Delta_chi_dec*(chi_e*cos(theta_e)-chi_c)*({self.Delta_cos_theta_c_n(2)[i]}) - \
            (chi_e*cos(theta_e)-chi_c)**2*({self.Delta_cos_theta_c_n(1)[i]})))')

        return Array(sympify(str_RQF_eff_SW, evaluate=True))

    def RQF_eff_Dopp(self):
        str_RQF_eff_Dopp = []
        for i in range(3):
            str_RQF_eff_Dopp.append(f'D_v_dec*5*sqrt(6)/16*sin(theta_e)**2*\
            (A/r_H*(3/4*({self.Delta_cos_theta_c_n(4)[i]}) \
            - 1/2*({self.Delta_cos_theta_c_n(2)[i]})) \
            + 2*B/(r_H**2)*(3/5*Delta_chi_dec*({self.Delta_cos_theta_c_n(5)[i]}) + \
            3/4*(chi_e*cos(theta_e)-chi_c)*({self.Delta_cos_theta_c_n(4)[i]}) -\
            1/3*Delta_chi_dec*({self.Delta_cos_theta_c_n(3)[i]}) - \
            1/2*(chi_e*cos(theta_e)-chi_c)*({self.Delta_cos_theta_c_n(2)[i]})))')

        return Array(sympify(str_RQF_eff_Dopp, evaluate=True))

    def v_l1m0_sym(self, name):
        theta_e = Symbol('theta_e')
        if name == 'SW':
            RDF = self.RDF_eff_SW()
        elif name == 'Dopp':
            RDF = self.RDF_eff_Dopp()

        return integrate(2*pi*sin(theta_e)*Y_l1m0()*RDF, theta_e)

    def q_s2l2m0_sym(self, name):
        theta_e = Symbol('theta_e')
        if name == 'SW':
            RQF = self.RQF_eff_SW()
        elif name == 'Dopp':
            RQF = self.RQF_eff_Dopp()
        return integrate(2*pi*sin(theta_e)*Y_s2l2m0()*RQF, theta_e)
