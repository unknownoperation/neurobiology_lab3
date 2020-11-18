import numpy as np


gK = 36.0  # (mS/cm^2)
gNa = 120.0  # (mS/cm^2)

Cm = 1.0  # (uF/cm^2)

VK = -77.0  # EK(mV)
VNa = 50.0  # EK(mV)
Vl = -54.387  # EK(mV)


KdA = 128  # nM
KdB = 500  # nM

VeqA = -65  # mV
VeqB = 55  # mV

gammaA = 8  # pS
gammaB = 10  # pS


def alpha_n(Vm):
    return 0.01 * (Vm + 55.0) / (1.0 - np.exp(-0.1 * (Vm + 55.0)))


def beta_n(Vm):
    return 0.125 * np.exp(-0.0125 * (Vm + 65.0))


def alpha_m(Vm):
    return 0.1 * (Vm + 40.0) / (1.0 - np.exp(-0.1 * (Vm + 40.0)))


def beta_m(Vm):
    return 4.0 * np.exp(-0.0556 * (Vm + 65.0))


def alpha_h(Vm):
    return 0.07 * np.exp(-0.05 * (Vm + 65.0))


def beta_h(Vm):
    return 1.0 / (1.0 + np.exp(-0.1 * (Vm + 35.0)))


def n_0(Vm=0.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))


def m_0(Vm=0.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))


def h_0(Vm=0.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))


def I_A_func(t0, Vm, L_A_func, V_A, N_A):
    P_A = 1.0 / (1.0 + KdA * V_A / (L_A_func(t0) + 1e-9))
    I_A = N_A * P_A * (Vm - VeqA) * gammaA
    return I_A


def I_B_func(t0, Vm, L_B_func, V_B, N_B):
    P_B = 1.0 / (1.0 + KdB * V_B / (L_B_func(t0) + 1e-9))
    I_B = N_B * P_B * (Vm - VeqB) * gammaB
    return I_B


def generate_computing_derivatives_function(input_stimulus_func, L_A_func, L_B_func, syn_params):
    def compute_derivatives(y, t0):
        dy = np.zeros((4,))

        Vm = y[0]
        n = y[1]
        m = y[2]
        h = y[3]

        # dVm/dt
        GK = (gK / Cm) * np.power(n, 4.0)
        GNa = (gNa / Cm) * np.power(m, 3.0) * h

        I_A = I_A_func(t0, Vm, L_A_func, syn_params['V_A'],  syn_params['N_A'])
        I_B = I_B_func(t0, Vm, L_B_func, syn_params['V_B'], syn_params['N_B'])

        dy[0] = (input_stimulus_func(t0) / Cm) - (GK * (Vm - VK)) - (GNa * (Vm - VNa)) - I_A / Cm - I_B / Cm

        # dn/dt
        dy[1] = (alpha_n(Vm) * (1.0 - n)) - (beta_n(Vm) * n)

        # dm/dt
        dy[2] = (alpha_m(Vm) * (1.0 - m)) - (beta_m(Vm) * m)

        # dh/dt
        dy[3] = (alpha_h(Vm) * (1.0 - h)) - (beta_h(Vm) * h)

        return dy
    return compute_derivatives
