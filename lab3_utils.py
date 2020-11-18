import numpy as np


def Id(t):
    # input stimulus
    return 0.0


def calculate_L(T, L_params):
    n_tics = len(T)
    res = np.zeros(n_tics)

    res[0] = L_params['L0']
    for i in range(2, n_tics):
        is_spiking = L_params['distr_func'](**L_params['distr_params']) > L_params['distr_threshold']
        res[i] = res[i - 1] + int(is_spiking) * L_params['L_in'] - L_params['L_out']
        res[i] = max(0.0, res[i])
    print(res)
    res = {T[i]: res[i] for i in range(n_tics)}

    def L_func(t):
        if t > 600:
            return 0.0
        return res[int(t)]

    return L_func