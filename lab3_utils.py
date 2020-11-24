import numpy as np


def Id(t):
    # input stimulus
    return 0.0


def calculate_L(T, L_params):
    # T in s
    n_tics = len(T)
    res = np.zeros(n_tics)

    res[0] = L_params['L0']
    delay = L_params['distr_func'](**L_params['distr_params'])
    for i in range(2, n_tics):
        is_spiking = False
        if delay <= 0:
            is_spiking = True
            delay = L_params['distr_func'](**L_params['distr_params'])

        res[i] = res[i - 1] + int(is_spiking) * L_params['L_in'] - L_params['L_out']
        res[i] = np.clip(res[i], 0.0, L_params['V_max'])
        delay -= 1
    #print(res)
    res = {T[i]: res[i] for i in range(n_tics)}
    print(res.values())

    def L_func(t):
        # input in s
        if t >= 6000:
            return 0.0
        return res[int(t)]
    
    return L_func