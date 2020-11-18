import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint

from model import n_0, m_0, h_0, generate_computing_derivatives_function, I_A_func, I_B_func
from lab3_utils import calculate_L, Id

# Set random seed (for reproducibility)
np.random.seed(30)


def plot_results(L_A_func, L_B_func, Vy, T, syn_params, plot_phase_space=False):
    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(np.array(T)*100, [I_A_func(T[i], Vy[i, 0], L_A_func, syn_params['V_A'], syn_params['N_A']) for i in range(len(T))])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('I synapse A')
    plt.grid()
    fig.show()

    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(np.array(T)*100, [I_B_func(T[i], Vy[i, 0], L_B_func, syn_params['V_B'], syn_params['N_B']) for i in range(len(T))])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('I synapse B')
    plt.grid()
    fig.show()

    # Neuron potential
    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(np.array(T)*100, Vy[:, 0])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Vm (mV)')
    ax.set_title('Neuron potential')
    plt.grid()
    fig.show()

    # n, m, h
    fig, ax = plt.subplots(figsize=(24, 14))
    for i, label in enumerate(['n', 'm', 'h']):
        ax.plot(np.array(T)*100, Vy[:, i + 1], label=label)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Potassium & Sodium ion-channel rates')
    ax.set_title('Potassium & Sodium ion-channel rates in time')
    ax.legend()
    plt.grid()
    fig.show()

    # Trajectories with limit cycles
    if plot_phase_space:
        for i, label in enumerate(['Vm - n', 'Vm - m', 'Vm - h']):
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.plot(Vy[:, 0], Vy[:, i + 1], label=label)
            ax.set_title('Limit cycles({})'.format(label))
            ax.legend()
            plt.grid()
            fig.show()


def run_simulation(Id, plot_phase_space=False):
    T = [int(t) for t in np.arange(0, 600, 1)]
    print(T)
    # A
    L_A_params = {
        'L0': 0.0,
        'distr_func': np.random.poisson,
        'distr_params': {'lam': 10},
        'distr_threshold': 10,
        'L_in': 0.01,
        'L_out': 0.005
    }
    L_A_func = calculate_L(T, L_A_params)


    # B
    L_B_params = {
        'L0': 0.0,
        'distr_func': np.random.uniform,
        'distr_params': {'low': 0, 'high': 10},
        'distr_threshold': 7,
        'L_in': 0.01,
        'L_out': 0.005
    }
    L_B_func = calculate_L(T, L_B_params)

    syn_params = {
        'V_A': 500,
        'N_A': 100000,
        'V_B': 500,
        'N_B': 100000,
    }

    Vm0 = -75.74
    Y = np.array([Vm0, n_0(Vm0), m_0(Vm0), h_0(Vm0)])

    compute_derivatives = generate_computing_derivatives_function(Id, L_A_func, L_B_func, syn_params)

    # Solve ODE system
    # Vy = (Vm[t0:tmax], n[t0:tmax], m[t0:tmax], h[t0:tmax])
    Vy = odeint(compute_derivatives, Y, T)

    plot_results(L_A_func, L_B_func, Vy, T, syn_params, plot_phase_space)


if __name__ == '__main__':
    ################################################################################
    #######################################    1   #################################
    run_simulation(Id, plot_phase_space=True)



