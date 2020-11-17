import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint

from model import n_0, m_0, h_0, generate_computing_derivatives_function
from lab3_utils import calculate_L, Id

# Set random seed (for reproducibility)
np.random.seed(30)


def plot_results(Idv, Vy, T, plot_phase_space=False):
    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(T, Idv)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(r'Current density (uA/$cm^2$)')
    ax.set_title('Stimulus (Current density)')
    plt.grid()
    fig.show()

    # Neuron potential
    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(T, Vy[:, 0])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Vm (mV)')
    ax.set_title('Neuron potential')
    plt.grid()
    fig.show()

    # n, m, h
    fig, ax = plt.subplots(figsize=(24, 14))
    for i, label in enumerate(['n', 'm', 'h']):
        ax.plot(T, Vy[:, i + 1], label=label)
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


def run_simulation(tmin, tmax, Id, plot_phase_space=False):
    n_tics = 777
    T = np.linspace(tmin, tmax, n_tics)

    # A
    L_A_params = {
        'L0': 1e-4,
        'distr_func': np.random.poisson,
        'distr_params': {'lam': 10},
        'distr_threshold': 3,
        'L_in': 100,
        'L_out': 10
    }
    L_A_func = calculate_L(T, L_A_params)


    # B
    L_B_params = {
        'L0': 1e-4,
        'distr_func': np.random.uniform,
        'distr_params': {'low': 0, 'high': 10},
        'distr_threshold': 5,
        'L_in': 100,
        'L_out': 10
    }
    L_B_func = calculate_L(T, L_B_params)

    syn_params = {
        'V_A': 100,
        'N_A': 10000,
        'V_B': 100,
        'N_B': 5000,
    }

    Vm0 = -65.0
    Y = np.array([Vm0, n_0(Vm0), m_0(Vm0), h_0(Vm0)])

    compute_derivatives = generate_computing_derivatives_function(Id, L_A_func, L_B_func, syn_params)

    # Solve ODE system
    # Vy = (Vm[t0:tmax], n[t0:tmax], m[t0:tmax], h[t0:tmax])
    Vy = odeint(compute_derivatives, Y, T)

    # Input stimulus
    Idv = [Id(t) for t in T]

    plot_results(Idv, Vy, T, plot_phase_space)


if __name__ == '__main__':
    ################################################################################
    #######################################    1   #################################
    tmin = 0.0
    tmax = 600.0

    run_simulation(tmin, tmax, Id, plot_phase_space=False)



