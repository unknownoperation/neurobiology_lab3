import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint

from model import n_0, m_0, h_0, generate_computing_derivatives_function
from lab3_utils import calculate_L, Id

# Set random seed (for reproducibility)
np.random.seed(30)


def plot_results(Vy, T, plot_phase_space=False):
    # Neuron potential
    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(np.array(T) * 10, Vy[:, 0])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Vm (mV)')
    ax.set_title('Neuron potential')
    plt.grid()
    fig.show()
    #
    # n, m, h
    fig, ax = plt.subplots(figsize=(24, 14))
    for i, label in enumerate(['n', 'm', 'h']):
        ax.plot(np.array(T) * 10, Vy[:, i + 1], label=label)
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


def check_spikes_ratio(Vy, T):
    def calc_spikes_ratio_by_window(Window_size):
        spikes_ratio = []
        for i in range(len(Vy) - Window_size - 1):
            spikes = 0
            for j in range(i, i + Window_size):
                if Vy[j][0] < 0.0 and Vy[j + 1][0] > 0.0:
                    spikes += 1
            spikes_ratio.append(spikes)
        return np.array(spikes_ratio)

    spikes_in_500ms = calc_spikes_ratio_by_window(50)

    # 1
    print("No zero spikes for 500ms: {}".format(np.all(spikes_in_500ms > 0)))

    # plot spikes ratio
    pos_ = Vy[:, 0] > 0
    HZ = np.array(np.split(np.array([False] + list((pos_[1:] == True) & (pos_[:-1] == False))), 60)).sum(axis=1)
    dots = np.where(HZ < 10)[0]
    dots10 = np.where(HZ >= 10)[0]
    dots15 = np.where(HZ >= 15)[0]

    plt.figure(constrained_layout=True, figsize=(24, 14))
    plt.plot(range(1, 61), HZ, label='Spike ratio')
    plt.axhline(y=10, color='y', linestyle='--', label='10Hz')
    plt.axhline(y=15, color='r', linestyle='--', label='15Hz')
    plt.scatter(dots + 1, HZ[dots], s=100, marker='o', c='g')
    plt.scatter(dots10 + 1, HZ[dots10], s=100, marker='o', c='y')
    plt.scatter(dots15 + 1, HZ[dots15], s=100, marker='o', c='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spike ratio')
    plt.legend()
    plt.show()


def plot_I(T, Vy_func, Y, Vy_old):
    Vy = np.array([Vy_func([Vy_old[i, 0], Vy_old[i, 1], Vy_old[i, 2], Vy_old[i, 3], 0, 0, 0, 0], t)
          for i, t in enumerate(T)])

    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(np.array(T) * 10, Vy[:, 4])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('I synapse A (nA)')
    plt.grid()
    fig.show()

    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(np.array(T) * 10, Vy[:, 5])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('I synapse B (nA)')
    plt.grid()
    fig.show()

    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(np.array(T) * 10, Vy[:, 6])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('I Na (nA)')
    plt.grid()
    fig.show()

    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(np.array(T) * 10, Vy[:, 7])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('I K (nA)')
    plt.grid()
    fig.show()


def run_simulation(Id, plot_phase_space=False):
    T = [int(t) for t in np.arange(0, 6000, 1)]
    print(T)


    L_A_params = {
        'L0': 0.0,
        'distr_func': np.random.poisson,
        'distr_params': {'lam': 23},
        'L_in': 1000,
        'L_out': 100,
        'V_max': 1000
    }

    L_B_params = {
        'L0': 0.0,
        'distr_func': np.random.uniform,
        'distr_params': {'low': 1, 'high': 7},
        'L_in': 1000,
        'L_out': 100,
        'V_max': 1000
    }

    L_A_func = calculate_L(T, L_A_params)
    L_B_func = calculate_L(T, L_B_params)

    syn_params = {
        'V_A': 900,
        'N_A': 6.877 * 1e9,
        'V_B': 1000,
        'N_B': 6.58 * 1e9,
    }

    Vm0 = -75.74
    Y = np.array([Vm0, n_0(Vm0), m_0(Vm0), h_0(Vm0), 0.0, 0.0, 0.0, 0.0])

    compute_derivatives = generate_computing_derivatives_function(Id, L_A_func, L_B_func, syn_params)

    # Solve ODE system
    Vy = odeint(compute_derivatives, Y, T)

    plot_results(Vy, T, plot_phase_space)
    check_spikes_ratio(Vy, T)
    plot_I(T, compute_derivatives, Y, Vy)


if __name__ == '__main__':
    ################################################################################
    #######################################    1   #################################
    run_simulation(Id, plot_phase_space=True)



