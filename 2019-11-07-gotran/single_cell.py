import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import tentusscher_model_2004_M as model


def tentusscher(plot=True):

    # Initial states
    y0 = model.init_state_values()

    # Parameters
    parameters = model.init_parameter_values()

    # Time steps
    tsteps = np.arange(0.0, 1000.0, 1e-1)

    # Solve ODE
    y = odeint(model.rhs, y0, tsteps, args=(parameters,))

    # Extract the membrane potential
    V_idx = model.state_indices("V")
    V = y.T[V_idx]

    # Extract monitored values
    monitor = np.array([model.monitor(r, t, parameters) for r, t in zip(y, tsteps)])
    i_Kr_idx = model.monitor_indices("i_Kr")
    i_Kr = monitor.T[i_Kr_idx]

    if plot:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(tsteps, V)
        ax[0].set_title("Voltage")
        ax[1].plot(tsteps, i_Kr)
        ax[1].set_title("iKr")
        ax[1].set_xlabel("Time (ms)")
        fig.savefig("tentusscher")
    return V, i_Kr


def tentusscher_Kr_blocker():

    # Initial states
    y0 = model.init_state_values()

    # Parameters
    parameters = model.init_parameter_values()

    # Get index of conductivity in potassium channel
    g_Kr_idx = model.parameter_indices("g_Kr")

    # Reduce conductivity by 50 %
    parameters[g_Kr_idx] *= 0.5

    # Time steps
    tsteps = np.arange(0.0, 1000.0, 1e-1)

    # Solve ODE
    y = odeint(model.rhs, y0, tsteps, args=(parameters,))

    # Extract the membrane potential
    V_idx = model.state_indices("V")
    V_block = y.T[V_idx]

    # Extract monitored values
    monitor = np.array([model.monitor(r, t, parameters) for r, t in zip(y, tsteps)])
    i_Kr_idx = model.monitor_indices("i_Kr")
    i_Kr_block = monitor.T[i_Kr_idx]

    V, i_Kr = tentusscher(plot=False)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(tsteps, V, label="Original")
    ax[0].plot(tsteps, V_block, linestyle="--", label="Block g_Kr")
    ax[0].set_title("Voltage")
    ax[1].plot(tsteps, i_Kr, label="Original")
    ax[1].plot(tsteps, i_Kr_block, linestyle="--", label="Block g_Kr")
    ax[1].set_title("iKr")
    ax[1].set_xlabel("Time (ms)")
    ax[0].legend()
    ax[1].legend()
    fig.savefig("tentusscher_Kr_blocker")


if __name__ == "__main__":
    tentusscher()
    # tentusscher_Kr_blocker()
