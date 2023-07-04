import control as ct
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from helpers import *
from inputs import *


# === CONSTANTS
_DURATION = 5
_STEPS = 1000
_TIME = np.linspace(0, _DURATION, _STEPS)


# === MAIN FUNCTION
def main():
    # Load previous scripts results:
    K = np.load("K_PP.npy")
    Ko = np.load("Ko_PP.npy")

    # Apply separation principle to simulate both controller and observer effects
    a11 = A
    a12 = -B@K
    a21 = Ko@C
    a22 = A - B@K - Ko@C
    Lambda = np.block([[a11, a12], [a21, a22]])
    print("Lambda:", Lambda)

    # Initial conditions
    z0 = np.concatenate((X0, np.zeros(len(X0))))

    # State-space model
    def separation_principle(z, t):
        return Lambda @ z

    # Simulate
    z = odeint(separation_principle, y0=z0, t=_TIME)
    
    # Plot results
    fig, axs = plt.subplots(3, sharex=True)
    for i in range(3):
        axs[i].plot(_TIME, z[:, i], label=Y_LABELS[i])
        axs[i].plot(_TIME, z[:, i+6], label=Y_LABELS[i] + " (estimated)", linestyle="--")
        axs[i].legend()
        axs[i].grid()
    plt.xlabel('t [s]')
    plt.tight_layout()


    # Show plots
    # if SHOW_PLOTS:
    plt.show()


# === START
if __name__ == "__main__":
    main()
