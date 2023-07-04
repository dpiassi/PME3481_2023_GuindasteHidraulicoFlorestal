import control as ct
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from inputs import *


# === CONSTANTS
_DURATION = 6
_STEPS = 1000
_TIME = np.linspace(0, _DURATION, _STEPS)


# === MAIN FUNCTION
def main():
    # === Apply Pole Placement observer
    observer_poles = np.array([-2.19E-01 + 3.24j, -2.19E-01 -3.24j, -6.04E-02 + 8.10E-01j, -6.04E-02 -8.10E-01j, -1.15, -1.18]) - np.ones(6)
    Ko = ct.place(np.transpose(A), np.transpose(C), observer_poles)
    Ko = np.transpose(Ko)

    # Print partial results
    print("Pole Placement matrices:")
    print("Ko =", bmatrix(Ko))

    # Observer matrix:
    O = A - Ko @ C
    print("O =", bmatrix(O))
    print()

    # Initialize closed loop system
    sys_error_pp = ct.ss(O, B, C, D)

    # Simulate step response
    T, yout = ct.step_response(sys_error_pp, _TIME, X0)

    # Plot step response
    plot_response(T, yout, [0, 1, 2], "θᵢ [rad]")
    plt.suptitle("Observador sintetizado por alocação de polos")
    save_plot("Observer_step_thetas.png")

    plot_response(T, yout, [3, 4, 5], "ωᵢ [rad/s]")
    plt.suptitle("Observador sintetizado por alocação de polos")
    save_plot("Observer_step_omegas.png")

    # Export for later use
    np.save('Ko_PP', Ko)

    # === Apply LQR observer
    Qo = np.diag([1, 1, 1, 1, 1, 1])
    Po = np.diag([1e-6, 1e-6, 1e-6, 1, 1, 1])
    (Ko, S, E) = ct.lqr(np.transpose(A), np.transpose(C), Qo, Po)
    Ko = np.transpose(Ko)

    # Print partial results
    print("LQR matrices:")
    print("Qo =", bmatrix(Qo))
    print("Po =", bmatrix(Po))
    print("Ko =", bmatrix(Ko))
    print("R =", bmatrix(S))  # Riccati matrix

    # Observer matrix:
    O = A - Ko @ C
    print("O =", bmatrix(O))
    print()

    # Initialize closed loop system
    sys_error_lq = ct.ss(O, B, C, D)

    # Simulate step response
    T, yout = ct.step_response(sys_error_lq, _TIME, X0)

    # Plot step response
    plot_response(T, yout, [0, 1, 2], "θᵢ [rad]")
    plt.suptitle("Observador sintetizado por LQ")
    save_plot("Observer_LQR_step_thetas.png")

    plot_response(T, yout, [3, 4, 5], "ωᵢ [rad/s]")
    plt.suptitle("Observador sintetizado por LQ")
    save_plot("Observer_LQR_step_omegas.png")

    # Export for later use
    np.save('Ko_LQR', Ko)

    # Show plots
    if SHOW_PLOTS:
        plt.show()


# === START
if __name__ == "__main__":
    main()
