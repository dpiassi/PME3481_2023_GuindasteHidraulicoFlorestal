import control as ct
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from inputs import *


# === CONSTANTS
_DURATION = 0.5
_STEPS = 200
_TIME = np.linspace(0, _DURATION, _STEPS)


# === MAIN FUNCTION
def main():
    # === Apply LQR observer
    Qo = np.diag([1, 1, 1, 1, 1, 1])
    Po = np.diag([1e-2, 1e-3, 5e-3, 1, 1, 1])
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

    # Export for later use
    np.save('Ko_LQR', Ko)

    # Print LQR observer poles as python list source code:
    print("poles =", np.linalg.eig(O)[0].tolist())
    print()

    # === Apply Pole Placement observer
    controller_poles = np.load("poles_PP.npy")
    observer_poles = controller_poles - 10 * np.ones(6)
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

    # Export for later use
    np.save('Ko_PP', Ko)

    # Show plots
    if SHOW_PLOTS:
        plt.show()


# === START
if __name__ == "__main__":
    main()
