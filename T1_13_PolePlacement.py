import control as ct
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from inputs import *


# === MAIN FUNCTION
def main():
    # Target poles:
    p = [-2.19E-01 + 3.24j, -2.19E-01 -3.24j, -6.04E-02 + 8.10E-01j, -6.04E-02 -8.10E-01j, -1.15, -1.18]

    # Place closed loop poles:
    K = ct.place(A, B, p)
    # K = ct.acker(A, B, p)

    # Export gain matrix
    np.save("K_PP", K)

    # New state matrix with feedback
    F = A - B @ K

    # Print gain matrix
    print("K =", bmatrix(K))
    print()

    # Get new system poles
    new_poles = np.linalg.eig(F)[0]  # E TODO
    print("Poles:", bmatrix(new_poles))

    # Initialize closed loop system
    sys_ClosedLoop = ct.ss(F, B, C, D)

    # Plot a pole/zero map for a linear system.
    plt.figure()
    ct.pzmap(sys_ClosedLoop, plot=True, grid=True,
             title="Pólos em malha fechada (alocação de polos)")
    save_plot("ClosedLoop_pzmap.png")

    # Simulate step response
    T, yout = ct.step_response(sys_ClosedLoop, TIME, X0)

    # Plot step response
    plot_response(T, yout, [0, 1, 2], "θᵢ [rad]")
    plt.suptitle("Resposta ao degrau (alocação de polos)")
    save_plot("ClosedLoop_step_thetas.png")

    plot_response(T, yout, [3, 4, 5], "ωᵢ [rad/s]")
    plt.suptitle("Resposta ao degrau (alocação de polos)")
    save_plot("ClosedLoop_step_omegas.png")

    # Plot control inputs
    plot_input(T, yout, K, [0, 1, 2], "uᵢ")
    plt.suptitle("Esforços para resposta ao degrau (alocação de polos)")
    save_plot("ClosedLoop_step_inputs.png")

    # Show plots
    if SHOW_PLOTS:
        plt.show()


# === START
if __name__ == "__main__":
    main()
