import control as ct
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from inputs import *


# === MAIN FUNCTION
def main():
    # Apply LQR controller
    Q = np.diag([1, 1, 1, 1, 1, 1])
    P = np.diag([1e-6, 1e-12, 1e-13])
    (K, S, E) = ct.lqr(A, B, Q, P)

    # Export gain matrix
    np.save("K_LQR", K)

    # New state matrix with feedback
    F = A - B @ K

    # Print partial results
    print("LQR matrices:")
    print("Q =", bmatrix(Q))
    print("P =", bmatrix(P))
    print("K =", bmatrix(K))
    print("F =", bmatrix(F))
    print("R =", bmatrix(S))  # Riccati matrix
    print()

    # Get new system poles
    new_poles = np.linalg.eig(F)[0]  # E TODO
    print("Poles:", bmatrix(new_poles))

    # Save poles for later use:
    np.save("poles_LQR", new_poles)

    # Print poles as python list source code:
    print("\npoles =", new_poles.tolist())

    # Initialize closed loop system
    sys_ClosedLoop = ct.ss(F, B, C, D)

    # Plot a pole/zero map for a linear system.
    plt.figure()
    ct.pzmap(sys_ClosedLoop, plot=True, grid=True,
             title="Pólos em malha fechada (LQR)")
    save_plot("ClosedLoop_LQR_pzmap.png")

    # Simulate step response
    T, yout = ct.step_response(sys_ClosedLoop, TIME, X0)

    # Plot step response
    plot_response(T, yout, [0, 1, 2], "θᵢ [rad]")
    plt.suptitle("Resposta ao degrau (LQR)")
    save_plot("ClosedLoop_LQR_step_thetas.png")

    # Plot control inputs
    plot_input(T, yout, K, [0, 1, 2], "uᵢ")
    plt.suptitle("Esforços para resposta ao degrau (LQR)")
    save_plot("ClosedLoop_LQR_step_inputs.png")

    # Print max input values
    print("\nMax input values:")
    u_count = len(U_LABELS)
    y = eval_input(yout, K)
    for i in range(u_count):
        print("max({}) = {:.2e}".format(U_LABELS[i], np.max(y[:, i])))

    # Show plots
    if SHOW_PLOTS:
        plt.show()


# === START
if __name__ == "__main__":
    main()
