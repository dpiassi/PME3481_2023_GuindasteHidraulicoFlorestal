import control as ct
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from inputs import *

# === CONSTANTS
SHOW_PLOTS = False
DURATION = 100
STEPS = 100
TIME = np.linspace(0, DURATION, STEPS)


# === FUNCTIONS
def plot_response(T, yout, columns: list[int], ylabel: str):
    u_count = len(U_LABELS)
    fig, axs = plt.subplots(u_count, sharex=True)

    for i in range(u_count):
        for column in columns:
            axs[i].plot(T, yout[column, 0], label=Y_LABELS[column])
        axs[i].set_title(U_LABELS[i])
        axs[i].axhline(linewidth=0.5, color='gray')
        axs[i].axvline(linewidth=0.5, color='gray')
        box = axs[i].get_position()
        axs[i].set_position([box.x0, box.y0, box.width * 0.9, box.height])

    for ax in axs.flat:
        ax.set(ylabel=ylabel)

    # Position the legend outside of the subplot area
    labels = [Y_LABELS[column] for column in columns]
    plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', shadow=True)
    plt.xlabel('t [s]')



# === MAIN FUNCTION
def main():
    # Create the state-space model
    sys_OpenLoop = ct.ss(A, B, C, D)

    # Plot a pole/zero map for a linear system.
    plt.figure()
    ct.pzmap(sys_OpenLoop, plot=True, grid=True, title="Pólos em malha aberta")
    save_plot("OpenLoop_pzmap.png")

    # Simulate step response
    T, yout = ct.step_response(sys_OpenLoop, TIME, X0)

    # Plot step response
    plot_response(T, yout, [0, 2, 4], "θᵢ [rad]")   
    plt.suptitle("Resposta ao degrau (malha aberta)")
    save_plot("OpenLoop_step_thetas.png")

    plot_response(T, yout, [1, 3, 5], "ωᵢ [rad/s]")
    plt.suptitle("Resposta ao degrau (malha aberta)")
    save_plot("OpenLoop_step_omegas.png")

    # Apply LQR controller
    Q = np.diag([4, 4, 8, 8, 16, 16])
    P = np.diag([1, 1, 1])
    (K, S, E) = ct.lqr(A, B, Q, P)

    # New state matrix with feedback
    F = A - B @ K

    # Print partial results
    print("Q = ", bmatrix(Q))
    print("P = ", bmatrix(P))
    print("K = ", bmatrix(K))
    print("F = ", bmatrix(F))
    print("R = ", bmatrix(S)) # Riccati matrix

    # Get new system poles
    new_poles = np.linalg.eig(F)[0] # E TODO
    print("Poles:\n", bmatrix(new_poles))

    # Initialize closed loop system
    sys_ClosedLoop = ct.ss(F, B, C, D)

    # Plot a pole/zero map for a linear system.
    plt.figure()
    ct.pzmap(sys_ClosedLoop, plot=True, grid=True, title="Pólos em malha fechada (LQR)")
    save_plot("ClosedLoop_LQR_pzmap.png")

    # Simulate step response
    T, yout = ct.step_response(sys_ClosedLoop, TIME, X0)

    # Plot step response
    plot_response(T, yout, [0, 2, 4], "θᵢ [rad]")   
    plt.suptitle("Resposta ao degrau (LQR)")
    save_plot("ClosedLoop_LQR_step_thetas.png")

    plot_response(T, yout, [1, 3, 5], "ωᵢ [rad/s]")
    plt.suptitle("Resposta ao degrau (LQR)")
    save_plot("ClosedLoop_LQR_step_omegas.png")

    # TODO Generate observers
    QN = np.diag([1, 1, 1])
    RN = np.diag([1, 0, 1, 0, 1, 0])
    L, P, E = ct.lqe(sys_ClosedLoop, QN, RN)

    # Print partial results
    print("QN = ", bmatrix(QN))
    print("RN = ", bmatrix(RN))
    print("L = ", bmatrix(L))
    print("P = ", bmatrix(P)) # Riccati matrix
    print("E = ", bmatrix(E))


    # Show plots
    if SHOW_PLOTS:
        plt.show()


# === START
if __name__ == "__main__":
    main()