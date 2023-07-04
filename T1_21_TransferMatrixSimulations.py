import control as ct
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from inputs import *


# === MAIN FUNCTION
def main():
    # === Open Loop
    # Evaluate state-transition matrices:
    Phi, Gamma = build_transfer_matrix(A, DELTA_TIME)

    # Simulate response
    X = solve_transfer_matrix(Phi, Gamma, B, X0, TIME)

    # Plot step response
    plot_response(TIME, X, [0, 1, 2], "θᵢ [rad]")
    plt.suptitle("Resposta ao degrau (malha aberta)")
    save_plot("TransferMatrix_OpenLoop_thetas.png")

    plot_response(TIME, X, [3, 4, 5], "ωᵢ [rad/s]")
    plt.suptitle("Resposta ao degrau (malha aberta)")
    save_plot("TransferMatrix_OpenLoop_omegas.png")

    # === Closed Loop
    # Load previous scripts results:
    K_PP = np.load("K_PP.npy")
    K_LQR = np.load("K_LQR.npy")
    Ko_PP = np.load("Ko_PP.npy")
    Ko_LQR = np.load("Ko_LQR.npy")

    # === Closed Loop === Pole Placement
    # Evaluate state-transition matrices:
    Phi, Gamma = build_transfer_matrix(A - B @ K_PP, DELTA_TIME)

    # Simulate response
    X = solve_transfer_matrix(Phi, Gamma, B, X0, TIME)

    # Plot step response
    plot_response(TIME, X, [0, 1, 2], "θᵢ [rad]")
    plt.suptitle("Resposta ao degrau (alocação de polos)")
    save_plot("TransferMatrix_ClosedLoop_PP_thetas.png")

    # === Closed Loop === LQR
    # Evaluate state-transition matrices:
    Phi, Gamma = build_transfer_matrix(A - B @ K_LQR, DELTA_TIME)

    # Simulate response
    X = solve_transfer_matrix(Phi, Gamma, B, X0, TIME)

    # Plot step response
    plot_response(TIME, X, [0, 1, 2], "θᵢ [rad]")
    plt.suptitle("Resposta ao degrau (LQR)")
    save_plot("TransferMatrix_ClosedLoop_LQR_thetas.png")


    # Show plots
    if SHOW_PLOTS:
        plt.show()


# === START
if __name__ == "__main__":
    main()
