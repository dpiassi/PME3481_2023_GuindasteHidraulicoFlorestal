import control as ct
import numpy as np
import matplotlib.pyplot as plt
from helpers import *


# === CONSTANTS
DURATION = 10
DELTA_TIME = 0.05
TIME = np.arange(0, DURATION, DELTA_TIME)


# === INPUTS
A = np.array([[-3, 1, 0], [2, -3, 2], [0, 1, -3]])
B = np.ones((3,1))
C = np.eye(3)
D = np.zeros((3,1))
X0 = 100 * np.ones(3)


# === MAIN FUNCTION
def main():
    # Evaluate state-transition matrices:
    Phi, Gamma = build_transfer_matrix(A, DELTA_TIME)

    # Simulate response
    X = solve_transfer_matrix(Phi, Gamma, B, X0, TIME)

    # Plot step response
    plt.plot(TIME, X[0, 0, :], 'r:', linewidth=2)
    plt.plot(TIME, X[1, 0, :], 'g--', linewidth=2)
    plt.plot(TIME, X[2, 0, :], 'b-.', linewidth=2)
    plt.legend(['Tanque 1', 'Tanque 2', 'Tanque 3'])
    plt.xlabel("Tempo (s)")
    plt.ylabel("Nível dos tanques")
    plt.title("Simulação em malha aberta")
    plt.show()


# === START
if __name__ == "__main__":
    main()
