import control as ct
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from inputs import *


# === MAIN FUNCTION
def main():
    # Load regulator's gain matrix:
    K = np.load("K_PP.npy") # 3x6

    # New state matrix with feedback:
    F = A - B @ K # 6x6

    # Drop last 3 rows of C:
    C_ = C[0:3, :] # 3x6
    print("C =", bmatrix(C_))

    # Drop last 3 rows of D:
    D_ = D[0:3, :] # 3x3
    print("D =", bmatrix(D_))

    # Solve linear system:
    ls_A = np.block([[A, B], [C_, D_]]) # 9x9
    ls_B = np.block([[np.zeros((6, 3))], [np.eye(3)]]) # 9x3
    print("ls_A =", bmatrix(ls_A))
    print("ls_B =", bmatrix(ls_B))
    N_xu = np.linalg.inv(ls_A) @ ls_B # 9x3
    print("N_xu =", bmatrix(N_xu))

    # Extract Nx and Nu:
    Nx = N_xu[0:6, :] # 6x3
    Nu = N_xu[6:9, :] # 3x3
    print("Nx =", bmatrix(Nx))
    print("Nu =", bmatrix(Nu))

    # Build new B matrix:
    Bt = B @ (Nu + K @ Nx) # 3x3
    print("Bt =", bmatrix(Bt))

    # Build new system:
    sys_StepTracking = ct.ss(F, Bt, C, D)

    # === Simulate response for each input:
    y = np.zeros((6, 3, STEPS))
    for i in range(3):
        # Build constant input:
        Xr = np.zeros((3, STEPS))
        Xr[i, :] = 1 # TODO change to whatever const number you want

        # Simulate response to input x_r:
        T, yout = ct.forced_response(sys_StepTracking, TIME, Xr)

        # Store response:
        y[:, i, :] = yout

    # === Plot response:
    plot_response(T, y, [0, 1, 2], "θᵢ [rad]")
    plt.suptitle("Rastreador de sinal constante (degrau unitário)")
    save_plot("StepTracking.png")

    # Show plots
    if SHOW_PLOTS:
        plt.show()


# === START
if __name__ == "__main__":
    main()
