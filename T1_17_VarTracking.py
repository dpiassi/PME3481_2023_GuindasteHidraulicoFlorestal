import control as ct
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from inputs import *


# === MAIN FUNCTION
def main():
    # Target reference state-space:
    Ar = np.zeros((6, 6))
    Ar[0, 3] = 1
    Ar[3, 0] = -1 # -omega^2
    print("Ar =", bmatrix(Ar))

    # Disturbance state-space:
    Aw = np.zeros((6, 6))
    Aw[0, 3] = 1
    Aw[3, 0] = -10 # -omega^2
    print("Aw =", bmatrix(Aw))

    # Drop last 3 rows of C:
    C_ = C[0:3, :] # 3x6
    print("C =", bmatrix(C_))

    # Drop last 3 rows of D:
    D_ = D[0:3, :] # 3x3
    print("D =", bmatrix(D_))
    
    # Load regulator's gain matrix:
    K = np.load("K_PP.npy") # 3x6

    # New state matrix with feedback:
    F = A - B @ K # 6x6

    # Build linear system:
    F1 = np.linalg.inv(F) # 6x6
    F2 = np.block([B, A - Ar]) # 6x9
    Ke = np.linalg.inv(C_ @ F1 @ B) @ C_ @ F1 @ F2 # 3x9
    print("Ke =", bmatrix(Ke))

    Ax = np.zeros((6, 1)) # 6x1 # TODO
    Ao = np.block([[Aw, Ax], [Ax, Ar]]) # 12x7
    Ay = np.block([[B, B@K]]) - B @ Ke # 6x9

    a11 = np.block([A-B@K, Ay]) # 6x15
    a21 = np.block([np.zeros((Ao.shape[0], 8)), Ao]) # 12x15
    print("a11 =", a11.shape)
    print("a21 =", a21.shape)
    AT = np.block([[a11], [a21]])
    print("Shape(AT) =", AT.shape)

    # Add 3 empty columns to AT:
    AT = np.block([AT, np.zeros((AT.shape[0], 3))])

    xTo = np.zeros((AT.shape[0], 1))
    xTo[0::2] = 1
    xTo = np.block([xTo, np.zeros((xTo.shape[0], 2))])

    # Build new system:
    sys_Tracking = ct.ss(AT, xTo, AT, xTo)

    # Simulate step response:
    T, yout = ct.step_response(sys_Tracking, TIME)

    # Plot response:
    plt.figure()
    plt.plot(T, yout[0, 0, :])
    plt.suptitle("Seguidor com referência variante e distúrbio")
    save_plot("VarTracking.png")

    # Show plots
    if SHOW_PLOTS:
        plt.show()


# === START
if __name__ == "__main__":
    main()
