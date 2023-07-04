import control as ct
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from inputs import *

# === CONSTANTS
SHOW_PLOTS = False
_DURATION = 300
_STEPS = 1000
_TIME = np.linspace(0, _DURATION, _STEPS)


# === MAIN FUNCTION
def main():
    # Create the state-space model
    sys_OpenLoop = ct.ss(A, B, C, D)

    #  Get the open loop transfer function
    tf_OpenLoop = ct.ss2tf(sys_OpenLoop)

    # Print transfer function of "input 1 to output 2"
    tf_OpenLoop_1_2 = tf_OpenLoop[1, 0]
    print("Input 1 to output 2:")
    print(tf_OpenLoop_1_2)    

    # Get system poles
    poles = np.linalg.eig(A)[0]
    print("Poles:", bmatrix(poles))
    
    # Plot a pole/zero map for a linear system.
    plt.figure()
    ct.pzmap(sys_OpenLoop, plot=True, grid=True, title="Pólos em malha aberta")
    save_plot("OpenLoop_pzmap.png")

    # Simulate step response
    T, yout = ct.step_response(sys_OpenLoop, _TIME, X0)

    # Plot step response
    plot_response(T, yout, [0, 1, 2], "θᵢ [rad]")
    plt.suptitle("Resposta ao degrau (malha aberta)")
    save_plot("OpenLoop_step_thetas.png")

    plot_response(T, yout, [3, 4, 5], "ωᵢ [rad/s]")
    plt.suptitle("Resposta ao degrau (malha aberta)")
    save_plot("OpenLoop_step_omegas.png")

    # Show plots
    if SHOW_PLOTS:
        plt.show()


# === START
if __name__ == "__main__":
    main()
