import control as ct
import matplotlib.pyplot as plt
from helpers import *
from inputs import *


# === MAIN FUNCTION
def main():
    # Initialize open loop system
    sys_OpenLoop = ct.ss(A, B, C, D)

    # Simulate step response
    T, yout = ct.step_response(sys_OpenLoop, TIME, X0)

    # Plot step response
    plot_response(T, yout, [0, 1, 2], "θᵢ [rad]")
    plt.suptitle("Resposta ao degrau (malha aberta)")
    save_plot("OpenLoop_step_thetas.png")

    plot_response(T, yout, [3, 4, 5], "ωᵢ [rad/s]")
    plt.suptitle("Resposta ao degrau (malha aberta)")
    save_plot("OpenLoop_step_omegas.png")

    # Simulate impulse response
    T, yout = ct.impulse_response(sys_OpenLoop, TIME, X0)

    # Plot impulse response
    plot_response(T, yout, [0, 1, 2], "θᵢ [rad]")
    plt.suptitle("Resposta ao impulso (malha aberta)")
    save_plot("OpenLoop_impulse_thetas.png")

    plot_response(T, yout, [3, 4, 5], "ωᵢ [rad/s]")
    plt.suptitle("Resposta ao impulso (malha aberta)")
    save_plot("OpenLoop_impulse_omegas.png")

    #  Get the open loop transfer function
    tf_OpenLoop = ct.ss2tf(sys_OpenLoop)

    # Iterate over all transfer functions in tf_OpenLoop:
    for i in range(3):
        for j in range(3):
            print("Input", i + 1, "to output", j + 1, ":")
            print(tf_OpenLoop[j, i])
            sys = ct.tf2ss(tf_OpenLoop[j, i])

            # Plot bode diagram (simulation in frequency domain)
            plt.figure()
            ct.bode_plot(sys, dB=True, omega_limits=OMEGA_LIMITS, omega_num=OMEGA_NUM)
            plt.suptitle("Diagrama de Bode (malha aberta): entrada " + U_LABELS[i] + " para saída " + Y_LABELS[j])
            save_plot(f"OpenLoop_bode_{i}_{j}.png")

    # Show plots
    if SHOW_PLOTS:
        plt.show()


# === START
if __name__ == "__main__":
    main()
