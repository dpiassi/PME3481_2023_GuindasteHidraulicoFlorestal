import numpy as np
from scipy.linalg import expm
import array_to_latex as a2l
import matplotlib.pyplot as plt
import os
from inputs import *

_FORMAT = '{:.2E}'
_PLOT_OUTPUT_DIR = 'plots/'

def bmatrix(a: np.array) -> str:
    return a2l.to_ltx(a, frmt = _FORMAT, arraytype = 'bmatrix', print_out=False)


def array(a: np.array) -> str:
    return a2l.to_ltx(a, frmt = _FORMAT, arraytype = 'array', print_out=False)


def save_plot(filename: str):
    if not os.path.exists(_PLOT_OUTPUT_DIR):
        os.makedirs(_PLOT_OUTPUT_DIR)
    path = _PLOT_OUTPUT_DIR + filename
    plt.savefig(path, dpi=300)


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


def eval_input(yout, K):
    y = np.zeros((yout.shape[2], yout.shape[1]))
    for i in range(yout.shape[2]):
        y[i] = K @ yout[:, 0, i]
    return y


def plot_input(T, yout, K, columns: list[int], ylabel: str):
    u_count = len(U_LABELS)
    fig, axs = plt.subplots(u_count, sharex=True)
    y = eval_input(yout, K)

    for i in range(u_count):
        for column in columns:
            axs[i].plot(T, y[:, column], label=U_LABELS[column])
        axs[i].set_title(Y_LABELS[i])
        axs[i].axhline(linewidth=0.5, color='gray')
        axs[i].axvline(linewidth=0.5, color='gray')
        box = axs[i].get_position()
        axs[i].set_position([box.x0, box.y0, box.width * 0.82, box.height])

    for ax in axs.flat:
        ax.set(ylabel=ylabel, yscale="log")

    # Position the legend outside of the subplot area
    labels = [U_LABELS[column] for column in columns]
    plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', shadow=True)
    plt.xlabel('t [s]')


def build_transfer_matrix(A, dt):
    Phi = expm(A * dt)
    Gamma = np.linalg.inv(A) @ (Phi - np.eye(A.shape[0]))
    return Phi, Gamma


def solve_transfer_matrix(Phi, Gamma, B, x0, t):
    x = np.zeros((Phi.shape[0], B.shape[1], t.shape[0]))
    for input_ in range(x.shape[1]):
        x[:, input_, 0] = x0
    for step in range(1, t.shape[0]):
        x[:, :, step] = Phi @ x[:, :, step - 1] + Gamma @ B
    return x