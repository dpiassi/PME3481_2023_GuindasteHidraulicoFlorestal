import numpy as np
import array_to_latex as a2l
import matplotlib.pyplot as plt
import os

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