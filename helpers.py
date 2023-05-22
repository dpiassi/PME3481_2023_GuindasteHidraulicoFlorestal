import numpy as np
import array_to_latex as a2l
import matplotlib.pyplot as plt

_FORMAT = '{:.2E}'

def bmatrix(a: np.array) -> str:
    return a2l.to_ltx(a, frmt = _FORMAT, arraytype = 'bmatrix', print_out=False)

def array(a: np.array) -> str:
    return a2l.to_ltx(a, frmt = _FORMAT, arraytype = 'array', print_out=False)

def save_plot(filename: str):
    plt.savefig(filename, dpi=300)