import numpy as np
from helpers import *
from inputs import *


# === MAIN FUNCTION
def main():
    # Evaluate state-transition matrices:
    Phi, Gamma = build_transfer_matrix(A, DELTA_TIME)
    print("Phi =", bmatrix(Phi))
    print("Gamma =", bmatrix(Gamma))


# === START
if __name__ == "__main__":
    main()
