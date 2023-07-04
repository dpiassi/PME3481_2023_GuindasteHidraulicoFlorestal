import control as ct
import matplotlib.pyplot as plt
from helpers import *
from inputs import *


# === MAIN FUNCTION
def main():
    # Evaluate number of states in the system
    nof_states = np.size(A, 0)
    print("Número de estados:", nof_states)

    # Evaluate controlability
    ctrb_matrix = ct.ctrb(A, B)
    print("Matriz de controlabilidade:", bmatrix(ctrb_matrix))

    ctrb_rank = np.linalg.matrix_rank(ctrb_matrix)
    print("Posto da matriz de controlabilidade:", ctrb_rank)

    print("Número de estados igual ao posto da matriz de controlabilidade?")
    if nof_states == ctrb_rank:
        print("Sim, ou seja, o sistema é totalmente controlável.")
    else:
        print("Não, ou seja, o sistema NÃO é totalmente controlável.")

    # Evaluate observability
    obsv_matrix = ct.obsv(A, C)
    print("Matriz de observabilidade:", bmatrix(obsv_matrix))

    obsv_rank = np.linalg.matrix_rank(obsv_matrix)
    print("Posto da matriz de observabilidade:", obsv_rank)

    print("Número de estados igual ao posto da matriz de observabilidade?")
    if nof_states == obsv_rank:
        print("Sim, ou seja, o sistema é totalmente observável.")
    else:
        print("Não, ou seja, o sistema NÃO é totalmente observável.")


# === START
if __name__ == "__main__":
    main()
