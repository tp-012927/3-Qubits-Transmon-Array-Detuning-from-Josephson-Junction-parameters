# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 08:58:06 2025

@author: 27TerryP
"""
import numpy as np
import matplotlib.pyplot as plt
from transmon_system import build_transmon_hamiltonian
import analysis_utils as autils

# -------------------------
# Physical constants
# -------------------------
hbar = 1.0545718e-34  # J·s
e = 1.60217662e-19    # C
h = 2 * np.pi * hbar

# -------------------------
# Base junction parameters from IBM Chip? 
# -------------------------
base_junction_params = [
    {"C": 10e-15, "R_N": 10e3, "Delta": 200e-6 * e},  
    {"C": 10e-15, "R_N": 10e3, "Delta": 200e-6 * e},  
    {"C": 10e-15, "R_N": 10e3, "Delta": 200e-6 * e},  
]

# Coupling capacitance (geometric)
C_c = 1e-15  # F

# -------------------------
# Core calculation functions
# -------------------------
def compute_Ic(R_N, Delta):
    """Compute Josephson critical current."""
    return (np.pi * Delta) / (2 * e * R_N)

def EJ_GHz(Ic):
    """Convert Josephson energy to GHz."""
    EJ_J = (hbar * Ic) / (2 * e)
    return EJ_J / h / 1e9

def EC_GHz(C):
    """Convert charging energy to GHz."""
    EC = e**2 / (2 * C)
    return EC / h * 1e-9

def compute_detunings(junction_params):
    """Compute detuning between all qubit pairs for given parameters."""
    EJ_list = [EJ_GHz(compute_Ic(p["R_N"], p["Delta"])) for p in junction_params]
    EC_list = [EC_GHz(p["C"]) for p in junction_params]

    # Approximate ω01 transition frequencies
    w01 = autils.approx_w01_gHz(EJ_list, EC_list)

    # Compute detuning for all qubit pairs
    pairs = []
    detunings = []
    N = len(w01)
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append(f"{i}-{j}")
            detunings.append(abs(w01[i] - w01[j]))
    return pairs, detunings

# -------------------------
# Sweep functionality
# -------------------------
def sweep_parameter(param_name, param_values, qubit_idx=0):
    """
    Sweep a single qubit parameter and return all pairwise detunings.
    """
    all_detunings = []  # will hold detuning arrays for each pair
    pairs_ref = None

    for val in param_values:
        params = [dict(p) for p in base_junction_params]
        params[qubit_idx][param_name] = val
        pairs, detunings = compute_detunings(params)

        if pairs_ref is None:
            pairs_ref = pairs
            for _ in pairs:
                all_detunings.append([])

        # record each pair's detuning at this sweep point
        for i, d in enumerate(detunings):
            all_detunings[i].append(d)

    return pairs_ref, np.array(all_detunings)


# -------------------------
# Visualization utility
# -------------------------
def plot_sweep_multi(x_values, pairs, detuning_matrix, xlabel, ylabel, title):
    plt.figure()
    for i, pair in enumerate(pairs):
        plt.plot(x_values, detuning_matrix[i], label=f"Pair {pair}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Sweep capacitance of qubit 0
    C_values = np.linspace(5e-15, 20e-15, 500)
    pairs, det_matrix_C = sweep_parameter("C", C_values, qubit_idx=0)
    plot_sweep_multi(C_values*1e15, pairs, det_matrix_C,
                     "Qubit 0 Capacitance C (fF)",
                     "Detuning (GHz)",
                     "Effect of Qubit 0 Capacitance on Pairwise Detuning")
    # Sweep delta of qubit 0
    Delta_values = np.linspace(100e-6*e, 250e-6*e, 500)
    pairs, det_matrix_Delta = sweep_parameter("Delta", Delta_values, qubit_idx=0)
    plot_sweep_multi(Delta_values/e*1e6, pairs, det_matrix_Delta,
                     "Qubit 0 Delta (superconducting energy gap)",
                     "Detuning (GHz)",
                     "Effect of Qubit 0 Delta on Pairwise Detuning")
    # Sweep Rn of qubit 0
    R_N_values = np.linspace(5e3, 25e3, 500)  # example: sweep from 5 kΩ to 25 kΩ
    pairs, det_matrix_R_N = sweep_parameter("R_N", R_N_values, qubit_idx=0)
    plot_sweep_multi(R_N_values / 1e3, pairs, det_matrix_R_N,
                     "Qubit 0 Normal Resistance Rn (kΩ)",
                     "Detuning (GHz)",
                     "Effect of Qubit 0 Rn on Pairwise Detuning")
