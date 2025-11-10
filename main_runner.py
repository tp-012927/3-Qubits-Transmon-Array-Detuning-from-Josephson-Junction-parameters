# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 08:49:06 2025

@author: 27TerryP
"""

# main_runner.py
import numpy as np
import matplotlib.pyplot as plt
from josephson import JosephsonJunction
from transmon_system import build_transmon_hamiltonian
import analysis_utils as autils
# Physical constants
hbar = 1.0545718e-34  # J·s
e = 1.60217662e-19    # C
h = 2 * np.pi * hbar

#C=capacitance
#R_N = Normal state resistance
# Delta = superconducting energy gap
junction_params = [
    {"C": 10e-15, "R_N": 10e3, "Delta": 200e-6 * e},  
    {"C": 12e-15, "R_N": 10e3, "Delta": 200e-6 * e},  
    {"C": 11e-15, "R_N": 10e3, "Delta": 200e-6 * e},  
]

# Coupling capacitance between qubits! Derived from geometric circuit layout
C_c = 1e-15  # F



# Function to compute Ic from R_N and Delta
def compute_Ic(R_N, Delta):
    return (np.pi * Delta) / (2 * e * R_N)

def EJ_GHz(Ic):
    EJ_J = (hbar * Ic) / (2 * e)
    return EJ_J / h / 1e9  # Convert J → GHz

# Function to compute EC in GHz
def EC_GHz(C):
    EC = e**2 / (2 * C)
    return EC / h * 1e-9  # GHz
EJ_GHz_list = []
EC_GHz_list = []
Ic_list = []

for p in junction_params:
    Ic = compute_Ic(p["R_N"], p["Delta"])
    Ic_list.append(Ic)
    EJ = EJ_GHz(Ic)
    EC = EC_GHz(p["C"])
    EJ_GHz_list.append(EJ)
    EC_GHz_list.append(EC)
    print(f"Ic={Ic:.2e} A, C={p['C']:.2e} F → EJ={EJ:.3f} GHz, EC={EC:.3f} GHz")



# Compute coupling T between qubits (example: nearest neighbor)

T_list = []
for i in range(len(junction_params) - 1):
    C1 = junction_params[i]["C"]
    C2 = junction_params[i + 1]["C"]
    EC1 = EC_GHz_list[i] * h * 1e9  # back to J
    EC2 = EC_GHz_list[i + 1] * h * 1e9
    T = C_c / np.sqrt(C1 * C2) * np.sqrt(EC1 * EC2) / h * 1e-9  # GHz
    T_list.append(T)
    print(f"Coupling T between qubit {i+1} and {i+2} → {T:.6f} GHz")
# Take average EC and coupling
EC_GHz = np.mean(EC_GHz_list)
T_GHz = np.mean(T_list)
# Build Hamiltonian
H = build_transmon_hamiltonian(EJ_list=EJ_GHz_list, EC=EC_GHz, T=T_GHz, N=3, cutoff=10)


# Diagonalize
GHz_to_rad = 2 * np.pi
eigenenergies = H.eigenenergies()
eigenenergies_GHz = eigenenergies / GHz_to_rad

# Print first few energies
print("\nEigenenergies (GHz):")
for idx, energy in enumerate(eigenenergies_GHz[:10]):
    print(f"State {idx}: {energy:.6f} GHz")


# --- Analysis Section ---
# Approximate transition frequencies
w01_approx = autils.approx_w01_gHz(EJ_GHz_list, EC_GHz_list)
print("\nApproximate ω01 (GHz) per qubit:", w01_approx)

# -- Deturning --
detunings = []
pairs = []
N = len(w01_approx)
for i in range(N):
    for j in range(i+1, N):
        d = abs(w01_approx[i] - w01_approx[j])
        detunings.append(d)
        pairs.append(f"{i}-{j}")
        print(f"Qubit {i}-{j} detuning: {d:.4f} GHz")


# Compute IPR
eigs_vals, eigs_states = H.eigenstates()
iprs = autils.compute_ipr(eigs_states)
print("\nIPR of first 10 eigenstates:", iprs[:10])


plt.bar(pairs, detunings, color="skyblue")
plt.ylabel("Detuning (GHz)")
plt.title("Qubit Pair Detunings")

for idx, det in enumerate(detunings):
    plt.text(idx, det + 0.01, f"{det:.3f}", ha='center', va='bottom')
plt.show()