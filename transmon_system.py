# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 08:49:37 2025

@author: 27TerryP
"""

# transmon_system.py
import numpy as np
from qutip import tensor, qeye, destroy, Qobj

def cos_phi_operator(dim):
    """Cos(phi) operator in truncated Fock basis."""
    data = np.zeros((dim, dim))
    'matrix of the size of the limit'
    for i in range(dim - 1):
        data[i, i + 1] = 0.5
        data[i + 1, i] = 0.5
    'created the matrix of 0.5, 0.5 thingy of cosphi'
    return Qobj(data)

def embed_operator(op, i, N, cutoff):
    """Embed single-qubit operator into N-qubit Hilbert space."""
    ops = [qeye(cutoff) for _ in range(N)]
    ops[i] = op
    return tensor(ops)
    "multiply the operators in case there are multiple"

def build_transmon_hamiltonian(EJ_list, EC, T, N=3, cutoff=10):
    """
    Build the N-transmon Hamiltonian.
    
    Parameters
    ----------
    EJ_list : array-like
        Josephson energies [GHz] for each qubit
    EC : float
        Charging energy [GHz] (common for all)
    T : float
        Coupling energy [GHz]
    N : int
        Number of transmons
    cutoff : int
        Number of levels per transmon
    """
    GHz_to_rad = 2 * np.pi
    EC_rad = EC * GHz_to_rad
    EJ_rad_list = np.array(EJ_list) * GHz_to_rad
    T_rad = T * GHz_to_rad
    "set it to angular frequency"

    n_op = destroy(cutoff).dag() * destroy(cutoff)
    'This is the charge operator approximation, Number operator counts excitation through up op and down op multiplication'
    cos_phi = cos_phi_operator(cutoff)
    'matrix with cutoff'
    # Initialize Hamiltonian
    H = 0;

    # On-site terms Literally just hamiltonian
    for i in range(N):
        H += 4 * EC_rad * embed_operator(n_op ** 2, i, N, cutoff)
        H -= EJ_rad_list[i] * embed_operator(cos_phi, i, N, cutoff)

    # Coupling terms hamiltonian
    for i in range(N - 1):
        H += T_rad * embed_operator(n_op, i, N, cutoff) * embed_operator(n_op, i + 1, N, cutoff)

    return H
