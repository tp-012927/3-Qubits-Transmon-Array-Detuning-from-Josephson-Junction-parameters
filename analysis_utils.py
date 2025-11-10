# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 07:58:58 2025

@author: 27TerryP
"""

# analysis_utils.py
import numpy as np
from itertools import combinations
from qutip import basis, tensor, qeye, sesolve, expect

GHz_to_rad = 2 * np.pi

# --------------------------
# 1) approximate transition freq ω01 (GHz)
# --------------------------
def approx_w01_gHz(EJ_GHz, EC_GHz):
    """
    Approximate transmon ω01 in GHz:
        ω01 ≈ sqrt(8 * EC * EJ) - EC
    Inputs:
      EJ_GHz, EC_GHz : floats (can be arrays) in GHz
    Returns:
      ω01 in GHz (same shape)
    Notes:
      This returns frequency in GHz (not angular frequency).
    """
    EJ = np.array(EJ_GHz)
    EC = np.array(EC_GHz)
    # guard against negative/zero EJ or EC:
    safe = np.maximum(0.0, 8.0 * EC * EJ)
    w01 = np.sqrt(safe) - EC
    return w01

# --------------------------
# 2) find resonant pairs
# --------------------------
def find_resonant_pairs(freqs_GHz, threshold_GHz=0.1):
    """
    Identify pairs (i,j) whose |f_i - f_j| < threshold.
    Returns list of tuples: [(i,j,delta), ...]
    """
    pairs = []
    for i, j in combinations(range(len(freqs_GHz)), 2):
        delta = abs(freqs_GHz[i] - freqs_GHz[j])
        if delta < threshold_GHz:
            pairs.append((i, j, delta))
    return pairs

# --------------------------
# 3) random EJ arrays (disorder)
# --------------------------
def random_EJ_array(N, mean_EJ_GHz, std_EJ_GHz, seed=None):
    """
    Generate random EJ values (Gaussian) for N qubits.
    Ensures EJ > 0 by truncation at small positive value.
    """
    rng = np.random.default_rng(seed)
    arr = rng.normal(loc=mean_EJ_GHz, scale=std_EJ_GHz, size=N)
    arr = np.maximum(arr, 1e-6)  # avoid nonpositive EJ
    return arr

# --------------------------
# 4) inverse participation ratio (IPR)
# --------------------------
def compute_ipr(eigenstates):
    """
    Compute IPR for a sequence of eigenstates (QuTiP Qobj states).
    eigenstates: list of Qobj kets returned by H.eigenstates() or eigenstates list.
    Returns array of IPR values (float).
    IPR = sum_i |c_i|^4 where c_i are components in chosen basis.
    """
    iprs = []
    for psi in eigenstates:
        vec = psi.full().ravel()        # complex amplitudes in computational basis
        probs = np.abs(vec)**2
        ipr = np.sum(probs**2)
        iprs.append(np.real_if_close(ipr))
    return np.array(iprs)

# --------------------------
# 5) helper to build local projector |1><1| for qubit i (for time evolution)
# --------------------------
def local_projector_one(N, cutoff, i):
    """
    Return operator that projects qubit i onto Fock state |1><1|.
    Uses tensor product embedding.
    """
    from qutip import projection
    p1 = projection(cutoff, 1, 1)  # |1><1| on single mode
    ops = [qeye(cutoff) for _ in range(N)]
    ops[i] = p1
    return tensor(ops)

# --------------------------
# 6) simple time-evolution example (optional)
# --------------------------
def time_evolve_populations(H, psi0, N, cutoff, tlist):
    """
    Evolve initial state psi0 with Hamiltonian H and return populations P_i(t)
    where P_i is expectation of projector onto |1> of qubit i.
    Returns times tlist and populations array shape (len(tlist), N)
    """
    # build projectors
    projectors = [local_projector_one(N, cutoff, i) for i in range(N)]
    # use sesolve (Schrodinger) to compute expectations
    result = sesolve(H, psi0, tlist, e_ops=projectors)
    # result.expect is list of arrays, one per e_op: shape (n_ops, len(tlist))
    pops = np.vstack(result.expect).T   # shape (len(tlist), N)
    return tlist, pops
