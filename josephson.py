# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 08:01:57 2025

@author: 27TerryP
"""

import numpy as np

# Physical constants
hbar = 1.054571817e-34    # Reduced Planck constant (J*s)
e = 1.602176634e-19       # Elementary charge (C)
h = 2 * np.pi * hbar      # Planck constant (J*s)


class JosephsonJunction:
    def __init__(self, Ic, C):
        """
        Parameters
        ----------
        Ic : float
            Critical current in Amperes
        C : float
            Junction capacitance in Farads
        """
        self.Ic = Ic
        self.C = C

    def EJ_J(self):
        """Josephson energy EJ in Joules."""
        return hbar * self.Ic / (2 * e)

    def EC_J(self):
        """Charging energy EC in Joules."""
        return (e)**2 / (2 * self.C)

    def EJ_GHz(self):
        """Josephson energy EJ in GHz."""
        return self.EJ_J() / h / 1e9

    def EJ_GHz(self):
        EJ_J = (hbar * self.Ic) / (2 * e)
        return EJ_J / h / 1e9
