#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:45:08 2024

@author: kmesman
"""
from qiskit.circuit import ParameterVector
from qiskit.circuit import Gate, QuantumCircuit

class SwapTestGate(Gate):
    """
    Custom SWAP test gate.
    Qubits: [data_qubits..., aux_qubit]
    """
    def __init__(self, n, m, label=None):
        """
        Args:
            n: Total size of the original register before removing m qubits.
            m: Number of qubits removed (used for offset).
            label: Optional label for the gate.
        """
        self.n = n
        self.m = m
        self.circ_size = 2 * (n - m) + 1
        super().__init__("SwapTest", num_qubits=self.circ_size, params=[], label=label)

    def _define(self):
        """Define the SWAP test structure."""
        aux = self.circ_size - 1
        qc = QuantumCircuit(self.circ_size)
        qc.h(aux)
        for i in range(self.n - self.m):
            qc.cswap(aux, i, (self.n - self.m) + i)
        qc.h(aux)
        self.definition = qc

class UniversalTwoQubit(Gate):
    """
    A custom gate that applies:
        CX(i, j)
        RY(theta0) on i
        RZ(theta1) on j
        CX(j, i)
        T(i)
        Sdg(j)
    """

    def __init__(self, thetas=None, label=None):
        """
        Args:
            thetas: List of parameters [theta0, theta1]. If None, creates parameters automatically.
            label: Optional label for the gate.
        """
        if thetas is None:
            thetas = ParameterVector("theta", 2)
        super().__init__("Custom2Q", num_qubits=2, params=thetas, label=label)

    @property
    def parameter_bounds(self):
        return [(0, 2 * pi), (0, 2 * pi)]

    def _define(self):
        """
        Defines the internal decomposition of the custom gate.
        """
        theta_iter = iter(self.params)
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.ry(next(theta_iter), 0)
        qc.rz(next(theta_iter), 1)
        qc.cx(1, 0)
        qc.t(0)
        qc.sdg(1)

        self.definition = qc

class U3TwoQubit(Gate):
    """
    A custom gate that applies:
        CX(i, j)
        U3(theta[0-2], j)
        CX(i, j)

    """

    def __init__(self, thetas=None, label=None):
        """
        Args:
            thetas: List of parameters [theta0, theta1]. If None, creates parameters automatically.
            label: Optional label for the gate.
        """
        if thetas is None:
            thetas = ParameterVector("theta", 3)
        super().__init__("Custom2Q", num_qubits=2, params=thetas, label=label)

    @property
    def parameter_bounds(self):
        return [(0, 2 * pi), (0, 2 * pi), (0, 2 * pi)]

    def _define(self):
        """
        Defines the internal decomposition of the custom gate.
        """
        theta_iter = iter(self.params)
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.u(next(theta_iter), next(theta_iter), next(theta_iter), 1)
        qc.cx(1, 0)

        self.definition = qc