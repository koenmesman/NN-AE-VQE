#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:45:08 2024

@author: kmesman
"""
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from numpy import pi

def TwoLocalUniversal(n, repetitions=1):
    num_params = (n * (n - 1))*repetitions
    theta = ParameterVector("gamma", length=num_params)
    thetait = iter(theta)
    qc = QuantumCircuit(n)
    for r in range(repetitions):
        for i in range(0, n):
            for j in range(i + 1, n):
                qc.cx(i, j)
                qc.ry(next(thetait), i)
                qc.rz(next(thetait), j)
                qc.cx(j, i)
                qc.t(i)
                qc.sdg(j)
    return qc, theta


def TwoLocalU3(n, layers=1):
    num_params = 3 * (n + (n**2) * (layers))
    theta = ParameterVector("gamma", length=num_params)
    thetait = iter(theta)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.u(next(thetait), next(thetait), next(thetait), q)
    for c in range(n):
        # control qubit
        for t in range(n):
            # target qubit
            if c != t:
                qc.cx(c, t)
                qc.u(next(thetait), next(thetait), next(thetait), t)
                qc.cx(c, t)
    for q in range(n):
        qc.u(next(thetait), next(thetait), next(thetait), q)
    for k in range(layers-1):
        for c in range(n):
            # control qubit
            for t in range(n):
                # target qubit
                if c != t:
                    qc.cx(c, t)
                    qc.u(next(thetait), next(thetait), next(thetait), t)
                    qc.cx(c, t)
        for q in range(n):
            qc.u(next(thetait), next(thetait), next(thetait), q)
    return qc, theta


def SWAP_test(n, m):
    circ_size = 2 * (n - m) + 1
    aux = circ_size - 1
    qc = QuantumCircuit(circ_size, 1)
    qc.h(aux)
    for i in range(n - m):
        qc.cswap(aux, i, (n - m) + i)
    qc.h(aux)
    return qc

def StronglyEntangled(qubits, layers=1):
    """
    Quantum model based on Schuld et al.'s Circuit-centric quantum classifiers.
    also in "Supervised learning with quantum-enhanced feature spaces" (2019)
    """

    param_len = qubits * 6
    theta = ParameterVector("theta", length=param_len)
    qc = QuantumCircuit(qubits)
    for li in range(layers):
        for qubit, t, p, l in zip(
            range(qubits),
            theta[:qubits],
            theta[qubits : 2 * qubits],
            theta[2 * qubits : 3 * qubits],
        ):
            qc.u(t, p, l, qubit)
    
        for qubit in range(qubits - 1):
            qc.cx(qubit, qubit + 1)
        qc.cx(qubits - 1, 0)
    
        for qubit, t, p, l in zip(
            range(qubits),
            theta[3 * qubits : 4 * qubits],
            theta[4 * qubits : 5 * qubits],
            theta[5 * qubits : 6 * qubits],
        ):
            qc.u(t, p, l, qubit)
    
        for qubit in range(qubits - 2):
            qc.cx(qubit, qubit + 2)
    
        for qubit in reversed(range(2, qubits)):
            qc.cx(qubit, qubit - 2)

    return qc, theta


def EntangledXYZ(qubits, repetitions=1):
    """
    Quantum model based on Schuld et al.'s Circuit-centric quantum classifiers.
    Modified for XYZ rotation gates.
    """

    param_len = qubits * 6
    theta = ParameterVector("theta", length=param_len)
    qc = QuantumCircuit(qubits)
    for qubit, t, p, l in zip(
        range(qubits),
        theta[:qubits],
        theta[qubits : 2 * qubits],
        theta[2 * qubits : 3 * qubits],
    ):
        qc.rx(t, qubit)
        qc.ry(p, qubit)
        qc.rz(l, qubit)

    for qubit in range(qubits - 1):
        qc.cx(qubit, qubit + 1)
    qc.cx(qubits - 1, 0)

    for qubit, t, p, l in zip(
        range(qubits),
        theta[3 * qubits : 4 * qubits],
        theta[4 * qubits : 5 * qubits],
        theta[5 * qubits : 6 * qubits],
    ):
        qc.rx(t, qubit)
        qc.ry(p, qubit)
        qc.rz(l, qubit)

    for qubit in range(qubits - 2):
        qc.cx(qubit, qubit + 2)

    for qubit in reversed(range(2, qubits)):
        qc.cx(qubit, qubit - 2)

    return qc, theta


def EntangledU2(qubits, repetitions=1):
    """
    Quantum model based on Schuld et al.'s Circuit-centric quantum classifiers.
    Modified to only include U2 gates as opposed to U3.
    """

    param_len = qubits * 4
    theta = ParameterVector("theta", length=param_len)
    qc = QuantumCircuit(qubits)
    for qubit, p, l in zip(range(qubits), theta[:qubits], theta[qubits : 2 * qubits]):
        qc.u(pi, p, l, qubit)

    for qubit in range(qubits - 1):
        qc.cx(qubit, qubit + 1)
    qc.cx(qubits - 1, 0)

    for qubit, p, l in zip(
        range(qubits), theta[2 * qubits : 3 * qubits], theta[3 * qubits : 4 * qubits]
    ):
        qc.u(pi/2, p, l, qubit)

    for qubit in range(qubits - 2):
        qc.cx(qubit, qubit + 2)

    for qubit in reversed(range(2, qubits)):
        qc.cx(qubit, qubit - 2)

    return qc, theta
