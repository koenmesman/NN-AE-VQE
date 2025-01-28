#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:45:08 2024

@author: kmesman
"""
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit


def TwoLocalUniversal(n, repetitions=1):
    num_params = n * (n - 1)
    theta = ParameterVector("gamma", length=num_params)
    thetait = iter(theta)
    qc = QuantumCircuit(n)
    
    for i in range(0, n):
        for j in range(i + 1, n):
            qc.cx(i, j)
            qc.ry(next(thetait), i)
            qc.rz(next(thetait), j)
            qc.cx(j, i)
            qc.t(i)
            qc.sdg(j)
    return qc, theta


def TwoLocalU3(n, repetitions=1):
    num_params = 3 * (n + (n**2) * (repetitions))
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
    for k in range(repetitions-1):
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
