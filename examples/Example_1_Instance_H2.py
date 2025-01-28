#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:53:56 2024

@author: kmesman
"""
import sys
import os
file_path = os.path.abspath(os.path.dirname(__file__)).replace('\\', '/')
lib_path = os.path.abspath(os.path.join(file_path, '../src')).replace('\\', '/')
sys.path.append(lib_path)

from ChemInstance import Instance

inst = Instance(["H", "H"], ansatz="exact")
test = inst.run(1)

print("exact diagonailization : {} Hartree".format(test))

inst = Instance(["H", "H"], ansatz="UCCSD")
test = inst.run(1)

print("VQE UCCSD : {} Hartree".format(test))

inst = Instance(["H", "H"], ansatz="SU2")
test = inst.run(1)

print("VQE SU2 : {} Hartree".format(test))
