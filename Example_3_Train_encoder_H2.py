#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:57:10 2024

@author: kmesman
"""

import sys
sys.path.append("../src")

from ChemInstance import Instance
from QAE import Encoder
from Ansatz import TwoLocalU3
import time
from qiskit.primitives import Estimator, StatevectorEstimator

ELEMENTS=["Li", "H"]

start = time.time()

inst = Instance(ELEMENTS, method="UCCSD")
inst.est = StatevectorEstimator

enc = Encoder(instance=inst, target=3, ansatz=TwoLocalU3)
enc.test=True
error = enc.train(num_states=6, vqe_reference="../data/Energies_{}_{}.json".format(*ELEMENTS))

print(time.time()-start)
print("achieved accuracy : {}".format(error))