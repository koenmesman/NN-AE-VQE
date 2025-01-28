#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../src")

from ChemInstance import Instance
from QAE import Encoder
from Ansatz import TwoLocalU3

ELEMENTS=["Li", "H"]

inst = Instance(ELEMENTS, method="UCCSD")
enc = Encoder(instance=inst, target=3, ansatz=TwoLocalU3)
error = enc.train(num_states=6, atom=ELEMENTS)

print("achieved accuracy : {}".format(error))
