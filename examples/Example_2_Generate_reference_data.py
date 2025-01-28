#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:41:40 2024

@author: kmesman
"""

import sys
import os
file_path = os.path.abspath(os.path.dirname(__file__)).replace('\\', '/')
lib_path = os.path.abspath(os.path.join(file_path, '../src')).replace('\\', '/')
sys.path.append(lib_path)

import numpy as np

from ChemInstance import Instance
from Utils import store


if __name__ == "__main__":
    NUM_STATES = 6
    ELEMENTS = ["H", "H"]
    TEST_POINTS = list(np.arange(0.2, 3.2, 2.8 / (NUM_STATES-1)))
    SAVE_FILE = "../data/Energies_{}_{}.json".format(*ELEMENTS)
    
    
    def energies_and_parameters(method):
        inst = Instance(ELEMENTS, method=method)
        r = []
        for p in TEST_POINTS:
            res = inst.run(p)
            
            save_data = {method+"_points":p,
            method:res,
            }
            r.append(res)
            if method != "exact":
                save_data[method+"_parameters"]=list(inst.parameters)
            store(SAVE_FILE, save_data)
        return r
    
    
    vqe = energies_and_parameters("UCCSD")
    #energies_and_parameters("SU2")
    ex = energies_and_parameters("exact")
