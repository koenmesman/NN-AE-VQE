#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json
import re
"""
def store(filename, data):
    try:
        with open(filename, "r+") as file:
            filedata = json.load(file)
            for k in data.keys():
                if k in filedata:
                    filedata[k].append(data[k][0])
                else:
                    filedata[k] = [data[k][0]]
            
            file.seek(0)
            json.dump(filedata, file)
            file.truncate()

    except Exception as error:
        print("writing new file for data", error)
        with open(filename, "w") as file:
            for k in data.keys():
                data[k] = [data[k][0]]
            json.dump(data, file)
"""

def store_vqe(filename, data):
    """
    Data is structured as: {<datatype>: {point:double, energy:double, 'parameters':list(double)}}
    with parameters not present for exact diagonalisation results.
    """

    try:
        with open(filename, "r+") as file:
            filedata = json.load(file)


            result_type = list(data.keys())[0]
            if result_type not in filedata.keys():
                for k in data[result_type].keys():
                    data[result_type][k] = [data[result_type][k]]
                filedata[result_type] = data[result_type]

            else:
                for k in data[result_type].keys():
                    if k in filedata[result_type]:
                        filedata[result_type][k].append(data[result_type][k])
                    else: filedata[result_type][k] = data[result_type][k]
            
            file.seek(0)
            json.dump(filedata, file)
            file.truncate()
    except Exception as error:
            print("writing new file for data", error)
            with open(filename, "w") as file:
                result_type = list(data.keys())[0]
                for k in data[result_type].keys():
                    if type(k) is not dict:
                        data[result_type][k] = [data[result_type][k]]
                    else: 
                        data[result_type][k] = data[result_type][k]
                json.dump(data, file)


def store_qae(filename, data):
    """
    Input data is structured as: {<compression>: {accuracy:double,
     'parameters':list(double), 'samples':int 'ansatz':str}}
    """

    try:
        with open(filename, "r+") as file:
            filedata = json.load(file)

            compression = list(data.keys())[0]
            if compression not in filedata.keys():
                filedata[compression] = [data[compression]]

            else:
               filedata[compression].append(data[compression])
            
            file.seek(0)
            json.dump(filedata, file)
            file.truncate()
    except Exception as error:
            print("writing new file for data", error)
            with open(filename, "w") as file:
                compression = list(data.keys())[0]
                data[compression] = [data[compression]]
                json.dump(data, file)


def load(filename):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except:
        raise Exception('File not found')


def rmse(errors):
    return np.sqrt(np.square(errors).mean())

def to_distance(config:str):
    "find 6th number in config and return as float"
    dist = re.findall("[0-9]|\.", config)
    return float("".join(dist[5:]))
