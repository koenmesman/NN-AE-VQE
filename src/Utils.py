#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json


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


def load(filename):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except:
        raise Exception('File not found')


def rmse(errors):
    return np.sqrt(np.square(errors).mean())
