#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json
import re

from typing import Dict, List, Any, Mapping, MutableMapping
import random
from typing import List, Sequence, Iterable, Any


Nested = Dict[str, Dict[str, Dict[str, List[Any]]]]

def _normalize_ab(v: Any) -> List[Any]:
    # a, b must be list of scalars
    if v is None:
        return []
    return v if isinstance(v, list) else [v]

def _normalize_c(v: Any) -> List[List[Any]]:
    # c must be list of lists
    if v is None:
        return []
    if isinstance(v, list):
        if not v:
            return []
        # If elements are lists already, keep as-is; otherwise wrap the flat list once
        if all(isinstance(x, list) for x in v):
            return v
        return [v]
    # scalar -> wrap twice
    return [[v]]

def _copy_shallow_list(lst: List[Any]) -> List[Any]:
    # copy list (for a,b) to avoid aliasing
    return list(lst)

def _copy_nested_list(lst: List[List[Any]]) -> List[List[Any]]:
    # copy list of lists (for c) to avoid aliasing
    return [list(inner) for inner in lst]

def merge_flat(dst: MutableMapping[str, Dict[str, Any]],
               src: Mapping[str, Dict[str, Any]]) -> None:
    """
    Merge `src` into `dst` in place.

    Expected structure:
      { name: { 'points': [..], 'energy': [..], 'parameters': [[..], ...] } }

    - If name missing, copy from src.
    - If name exists, append:
        points/energy -> extend with floats
        parameters    -> extend with each inner list
    """
    for name, payload in src.items():
        dst_payload = dst.setdefault(name, {})

        if "points" in payload:
            pts = payload["points"]
            if "points" in dst_payload:
                dst_payload["points"].extend(pts)
            else:
                dst_payload["points"] = _copy_shallow_list(pts)

        if "energy" in payload:
            ene = payload["energy"]
            if "energy" in dst_payload:
                dst_payload["energy"].extend(ene)
            else:
                dst_payload["energy"] = _copy_shallow_list(ene)

        if "parameters" in payload:
            pars = payload["parameters"]
            if "parameters" in dst_payload:
                dst_payload["parameters"].extend(_copy_nested_list(pars))
            else:
                dst_payload["parameters"] = _copy_nested_list(pars)


def merge_nested(dst: Nested, src: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    """
    Merge `src` into `dst` in place.

    Structure: 
      { key1: { key2: { 'a': [..], 'b': [..], 'c': [[..], ...] } } }

    - Creates missing key1/key2.
    - Ensures 'a' and 'b' are lists; 'c' is a list of lists.
    - Appends normalized data:
        * a/b: extends with items
        * c:   extends with inner lists (groups)
    """
    for k1, level1 in src.items():
        dst_level1 = dst.setdefault(k1, {})
        for k2, payload in level1.items():
            dst_payload = dst_level1.setdefault(k2, {})

            # Normalize incoming values (if present)
            if "points" in payload:
                na = _normalize_ab(payload["points"])
                if "points" in dst_payload:
                    dst_payload["points"].extend(na)
                else:
                    dst_payload["points"] = _copy_shallow_list(na)

            if "energy" in payload:
                nb = _normalize_ab(payload["energy"])
                if "energy" in dst_payload:
                    dst_payload["energy"].extend(nb)
                else:
                    dst_payload["energy"] = _copy_shallow_list(nb)

            if "parameters" in payload:
                nc = _normalize_c(payload["parameters"])
                if "parameters" in dst_payload:
                    dst_payload["parameters"].extend(_copy_nested_list(nc))
                else:
                    dst_payload["parameters"] = _copy_nested_list(nc)
    return dst_payload



def store_vqe(filename, data):
    """
    Data is structured as: {<datatype>: {point:double, energy:double, 'parameters':list(double)}}
    with parameters not present for exact diagonalisation results.
    """

    try:
        with open(filename, "r+") as file:
            filedata = json.load(file)
            merge_flat
            
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

    Simple implementation as we assume new data is always singular.
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


def store_aevqe(filename, data):
    """
    Data (multiple) is structured as: {<compression>:
     {<ansatz_name>: {point:list(float), energy:list(float), 'parameters':list(list(float))}}}
    """
    try:
        with open(filename, "r+") as file:
            filedata = json.load(file)
            merge_nested(dst=filedata, src=data)
            
            file.seek(0)
            json.dump(filedata, file)
            file.truncate()
    except Exception as error:
            print("writing new file for data", error)
            with open(filename, "w") as file:
                result_type = list(data.keys())[0]
                for k in data[result_type].keys():
                        data[result_type][k] = data[result_type][k]
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

def sample_data(
    num_points: int,
    *datasets: Sequence[Sequence[Any]],
    seed: int | None = None,
) -> List[List[List[Any]]]:
    """
    Sample the same 'num_points' IDs (from the intersection of all datasets' first rows)
    and return the sampled view for each dataset.

    Each dataset is a list-of-lists:
      - dataset[0] is the list of IDs/keys (must be hashable & ideally unique)
      - dataset[1:], each is a row aligned with dataset[0] by index

    Args:
        num_points: number of IDs to sample (clipped to size of the intersection).
        *datasets: any number of datasets in the structure described above.
        seed: optional RNG seed for reproducibility.

    Returns:
        A list with one sampled dataset per input dataset.
        For each sampled dataset:
          - [0] is the sampled list of IDs (same order across all outputs)
          - [1:] are the sampled rows aligned to those IDs
    """
    if not datasets:
        raise ValueError("Provide at least one dataset.")


    # Basic validation & compute intersection of keys across all datasets
    key_sets = []
    for i, ds in enumerate(datasets, start=1):
        if not ds or not isinstance(ds[0], Sequence):
            raise ValueError(f"Dataset #{i} must have a header row at index 0.")
        key_sets.append(set(ds[0]))

        # Optional sanity check: all rows same length as header
        header_len = len(ds[0])
        for r, row in enumerate(ds[1:], start=1):
            if len(row) != header_len:
                raise ValueError(
                    f"Dataset #{i} row {r} length {len(row)} != header length {header_len}"
                )

    common_keys = set.intersection(*key_sets)
    if not common_keys:
        raise ValueError("No common IDs across the provided datasets.")

    # Choose sample of IDs
    rng = random.Random(seed)
    common_list = list(common_keys)
    if num_points > len(common_list):
        num_points = len(common_list)
    sampled_ids = rng.sample(common_list, num_points)

    # Build sampled view for each dataset using index maps for speed
    sampled_all: List[List[List[Any]]] = []
    for ds in datasets:
        header = ds[0]
        index_of = {k: i for i, k in enumerate(header)}
        # If duplicates exist in header, last occurrence wins in the dict above.
        sampled_ds = [sampled_ids]
        for row in ds[1:]:
            sampled_row = [row[index_of[k]] for k in sampled_ids]
            sampled_ds.append(sampled_row)
        sampled_all.append(sampled_ds)

    return sampled_all
