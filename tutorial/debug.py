import __init__
from Utils import load, store_aevqe

import copy

data = load("../data/aevqe_data_HH_1000.json")
print(len(data["4_3"]["rxry_cx_circ-2"][0]))


new_data = copy.copy(data)
new_data["4_3"]["rxry_cx_circ-2"] = data["4_3"]["rxry_cx_circ-2"][0]
print(len(new_data["4_3"]["rxry_cx_circ-2-grad"])) 

store_aevqe("../data/aevqe_data_HH_1000_fix.json", new_data)