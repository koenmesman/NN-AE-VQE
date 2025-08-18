import __init__
from Utils import load, store_aevqe

import copy

data = load("../data/exact_data.json")
print(len(data["exact"]["points"]))


new_data = copy.copy(data)
data["exact"]["points"] = data["exact"]["points"][0]
data["exact"]["energy"] = data["exact"]["energy"][0]

print(len(data["exact"]["points"]))

store_aevqe("../data/exact_data_fix.json", new_data)