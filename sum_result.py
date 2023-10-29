import importlib
from collections import defaultdict
import numpy as np

module_name_list = [
    "sam_med3d",
    # "sam_med2d_ft2d"
]

for module_idx, module_name in enumerate(module_name_list):
    try:
        module = importlib.import_module("results."+module_name)
    except:
        raise ValueError("file not found", module_name)
    dice_Ts = module.dice_Ts
    final_results = defaultdict(list)
    for k, v in dice_Ts.items():
        k = k.split("/")
        # print(k)
        cls, dataset, data_type, case = k[-4], k[-3], k[-2], k[-1]
        # print(cls, dataset, data_type, case)
        final_results[cls].append(v*100)
    print(f"-----[{module_name}]-------")
    print("cls\t|  dice")
    for k, v in final_results.items():
        print( k, "\t|", "{:.2f}%".format(float(np.mean(v))))
    print("----------------------------")

    