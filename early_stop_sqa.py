import math
import os
import operator
import json
import re

import numpy as np

from utils import delete_extra_zero, _strip_string
# from math_equivalence import is_equiv
import statistics
import random
from tqdm import trange

def find_answer(sample):
    if sample['answer']['Yes']:
        return 1
    else:
        return 0


def extract_answer(generated_answer):
    if 'the answer is yes' in generated_answer.lower():
        return 1
    elif 'the answer is no' in generated_answer.lower():
        return 0
    else:
        return -1


for model_type in ["Llama2","GPT4","GPT3.5"]:
    task_type = 'strategy'
    dir_name = "{}_result/{}/".format(model_type, task_type) + 'T0.7.jsonl'
    with open(dir_name, "r") as f:
        data = f.readlines()
        f.close()
    all_dict ={}
    for slice_ in [2,3,4,5,6,7,8,9,10]:
        for sc_num in [5,10,15,20,25,30,35,40]:
            random.seed(0)
            if slice_ == 100:
                slice = sc_num
            else:
                slice=slice_
            if slice > sc_num: continue
            result_list = []
            num_list = []
            for ii in range(50):
                empty_num = 0
                right_nums = 0
                n_nums = 0
                dir_name = "{}_result/{}/".format(model_type, task_type) + 'T0.7.jsonl'
                with open(dir_name, "r") as f:
                    data = f.readlines()
                    f.close()
                right_num = 0
                for line in data:
                    tem = json.loads(line)
                    answer = find_answer(tem)
                    shu_batch = tem['generated_answer']
                    random.shuffle(shu_batch)
                    predict_list = []
                    for index in range(sc_num // slice):
                        pre_dict = {}
                        cur_batch = shu_batch[index * slice: (index + 1) * slice]
                        for seed in range(slice):
                            tem_1 = cur_batch[seed]
                            predict1 = extract_answer(tem_1)
                            predict_list.append(predict1)
                        for i in predict_list[index * slice:(index + 1) * slice]:
                                pre_dict[i] = pre_dict.get(i, 0) + (1 / slice)
                        if len(pre_dict) == 1:
                            break
                        pre_dict = {}
                        for i in predict_list:
                                pre_dict[i] = pre_dict.get(i, 0) + (1 / len(predict_list))
                    n_nums += (index + 1) * slice
                    res = sorted(pre_dict.items(), key=operator.itemgetter(1), reverse=True)

                    if len(res) and res[0][0] == answer:
                        right_nums += 1
                all_nums = len(data) - empty_num
                result_list.append(100 * right_nums / all_nums)
                num_list.append(n_nums / all_nums)
            all_dict["{}_{}".format(sc_num, slice_)] = [np.array(result_list).mean(), np.array(result_list).var(),
                                                        np.array(num_list).mean(), np.array(num_list).var()]
            # print("{}-{}  {}-{}: acc:{} L:{}".format(task_type,model_type,slice_,sc_num,np.array(result_list).mean(),np.array(num_list).mean()))
            print("mean_acc = {}".format(np.array(result_list).mean()))
            print("var_acc = {}".format(np.array(result_list).var()))
            print("mean_num = {}".format(np.array(num_list).mean()))
            print("var_num = {}".format(np.array(num_list).var()))
    with open("{}_result/{}/".format(model_type, task_type)+"slice_new_last.json","w")as f:
        json.dump(all_dict,f)
        f.close()