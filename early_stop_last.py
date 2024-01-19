import math
import os
import operator
import json
import re
import numpy as np
import random


def extract_answer(generated_answer):
    answer_text = generated_answer.lower().split('the answer is')[-1]
    answer_text = ''.join(re.split(r'[^A-Za-z]', answer_text))
    return answer_text


for model_type in ["GPT3.5","GPT4","Llama2"]:
    task_type = 'last_letters'
    dir_name = "{}_result/{}/".format(model_type, task_type) + 'T0.7.jsonl'

    with open(dir_name, "r") as f:
            data = f.readlines()
            f.close()
    all_dict ={}
    for slice_ in [5]:
        for sc_num in [40]:#32
            random.seed(0)
            if slice_ == 100:
                slice = sc_num
            else:
                slice=slice_
            if slice > sc_num: continue
            print("self consistency num is {}".format(sc_num))
            result_list = []
            num_list = []
            for ii in range(50):
                empty_num = 0
                right_nums = 0
                n_nums = 0
                right_num = 0
                for line in data:
                    tem = json.loads(line)
                    answer = tem['answer']
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
                        if len(pre_dict)==1 and -1 not in pre_dict.keys():
                            break
                        pre_dict = {}
                        for i in predict_list:
                            pre_dict[i] = pre_dict.get(i, 0) + (1 / len(predict_list))
                    n_nums += (index + 1) * slice
                    res = sorted(pre_dict.items(), key=operator.itemgetter(1), reverse=True)
                    if res[0][0] == answer:
                        right_nums += 1
                all_nums = len(data) - empty_num
                result_list.append(100 * right_nums / all_nums)
                num_list.append(n_nums / all_nums)
            all_dict["{}_{}".format(sc_num,slice_)] = [np.array(result_list).mean(),np.array(result_list).var(),np.array(num_list).mean(),np.array(num_list).var()]
            print("mean_acc = {}".format(np.array(result_list).mean()))
            print("var_acc = {}".format(np.array(result_list).var()))
            print("mean_num = {}".format(np.array(num_list).mean()))
            print("var_num = {}".format(np.array(num_list).var()))
    with open("{}_result/{}/".format(model_type, task_type)+"slice_new2.json","w")as f:
        json.dump(all_dict,f)
        f.close()