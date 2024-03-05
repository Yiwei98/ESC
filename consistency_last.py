import math
import os
import operator
import json
import re
from utils import delete_extra_zero, _strip_string
# from math_equivalence import is_equiv
import statistics
import numpy as np
import random
from tqdm import trange

random.seed(0)

def extract_answer(generated_answer):
    answer_text = generated_answer.lower().split('the answer is')[-1]
    answer_text = ''.join(re.split(r'[^A-Za-z]', answer_text))
    return answer_text


def entropy(Plist):
    if len(Plist):
        result = 0
        for x in Plist.values():
            result += (-x) * math.log(x, 2)
        return result
    else:
        return 0


model_type = "GPT3.5"
task_type = 'last_letters'
sc_num = 5
print("self consistency num is {}".format(sc_num))
result_list = []
p_list = []
for ii in range(1):
    empty_num = 0
    right_nums = 0
    wrong_nums = 0
    right_entropy = 0
    wrong_entropy = 0
    dir_name = "{}_result/{}/".format(model_type, task_type) + 'T0.7.jsonl'
    entropy_name = "{}_result/{}/".format(model_type, task_type) + 'probs{}.json'.format(sc_num)
    with open(dir_name, "r") as f:
        data = f.readlines()
        f.close()
    fe = open(entropy_name, "w")
    right_num = 0
    for line in data:
        tem = json.loads(line)
        answer = tem['answer']
        cur_batch = random.sample(tem['generated_answer'], sc_num)
        predict_list = []
        pre_dict = {}
        for seed in range(sc_num):
            tem_1 = cur_batch[seed]
            predict1 = extract_answer(tem_1)
            # if predict1 != -1:
            #     predict_list.append(predict1)
            predict_list.append(predict1)
            # if predict1 in pre_dict.keys():
            #     pre_dict[predict1] += 1
            # else:
            #     pre_dict[predict1] = 1
        # print(predict_list)
        for i in predict_list:
            pre_dict[i] = pre_dict.get(i, 0) + (1 / len(predict_list))
        res = sorted(pre_dict.items(), key=operator.itemgetter(1), reverse=True)
        if res[0][0] == answer:
            right_nums += 1
            right_entropy += entropy(pre_dict)
            p_list.append([t[1] for t in res])
        else:
            wrong_nums += 1
            wrong_entropy += entropy(pre_dict)
            p_list.append([t[1] for t in res])
    all_nums = len(data)
    print("overall_acc = {}".format(100 * right_nums / all_nums))
    print("right entropy = {}".format(right_entropy / right_nums))
    print("wrong entropy = {}".format(wrong_entropy / wrong_nums))
    result_list.append(100 * right_nums / all_nums)

print("mean_acc = {}".format(np.array(result_list).mean()))
print("var_acc = {}".format(np.array(result_list).var()))
json.dump(p_list, fe)
