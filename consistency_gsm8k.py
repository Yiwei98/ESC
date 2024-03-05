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

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def extract_math_answer(pred_str):
    if ('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif ('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred = a

    else:
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if (len(pred) >= 1):
            # print(pred_str)
            pred = pred[-1]
        else:
            pred = ''
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    pred = _strip_string(pred)
    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred = a
    return pred


def entropy(Plist):
    if len(Plist):
        result = 0
        for x in Plist.values():
            result += (-x) * math.log(x, 2)
        return result
    else:
        return 0


model_type = "GPT3.5"
task_type = 'gsm8k'
sc_num = 5
print("self consistency num is {}".format(sc_num))
result_list = []
num_list = []
p_list = []
for ii in range(1):
    empty_num = 0
    right_nums = 0
    wrong_nums = 0
    right_entropy = 0
    wrong_entropy = 0
    dir_name = "{}_result/{}/".format(model_type, task_type) + 'T0.7.jsonl'
    entropy_name = "{}_result/{}/".format(model_type, task_type) + 'probs{}.json'.format(sc_num)
    if model_type == 'Llama2':
        with open(dir_name, "r") as f:
            data = json.load(f)
            f.close()
    else:
        with open(dir_name, "r") as f:
            data = f.readlines()
            f.close()
    fe = open(entropy_name, "w")
    for line in data:
        tem = json.loads(line)
        answer = extract_answer(tem['answer'])
        cur_batch = random.sample(tem['generated_answer'], sc_num)
        predict_list = []
        pre_dict = {}
        for seed in range(sc_num):
            tem_1 = cur_batch[seed]
            number_list = re.findall(r"\d+\.?\d*", tem_1)
            try:
                predict1 = number_list[-1].strip('.')
            except:
                predict1 = -1000
                # continue
            predict_list.append(predict1)
        num = len(predict_list)
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
