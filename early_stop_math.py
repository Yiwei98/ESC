
import operator
import json
import re
import numpy as np
from utils import delete_extra_zero, _strip_string
import random
from tqdm import trange


def extract_math_answer(pred_str):
    if ('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif ('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if (len(ans) and ans[0] == '{'):
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


def find_math_answer(s):
    assert ('boxed' in s)
    # s = s.replace(",", "")
    ans = s.split('boxed')[-1]
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
    return a

for model_type in ["Llama2","GPT4","GPT3.5"]:
    task_type = 'MATH'
    dir_name = "{}_result/{}/".format(model_type, task_type) + 'T0.5.jsonl'
    with open(dir_name, "r") as f:
        data = f.readlines()
        f.close()
    all_dict ={}
    for slice_ in [5,100]:
        for sc_num in [8,16,24,32,40,48,56,64]:
            random.seed(0)
            if slice_ == 100:
                slice = sc_num
            else:
                slice=slice_
            if slice > sc_num: continue
            result_list = []
            num_list = []
            for ii in range(50):
                right_nums = 0
                n_nums = 0
                dir_name = "{}_result/{}/".format(model_type, task_type) + 'T0.5.jsonl'
                if model_type == 'GPT3' and task_type == 'MATH':
                    with open(dir_name, "r") as f:
                        data = json.load(f)
                        f.close()
                else:
                    with open(dir_name, "r") as f:
                        data = f.readlines()
                        f.close()
                for line in data:
                    tem = json.loads(line)
                    answer = find_math_answer(tem['answer'])
                    shu_batch = tem['generated_answer']
                    # shu_batch = tem['generated_answer'][g * 64: g * 64 + 64]
                    random.shuffle(shu_batch)
                    predict_list = []
                    for index in range(sc_num // slice):
                        answer = find_math_answer(tem['answer'])
                        cur_batch = shu_batch[index * slice: (index + 1) * slice]
                        pre_dict = {}
                        for seed in range(slice):
                            tem_1 = cur_batch[seed]
                            try:
                                predict1 = extract_math_answer(tem_1)
                            except:
                                predict1 = ''
                            predict_list.append(predict1)
                        num = len(predict_list)
                        for i in predict_list[index * slice:(index + 1) * slice]:
                            pre_dict[i] = pre_dict.get(i, 0) + (1 / slice)
                        if len(pre_dict) == 1:
                            break
                        pre_dict = {}
                        for i in predict_list:
                            pre_dict[i] = pre_dict.get(i, 0) + (1 / len(predict_list))
                    n_nums += (index + 1) * slice
                    res = sorted(pre_dict.items(), key=operator.itemgetter(1), reverse=True)
                    if res[0][0] == answer:
                        right_nums += 1
                all_nums = len(data)
                result_list.append(100 * right_nums / all_nums)
                num_list.append(n_nums / all_nums)
            all_dict["{}_{}".format(sc_num, slice_)] = [np.array(result_list).mean(), np.array(result_list).var(),
                                                        np.array(num_list).mean(), np.array(num_list).var()]
            print("mean_acc = {}".format(np.array(result_list).mean()))
            print("var_acc = {}".format(np.array(result_list).var()))
            print("mean_num = {}".format(np.array(num_list).mean()))
            print("var_num = {}".format(np.array(num_list).var()))

    with open("{}_result/{}/".format(model_type, task_type)+"slice_new2.json","w")as f:
        json.dump(all_dict,f)
        f.close()