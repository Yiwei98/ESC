import json
import math
import scipy.stats as stats
import os

model_types = ["GPT4","GPT3.5","Llama2"]
task_types = ['gsm8k','last_letters','common','strategy','coin','MATH']

def cal_n(windows,max_up,p):
    sumup=0
    quan=0
    for t in p:
        quan += pow(t,windows)
    for i in range(max_up//windows-1):
        sumup+=quan*pow(1-quan,i)*(1+i)
    sumup += pow(1 - quan, max_up//windows-1) * (max_up//windows)
    sumup*=windows
    return sumup


sc_num = 5
max_up = 40
for model_type in model_types:
    for task_type in task_types:
        dir = "{}_result/{}/probs{}.json".format(model_type,task_type,sc_num)
        if not os.path.exists(dir):
            continue
        with open(dir, "r") as f:
            l = json.load(f)
            f.close()

        sj_sumup = 0
        avg_all, var_all, count_all,p_all = 0, 0, 0, 0

        for tem in l:
            count_all+=1
            sj_sumup += cal_n(sc_num,max_up,tem)
            for i in range(1,len(tem)):
                p_all+=1-stats.norm.cdf((sc_num-tem[i]*sc_num)/math.sqrt(tem[i]*sc_num*(1-tem[i])))
                avg_all+=tem[i]*sc_num
                var_all+=tem[i]*sc_num*(1-tem[i])
        avg_all/=count_all
        var_all/=count_all
        z = (sc_num-avg_all)/math.sqrt(var_all)
        p_all /= count_all
        cdf_value = stats.norm.cdf(z)
        # cdf_value_true = stats.norm.cdf(z_all)
        pp = "{:e}".format(1-cdf_value)
        p_all_="{:e}".format(p_all)
        Exp_num = sj_sumup/count_all
        print("Maxnum-{} window size-{} {}-{}  The proportion of performance affected:{}    Expected sampling times:{}".format(max_up,sc_num,model_type,task_type,p_all_,Exp_num))
