import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from tqdm import tqdm_notebook
from itertools import product
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

def optimize_SARIMA(series, parameters_list, s):
    """
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, d, q, P, D, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
    """
    
    best_aic = float('inf')
    
    
    for param in tqdm_notebook(parameters_list):
        try: model = sm.tsa.statespace.SARIMAX(series, order=(param[0], param[1], param[2]),
                                               seasonal_order=(param[3], param[4], param[5], s)).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        
        #Save best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
    
    return best_param, best_aic, best_model

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():

    name_list = [] # all names
    data = []

    ar_pd = {} # name-72预测点 dict
    #MAE = []

    with open("../data/dataset_campus_competition.txt","r") as f:
        Lines = f.readlines()
        for line in Lines:
            ecs = line.split()
            name_list.append(ecs[0])
            val = eval(ecs[1])
            val = ecs[1].split(',')
            data.extend(val)

    data = np.array(data)
    data = data.reshape(len(name_list),-1)
    idx = np.where(data == 'NA')
    data[idx] = np.nan

    ecs_NA = set(idx[0]) # 7*24时间点存在CPU数据NAN的ECS实例标号
    #ecs_full = set(range(0,120)) - ecs_NA
    #ecs_NA = list(ecs_NA)
    #ecs_full = list(ecs_full)
    #ecs_NA.sort()
    #ecs_full.sort()

    data = data.astype('float')
    #full = data[ecs_full]
    #na = data[ecs_NA]

    # 设置模型超参数
    ps = qs = range(0,3)
    Ps = Qs = range(0, 2)
    d = D = [1]
    s = 24

    for i in range(data.shape[0]):
        
        if i in ecs_NA:
             d = D = [0]
        parameters = product(ps, d, qs, Ps, D, Qs)
        parameters_list = list(parameters)

        ecs = pd.Series(data[i])
        _, _, res_model = optimize_SARIMA(ecs, parameters_list, s)

        '''
        tik = 5*24
        pred = res_model.get_prediction(start=tik,  dynamic=True, full_results=True)
        y_forecasted = pred.predicted_mean
        y_truth = ecs0[5*24:]
        mae = mean_absolute_percentage_error(y_truth, y_forecasted)
        MAE.append(mae)
        '''
        pred_72 = res_model.get_forecast(steps=3*24)
        res_pd = np.array(pred_72.predicted_mean)
        res_pd = np.around(res_pd,decimals=2)
        res_pd[np.where(res_pd<=0)] = 0.0
        ar_pd[name_list[i]] = res_pd
    
    with open("../data/test-output.txt","w") as file:
        for i in range(data.shape[0]):
            ecs_name = name_list[i]
            ecs = ar_pd[ecs_name]       
            ecs = ecs.astype('str')
            ecs = str(name_list[i]) + " " + "\"" + ",".join(ecs) + "\"" +"\n"
            #print(ecs)
            file.write(ecs)

if __name__ == "__main__":
    main()
