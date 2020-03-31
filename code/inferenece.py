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

def predict(ecs):
    '''
        Get history data ecs
        Return predict data as str
    '''
    ecs = ecs.split(',')
    ecs = np.array(ecs)
    idx = np.where(ecs=="NA")
    ecs[idx] = np.nan
    flag = idx[0].size # nan exists or not
    
    ecs = ecs.astype('float')
    ps = qs = range(0,3)
    Ps = Qs = range(0, 2)
    d = D = [1]
    s = 24

    if flag:
        d = D = [0]
    parameters = product(ps, d, qs, Ps, D, Qs)
    parameters_list = list(parameters)

    _, _, res_model = optimize_SARIMA(ecs, parameters_list, s)
    pred_72 = res_model.get_forecast(steps=3*24)
    res_pd = np.array(pred_72.predicted_mean)
    res_pd = np.around(res_pd,decimals=2)
    res_pd[np.where(res_pd<=0)] = 0.0

    resp = res_pd.astype('str')
    resp = "\"" + ",".join(resp) + "\""
    return resp

from flask import request, jsonify, Flask
app = Flask(__name__)

@app.route('/predict', methods = ["POST"])
def post_data():
    #json data as follows
    #{"resource_id":"ecs0","history_data":"1010.09,..." }
    
    
    data = request.get_json(force=True)                # 获取 JSON 数据
    ecs_name = data["resource_id"]   # 获取参数并转变为 DataFrame 结构
    ecs = eval(data["history_data"])
    
    res = {}
    res[ecs_name] = predict(ecs)
    
    response_body = jsonify(res)

    return response_body

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')