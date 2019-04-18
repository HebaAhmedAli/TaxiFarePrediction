import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

def getTrainedLgbModel(xTrain,yTrain):
    trainData=lgb.Dataset(xTrain,label=yTrain)
    param = {'num_leaves':31, 'num_trees':5000, 'objective':'regression'}
    param['metric'] = 'l2_root'
    num_round=5000
    cv_results = lgb.cv(param, trainData, num_boost_round=num_round, nfold=10,verbose_eval=20, early_stopping_rounds=20,stratified=False)

    print('Best num_boost_round:', len(cv_results['rmse-mean']))

    lgb_bst=lgb.train(param,trainData,len(cv_results['rmse-mean']))
    return lgb_bst

def predictAndEvaluateModel(lgb_bst,xTest,yTest,xTrain,yTrain):
    lgb_pred = lgb_bst.predict(xTest)
    lgb_rmse=np.sqrt(mean_squared_error(lgb_pred, yTest))
    print("RMSE for Light GBM is ",lgb_rmse)

    lgb_train_rmse=np.sqrt(mean_squared_error(lgb_bst.predict(xTrain),yTrain))
    print("Train RMSE for Light GBM is ", lgb_train_rmse)

    variance=lgb_train_rmse - lgb_rmse
    print("Variance of Light GBM is ", variance)