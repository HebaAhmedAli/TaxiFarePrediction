import dataPreparation
import lightGBM

train,test = dataPreparation.readAndCleanData('data/tryTrain.csv','data/tryTest.csv')
print("Shape of Training Data after cleaning ",train.shape)
print("Shape of Testing Data after cleaning", test.shape)

# TODO: Call function for dataAnalysis here.

# For ( heba & fatema but sobhy & feryal will use train & test above only)
# For training and testing using models.
xTrain, xTest, yTrain, yTest \
=dataPreparation.prepareDataForModel(train,'fare_amount',dropCols=['key','pickup_datetime'],isTrain=True,split=0.2)


testData=dataPreparation.prepareDataForModel(test,'fare_amount',dropCols=['key','pickup_datetime'],isTrain=False)

# Add feature engineering to the data.
xTrain=dataPreparation.addFeatureEngineering(xTrain)
xTest=dataPreparation.addFeatureEngineering(xTest)
testData=dataPreparation.addFeatureEngineering(testData)

xTrain.to_csv("data/sample_X_train_cleaned.csv",index=False)
xTest.to_csv("data/sample_X_test_cleaned.csv",index=False)
testData.to_csv("data/sample_test_cleaned.csv",index=False)
print(xTrain.dtypes)

# TODO: Call model training and testing here.
#lgbBst=lightGBM.getTrainedLgbModel(xTrain,yTrain)
#lightGBM.predictAndEvaluateModel(lgbBst,xTest,yTest,xTrain,yTrain)

# The best tuned paramters -we tune the paramters in colab notebook and get final results here-
best = {'x_max_depth': 9.0,
        'x_subsample': 0.9920618722172806,
        'x_colsample': 0.8288999698751736,
        'x_learning_rate': 0.06585147300487698,
        'x_num_leaves': 50,
        'x_subsample_freq': 100,
        'x_n_estimators': 1000
        }
lgbBst=lightGBM.getTrainedLgbModelAfterTuning(best,xTrain,yTrain,xTest,yTest)
lightGBM.predictAndEvaluateModel(lgbBst,xTest,yTest,xTrain,yTrain)
