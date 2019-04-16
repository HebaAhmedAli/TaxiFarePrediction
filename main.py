import dataPreparation

train,test = dataPreparation.readAndCleanData('data/tryTrain.csv','data/tryTest.csv')
print("Shape of Training Data after cleaning ",train.shape)
print("Shape of Testing Data after cleaning", test.shape)

# TODO: Call function for dataAnalysis here.

# For ( heba & fatema but sobhy & feryal will use train & test above only)
# For training and testing using models.
xTrain, xTest, yTrain, yTest \
=dataPreparation.prepareDataForModel(train,'fare_amount',dropCols=['key','pickup_datetime'],isTrain=True,split=0.2)

testData=dataPreparation.prepareDataForModel(test,'fare_amount',dropCols=['key','pickup_datetime'],isTrain=False)

# TODO: Call model training and testing here.