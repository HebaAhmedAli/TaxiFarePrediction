import pandas as pd # CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import calendar
import constants
import utils

def encodeDays(dayOfWeek):
    dayDict={'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6}
    return dayDict[dayOfWeek]

def cleanData(data):
    # Drop nulls.
    # print(data.isnull().sum())
    data = data.dropna(how = 'any', axis = 'rows')
    
    # Create datetime features based on pickup_datetime
    data['pickup_datetime']=pd.to_datetime(data['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')
    data['pickup_day']=data['pickup_datetime'].apply(lambda x:x.day)
    data['pickup_hour']=data['pickup_datetime'].apply(lambda x:x.hour)
    data['pickup_day_of_week']=data['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
    data['pickup_month']=data['pickup_datetime'].apply(lambda x:x.month)
    data['pickup_year']=data['pickup_datetime'].apply(lambda x:x.year)
    
    data['pickup_day_of_week']=data['pickup_day_of_week'].apply(lambda x:encodeDays(x))

    boundary={'min_lng':-74.263242,
              'min_lat':40.573143,
              'max_lng':-72.986532, 
              'max_lat':41.709555}
    
    if 'fare_amount' in data.columns:
        data=data[data['fare_amount']>=0]
        data.loc[~((data.pickup_longitude >= boundary['min_lng'] ) & (data.pickup_longitude <= boundary['max_lng']) &
            (data.pickup_latitude >= boundary['min_lat']) & (data.pickup_latitude <= boundary['max_lat']) &
            (data.dropoff_longitude >= boundary['min_lng']) & (data.dropoff_longitude <= boundary['max_lng']) &
            (data.dropoff_latitude >=boundary['min_lat']) & (data.dropoff_latitude <= boundary['max_lat'])),'is_outlier_loc']=1
        data.loc[((data.pickup_longitude >= boundary['min_lng'] ) & (data.pickup_longitude <= boundary['max_lng']) &
            (data.pickup_latitude >= boundary['min_lat']) & (data.pickup_latitude <= boundary['max_lat']) &
            (data.dropoff_longitude >= boundary['min_lng']) & (data.dropoff_longitude <= boundary['max_lng']) &
            (data.dropoff_latitude >=boundary['min_lat']) & (data.dropoff_latitude <= boundary['max_lat'])),'is_outlier_loc']=0

        # Let us drop rows, where location is outlier
        data=data.loc[data['is_outlier_loc']==0]
        data.drop(['is_outlier_loc'],axis=1,inplace=True)
    
    data=data[data['passenger_count']<=8]
    
    return data

def readAndCleanData(trainPath,testPath):
    train =  pd.read_csv(trainPath, nrows=3000000)
    #print(train.dtypes)
    test = pd.read_csv(testPath)
    #print(test.dtypes)
    train = cleanData(train)
    test = cleanData(test)
    return train,test
    

def prepareDataForModel(data,target,dropCols,isTrain=True,split=0.25):
    dataPrepared=data.drop(dropCols,axis=1)
    # One Hot Encoding of categorical variables.
    # dataPrepared=pd.get_dummies(dataPrepared) # TODO: Try uncomment.

    if isTrain==True:
        X=dataPrepared.drop([target],axis=1)
        y=dataPrepared[target]
        # Dividing training data into train and validation data sets.
        xTrain, xTest, yTrain, yTest = train_test_split(X,y, test_size=split,random_state=123)
        
        print("Shape of Training Features",xTrain.shape)
        print("Shape of Validation Features ",xTest.shape)
        
        return xTrain, xTest, yTrain, yTest
    else:
        print ("Shape of Test Data",dataPrepared.shape)
        return dataPrepared

def addFeatureEngineering(data):
    # Add features of weather the pickup or dropoff isAirport.
    data['is_pickup_la_guardia']=data.apply(lambda row:utils.isAirport(row['pickup_latitude'],row['pickup_longitude'],'LaGuardia'),axis=1)
    data['is_dropoff_la_guardia']=data.apply(lambda row:utils.isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'LaGuardia'),axis=1)
    data['is_pickup_EWR']=data.apply(lambda row:utils.isAirport(row['pickup_latitude'],row['pickup_longitude'],'EWR'),axis=1)
    data['is_dropoff_EWR']=data.apply(lambda row:utils.isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'EWR'),axis=1)
    data['is_pickup_JFK']=data.apply(lambda row:utils.isAirport(row['pickup_latitude'],row['pickup_longitude'],'JFK'),axis=1)
    data['is_dropoff_JFK']=data.apply(lambda row:utils.isAirport(row['dropoff_latitude'],row['dropoff_longitude'],'JFK'),axis=1)
    
    # Add features of weather the pickup or dropoff is Borough or other
    data['pickup_borough']=data.apply(lambda row:utils.getBorough(row['pickup_latitude'],row['pickup_longitude']),axis=1)
    data['dropoff_borough']=data.apply(lambda row:utils.getBorough(row['dropoff_latitude'],row['dropoff_longitude']),axis=1)
    
    # data=pd.get_dummies(data)
    
    # Add is_lower_manhattan feature.
    data['is_pickup_lower_manhattan']=data.apply(lambda row:utils.isLowerManhattan(row['pickup_latitude'],row['pickup_longitude']),axis=1)
    data['is_dropoff_lower_manhattan']=data.apply(lambda row:utils.isLowerManhattan(row['dropoff_latitude'],row['dropoff_longitude']),axis=1)
    
    # Add utils.distance of pickup and dropoff to airports.
    data['distance_jfk']=data.apply(lambda row:utils.distance(row['pickup_latitude'],row['pickup_longitude'],constants.jfk[1],constants.jfk[0]),axis=1)
    data['distance_jfk']=data.apply(lambda row:utils.distance(row['dropoff_latitude'],row['dropoff_longitude'],constants.jfk[1],constants.jfk[0]),axis=1)
    data['distance_ewr']=data.apply(lambda row:utils.distance(row['pickup_latitude'],row['pickup_longitude'],constants.ewr[1],constants.ewr[0]),axis=1)
    data['distance_ewr']=data.apply(lambda row:utils.distance(row['dropoff_latitude'],row['dropoff_longitude'],constants.ewr[1],constants.ewr[0]),axis=1)
    data['distance_laguardia']=data.apply(lambda row:utils.distance(row['pickup_latitude'],row['pickup_longitude'],constants.lgr[1],constants.lgr[0]),axis=1)
    data['distance_laguardia']=data.apply(lambda row:utils.distance(row['dropoff_latitude'],row['dropoff_longitude'],constants.lgr[1],constants.lgr[0]),axis=1)

    # Add utils.distance of pickup and dropoff to boroughs.
    data['distance_manhattan']=data.apply(lambda row:utils.distance(row['pickup_latitude'],row['pickup_longitude'],constants.manhattan[1],constants.manhattan[0]),axis=1)
    data['distance_queens']=data.apply(lambda row:utils.distance(row['pickup_latitude'],row['pickup_longitude'],constants.queens[1],constants.queens[0]),axis=1)
    data['distance_brooklyn']=data.apply(lambda row:utils.distance(row['pickup_latitude'],row['pickup_longitude'],constants.brooklyn[1],constants.brooklyn[0]),axis=1)
    data['distance_bronx']=data.apply(lambda row:utils.distance(row['pickup_latitude'],row['pickup_longitude'],constants.bronx[1],constants.bronx[0]),axis=1)
    data['distance_statenisland']=data.apply(lambda row:utils.distance(row['pickup_latitude'],row['pickup_longitude'],constants.staten_island[1],constants.staten_island[0]),axis=1)

    data['distance_manhattan']=data.apply(lambda row:utils.distance(row['dropoff_latitude'],row['dropoff_longitude'],constants.manhattan[1],constants.manhattan[0]),axis=1)
    data['distance_queens']=data.apply(lambda row:utils.distance(row['dropoff_latitude'],row['dropoff_longitude'],constants.queens[1],constants.queens[0]),axis=1)
    data['distance_brooklyn']=data.apply(lambda row:utils.distance(row['dropoff_latitude'],row['dropoff_longitude'],constants.brooklyn[1],constants.brooklyn[0]),axis=1)
    data['distance_bronx']=data.apply(lambda row:utils.distance(row['dropoff_latitude'],row['dropoff_longitude'],constants.bronx[1],constants.bronx[0]),axis=1)
    data['distance_statenisland']=data.apply(lambda row:utils.distance(row['dropoff_latitude'],row['dropoff_longitude'],constants.staten_island[1],constants.staten_island[0]),axis=1)


    return data
