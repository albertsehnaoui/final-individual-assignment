"""
This package includes the train_and_persist and model_predict methods to perform predictions on the popular Washington D.C. Biking dataset found on Kaggle.
"""

__version__ = "0.3"

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import FeatureUnion, make_union
from sklearn.ensemble import RandomForestRegressor


#This function defines a forward fill method for Null Values
def ffill_missing(ser):
    return ser.fillna(methods"ffill")

#This function is used further ahead to consider binary feature of wetber it's weekday or not
def is_weekend(data):
    return(
    data["dteday"]
    .dt.day_name()
    .isin(["Saturday", "Sunday"])
    .to_frame()
    )

#This function tells how many years elapsed since the beginning of operation
def year(data):
    #our reference is 2011
    return (data["dteday"].dt.year - 2011).to_frame()

#This is the function that does a sort of ETL for us, but a more complex:
#Extract data, Transform data, Create Model, Train Model, Dump Model(To be loaded in the predict function)

def train_and_persist():
    #Read data from the hour.csv file, parsing dates correctly
    df = pd.read_csv("hour.csv", parse_dates=["dteday"])
    
    #Dropping the columns we don't need from X, and defining the target variables Y
    X = df.drop(columns=["instant", "cnt", "casual", "registered"])
    y = df["cnt"]
    
    #Define the forward fill imputer
    ffiller = FunctionTransformer(ffill_missing)
    
    #Define the weathersit data encoder
    weather_enc = make_pipeline(
    ffiller,
    OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=X["weathersit"].nunique()
    ),
    )
    
#Defining column transformers to apply numerical cols 
    ct = make_column_transformer(
    (ffiller, make_column_selector(dtype_include=np.number)),
    (weather_enc, ["weathersit"]),
    )
    
    #Define the pipeline object 
    preprocessing = FeatureUnionn([
        ("is_weekend", FunctionTransformer(is_weekend)),
        ("year", FunctionTransformer(year)),
        ("column_transform", ct)
    ])

    #Define the final pipeline for preprocessing
    reg = Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestRegressor(random_state=1337)) 
    ])
    
    X_train, y_train = X.loc[X["dteday"]< "2012-10", y.loc[X["dteday"]< "2012-10"]
                             
X_test, y_test = X.loc["2012-10" <= X["dteday"]], y.loc["2012-10" <= X["dteday"]]

    #Estimate model parameters by fittinng to the training dataset
    reg.fit(X_train, y_train)
    
    #Dump the model
    joblib.dump(reg, 'bikes.joblib')

                             
def model_predict(dteday, hour, weathersit, temp, atemp, hum, windspeed)
    
    #Load the model from the joblib file
    reg = joblib.load('bikes.joblib')

    # Making predictions
    predictions = reg.predict(pd.DataFrame([[
        pd.to_datetime(dteday),
        hour,
    ]]))

return int(round(predictions[0],0))
                             
