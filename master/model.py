"""
This package includes the train_and_persist and model_predict methods to perform predictions on the popular Washington D.C. Biking dataset found on Kaggle.
"""

__version__ = "0.3"

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer,make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import FeatureUnion, make_union
from sklearn.ensemble import RandomForestRegressor


# import the dataset
df = pd.read_csv("hour.csv", parse_dates=["dteday"])
df.head()

df.info()

# Dropping the columns we don't need from X

X = df.drop(columns=["instant", "cnt", "casual", "registered"])
y = df["cnt"]


#This function defines a forward fill method for Null Values
def ffill_missing(ser):
    return ser.fillna(methods="ffill")


#Define the forward fill imputer
ffiller = FunctionTransformer(ffill_missing)


#Define the weathersit data encoder
weather_enc = make_pipeline(
    ffiller,
    OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=X["weathersit"].nunique()
    ),
)
weather_enc


ct = make_column_transformer(
    (ffiller, make_column_selector(dtype_include=np.number)),
    (weather_enc, ["weathersit"]),
)
ct

#This function is used further ahead to consider binary feature of wether the day is weekday or weekend
def is_weekend(data):
    return (
        data["dteday"]
        .dt.day_name()
        .isin(["Saturday", "Sunday"])
        .to_frame()
    )


#This function is used to tell how many years have elapsed since the beginning of operation (began in 2011)
def year(data):
    # Our reference year is 2011, the beginning of the training dataset
    return (data["dteday"].dt.year - 2011).to_frame()

#Defining the preprocessing pipeline object to apply all the data
preprocessing = FeatureUnion([
    ("is_weekend", FunctionTransformer(is_weekend)),
    ("year", FunctionTransformer(year)),
    ("column_transform", ct)
])


reg = Pipeline([("preprocessing", preprocessing), ("model", RandomForestRegressor())])
reg


X_train, y_train = X.loc[X["dteday"] < "2012-10"], y.loc[X["dteday"] < "2012-10"]
X_test, y_test = X.loc["2012-10" <= X["dteday"]], y.loc["2012-10" <= X["dteday"]]


reg.fit(X_train, y_train)


 #Dump the model
    
joblib.dump(reg, 'bikes.joblib')

reg.score(X_test, y_test)

y_pred = reg.predict(X_test)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fig, ax = plt.subplots(figsize=(15, 5))

df.loc[df["dteday"] < "2012-10"].set_index("instant")["cnt"].plot(ax=ax, label="Train")
df.loc["2012-10" <= df["dteday"]].set_index("instant")["cnt"].plot(ax=ax, label="Test")

pd.Series(y_pred, index=df.loc["2012-10" <= df["dteday"], "instant"]).plot(ax=ax, color="k", label="Prediction")

ax.legend(loc=2, shadow=True, facecolor="0.97")
ax.set_xlim(15100, 15500)


X_train.head()


reg.predict(pd.DataFrame([[
    pd.to_datetime("2012-11-01"),
    10,
    "Clear, Few clouds, Partly cloudy, Partly cloudy",
    0.3,
    0.31,
    0.8,
    0.0,
]], columns=[
    'dteday',
    'hr',
    'weathersit',
    'temp',
    'atemp',
    'hum',
    'windspeed'
]))