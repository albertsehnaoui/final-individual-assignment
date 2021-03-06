{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "white-shaft",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import joblib as joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "dedicated-friendship",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>instant</th>\n",
       "      <th>dteday</th>\n",
       "      <th>hr</th>\n",
       "      <th>weathersit</th>\n",
       "      <th>temp</th>\n",
       "      <th>atemp</th>\n",
       "      <th>hum</th>\n",
       "      <th>windspeed</th>\n",
       "      <th>casual</th>\n",
       "      <th>registered</th>\n",
       "      <th>cnt</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>2011-01-01</td>\n",
       "      <td>0</td>\n",
       "      <td>Clear, Few clouds, Partly cloudy, Partly cloudy</td>\n",
       "      <td>0.24</td>\n",
       "      <td>0.2879</td>\n",
       "      <td>0.81</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>13</td>\n",
       "      <td>16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2011-01-01</td>\n",
       "      <td>1</td>\n",
       "      <td>Clear, Few clouds, Partly cloudy, Partly cloudy</td>\n",
       "      <td>0.22</td>\n",
       "      <td>0.2727</td>\n",
       "      <td>0.80</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8</td>\n",
       "      <td>32</td>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>2011-01-01</td>\n",
       "      <td>2</td>\n",
       "      <td>Clear, Few clouds, Partly cloudy, Partly cloudy</td>\n",
       "      <td>0.22</td>\n",
       "      <td>0.2727</td>\n",
       "      <td>0.80</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5</td>\n",
       "      <td>27</td>\n",
       "      <td>32</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>2011-01-01</td>\n",
       "      <td>3</td>\n",
       "      <td>Clear, Few clouds, Partly cloudy, Partly cloudy</td>\n",
       "      <td>0.24</td>\n",
       "      <td>0.2879</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>10</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>2011-01-01</td>\n",
       "      <td>4</td>\n",
       "      <td>Clear, Few clouds, Partly cloudy, Partly cloudy</td>\n",
       "      <td>0.24</td>\n",
       "      <td>0.2879</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   instant     dteday  hr                                       weathersit  \\\n",
       "0        1 2011-01-01   0  Clear, Few clouds, Partly cloudy, Partly cloudy   \n",
       "1        2 2011-01-01   1  Clear, Few clouds, Partly cloudy, Partly cloudy   \n",
       "2        3 2011-01-01   2  Clear, Few clouds, Partly cloudy, Partly cloudy   \n",
       "3        4 2011-01-01   3  Clear, Few clouds, Partly cloudy, Partly cloudy   \n",
       "4        5 2011-01-01   4  Clear, Few clouds, Partly cloudy, Partly cloudy   \n",
       "\n",
       "   temp   atemp   hum  windspeed  casual  registered  cnt  \n",
       "0  0.24  0.2879  0.81        0.0       3          13   16  \n",
       "1  0.22  0.2727  0.80        0.0       8          32   40  \n",
       "2  0.22  0.2727  0.80        0.0       5          27   32  \n",
       "3  0.24  0.2879  0.75        0.0       3          10   13  \n",
       "4  0.24  0.2879  0.75        0.0       0           1    1  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"hour.csv\", parse_dates=[\"dteday\"])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "knowing-fancy",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already up-to-date: scikit-learn in ./.local/lib/python3.8/site-packages (0.24.1)\n",
      "Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in ./opt/anaconda3/lib/python3.8/site-packages (from scikit-learn) (2.1.0)\n",
      "Requirement already satisfied, skipping upgrade: joblib>=0.11 in ./opt/anaconda3/lib/python3.8/site-packages (from scikit-learn) (0.17.0)\n",
      "Requirement already satisfied, skipping upgrade: numpy>=1.13.3 in ./opt/anaconda3/lib/python3.8/site-packages (from scikit-learn) (1.19.2)\n",
      "Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in ./opt/anaconda3/lib/python3.8/site-packages (from scikit-learn) (1.5.2)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install --user --upgrade scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "charged-navigation",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 17379 entries, 0 to 17378\n",
      "Data columns (total 11 columns):\n",
      " #   Column      Non-Null Count  Dtype         \n",
      "---  ------      --------------  -----         \n",
      " 0   instant     17379 non-null  int64         \n",
      " 1   dteday      17379 non-null  datetime64[ns]\n",
      " 2   hr          17379 non-null  int64         \n",
      " 3   weathersit  17279 non-null  object        \n",
      " 4   temp        17280 non-null  float64       \n",
      " 5   atemp       17279 non-null  float64       \n",
      " 6   hum         17279 non-null  float64       \n",
      " 7   windspeed   17279 non-null  float64       \n",
      " 8   casual      17379 non-null  int64         \n",
      " 9   registered  17379 non-null  int64         \n",
      " 10  cnt         17379 non-null  int64         \n",
      "dtypes: datetime64[ns](1), float64(4), int64(5), object(1)\n",
      "memory usage: 1.5+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "specific-recorder",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Dropping the columns we don't need from X, and defining the target variables Y\n",
    "X = df.drop(columns=[\"instant\", \"cnt\", \"casual\", \"registered\"])\n",
    "y = df[\"cnt\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "critical-driving",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "rough-rebate",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector\n",
    "from sklearn.impute import SimpleImputer, KNNImputer\n",
    "from sklearn.pipeline import Pipeline, make_pipeline\n",
    "from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "overhead-selection",
   "metadata": {},
   "outputs": [],
   "source": [
    "# This function defines a forward fill method to fill NAs\n",
    "def ffill_missing(ser):\n",
    "    return ser.fillna(method=\"ffill\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "double-warrant",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define the forward fill imputer\n",
    "ffiller = FunctionTransformer(ffill_missing)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "collective-merchandise",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(steps=[('functiontransformer',\n",
       "                 FunctionTransformer(func=<function ffill_missing at 0x7fbbfc867e50>)),\n",
       "                ('ordinalencoder',\n",
       "                 OrdinalEncoder(handle_unknown='use_encoded_value',\n",
       "                                unknown_value=4))])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Define the weathersit data encoder\n",
    "weather_enc = make_pipeline(\n",
    "    ffiller,\n",
    "    OrdinalEncoder(\n",
    "        handle_unknown=\"use_encoded_value\", unknown_value=X[\"weathersit\"].nunique()\n",
    "    ),\n",
    ")\n",
    "weather_enc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "hollow-lyric",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ColumnTransformer(transformers=[('functiontransformer',\n",
       "                                 FunctionTransformer(func=<function ffill_missing at 0x7fbbfc867e50>),\n",
       "                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7fbbfc85bcd0>),\n",
       "                                ('pipeline',\n",
       "                                 Pipeline(steps=[('functiontransformer',\n",
       "                                                  FunctionTransformer(func=<function ffill_missing at 0x7fbbfc867e50>)),\n",
       "                                                 ('ordinalencoder',\n",
       "                                                  OrdinalEncoder(handle_unknown='use_encoded_value',\n",
       "                                                                 unknown_value=4))]),\n",
       "                                 ['weathersit'])])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ct = make_column_transformer(\n",
    "    (ffiller, make_column_selector(dtype_include=np.number)),\n",
    "    (weather_enc, [\"weathersit\"]),\n",
    ")\n",
    "ct"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "suspected-space",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import FeatureUnion, make_union"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "extensive-retirement",
   "metadata": {},
   "outputs": [],
   "source": [
    "#This function is used further ahead to consider binary feature of wether the day is weekday or weekend\n",
    "def is_weekend(data):\n",
    "    return (\n",
    "        data[\"dteday\"]\n",
    "        .dt.day_name()\n",
    "        .isin([\"Saturday\", \"Sunday\"])\n",
    "        .to_frame()\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "toxic-forestry",
   "metadata": {},
   "outputs": [],
   "source": [
    "#This function is used to tell how many years have elapsed since the beginning of operation (began in 2011)\n",
    "def year(data):\n",
    "    # Our reference year is 2011, the beginning of the training dataset\n",
    "    return (data[\"dteday\"].dt.year - 2011).to_frame()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "compatible-spirituality",
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "FeatureUnion(transformer_list=[('is_weekend',\n",
       "                                FunctionTransformer(func=<function is_weekend at 0x7fbbfc86ad30>)),\n",
       "                               ('year',\n",
       "                                FunctionTransformer(func=<function year at 0x7fbbfc886040>)),\n",
       "                               ('column_transform',\n",
       "                                ColumnTransformer(transformers=[('functiontransformer',\n",
       "                                                                 FunctionTransformer(func=<function ffill_missing at 0x7fbbfc867e50>),\n",
       "                                                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7fbbfc85bcd0>),\n",
       "                                                                ('pipeline',\n",
       "                                                                 Pipeline(steps=[('functiontransformer',\n",
       "                                                                                  FunctionTransformer(func=<function ffill_missing at 0x7fbbfc867e50>)),\n",
       "                                                                                 ('ordinalencoder',\n",
       "                                                                                  OrdinalEncoder(handle_unknown='use_encoded_value',\n",
       "                                                                                                 unknown_value=4))]),\n",
       "                                                                 ['weathersit'])]))])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "preprocessing = FeatureUnion([\n",
    "    (\"is_weekend\", FunctionTransformer(is_weekend)),\n",
    "    (\"year\", FunctionTransformer(year)),\n",
    "    (\"column_transform\", ct)\n",
    "])\n",
    "preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "geographic-smart",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "fourth-genesis",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(steps=[('preprocessing',\n",
       "                 FeatureUnion(transformer_list=[('is_weekend',\n",
       "                                                 FunctionTransformer(func=<function is_weekend at 0x7fbbfc86ad30>)),\n",
       "                                                ('year',\n",
       "                                                 FunctionTransformer(func=<function year at 0x7fbbfc886040>)),\n",
       "                                                ('column_transform',\n",
       "                                                 ColumnTransformer(transformers=[('functiontransformer',\n",
       "                                                                                  FunctionTransformer(func=<function ffill_missing at 0x7fbbfc867e50>),\n",
       "                                                                                  <sklearn.compose._column_transformer.make_column_selector object at 0x7fbbfc85bcd0>),\n",
       "                                                                                 ('pipeline',\n",
       "                                                                                  Pipeline(steps=[('functiontransformer',\n",
       "                                                                                                   FunctionTransformer(func=<function ffill_missing at 0x7fbbfc867e50>)),\n",
       "                                                                                                  ('ordinalencoder',\n",
       "                                                                                                   OrdinalEncoder(handle_unknown='use_encoded_value',\n",
       "                                                                                                                  unknown_value=4))]),\n",
       "                                                                                  ['weathersit'])]))])),\n",
       "                ('model', RandomForestRegressor())])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg = Pipeline([(\"preprocessing\", preprocessing), (\"model\", RandomForestRegressor())])\n",
    "reg"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "choice-manufacturer",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, y_train = X.loc[X[\"dteday\"] < \"2012-10\"], y.loc[X[\"dteday\"] < \"2012-10\"]\n",
    "X_test, y_test = X.loc[\"2012-10\" <= X[\"dteday\"]], y.loc[\"2012-10\" <= X[\"dteday\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "alone-beach",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(steps=[('preprocessing',\n",
       "                 FeatureUnion(transformer_list=[('is_weekend',\n",
       "                                                 FunctionTransformer(func=<function is_weekend at 0x7fbbfc86ad30>)),\n",
       "                                                ('year',\n",
       "                                                 FunctionTransformer(func=<function year at 0x7fbbfc886040>)),\n",
       "                                                ('column_transform',\n",
       "                                                 ColumnTransformer(transformers=[('functiontransformer',\n",
       "                                                                                  FunctionTransformer(func=<function ffill_missing at 0x7fbbfc867e50>),\n",
       "                                                                                  <sklearn.compose._column_transformer.make_column_selector object at 0x7fbbfc85bcd0>),\n",
       "                                                                                 ('pipeline',\n",
       "                                                                                  Pipeline(steps=[('functiontransformer',\n",
       "                                                                                                   FunctionTransformer(func=<function ffill_missing at 0x7fbbfc867e50>)),\n",
       "                                                                                                  ('ordinalencoder',\n",
       "                                                                                                   OrdinalEncoder(handle_unknown='use_encoded_value',\n",
       "                                                                                                                  unknown_value=4))]),\n",
       "                                                                                  ['weathersit'])]))])),\n",
       "                ('model', RandomForestRegressor())])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "contained-pickup",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['bikes.joblib']"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    " #Dump the model\n",
    "    \n",
    "joblib.dump(reg, 'bikes.joblib')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "spanish-argument",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8028155482164009"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "painful-command",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = reg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "distributed-forty",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "sns.set()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "qualified-influence",
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "female-riverside",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(15100.0, 15500.0)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(figsize=(15, 5))\n",
    "\n",
    "df.loc[df[\"dteday\"] < \"2012-10\"].set_index(\"instant\")[\"cnt\"].plot(ax=ax, label=\"Train\")\n",
    "df.loc[\"2012-10\" <= df[\"dteday\"]].set_index(\"instant\")[\"cnt\"].plot(ax=ax, label=\"Test\")\n",
    "\n",
    "pd.Series(y_pred, index=df.loc[\"2012-10\" <= df[\"dteday\"], \"instant\"]).plot(ax=ax, color=\"k\", label=\"Prediction\")\n",
    "\n",
    "ax.legend(loc=2, shadow=True, facecolor=\"0.97\")\n",
    "ax.set_xlim(15100, 15500)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "vocational-testing",
   "metadata": {},
   "source": [
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "going-advocacy",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>dteday</th>\n",
       "      <th>hr</th>\n",
       "      <th>weathersit</th>\n",
       "      <th>temp</th>\n",
       "      <th>atemp</th>\n",
       "      <th>hum</th>\n",
       "      <th>windspeed</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2011-01-01</td>\n",
       "      <td>0</td>\n",
       "      <td>Clear, Few clouds, Partly cloudy, Partly cloudy</td>\n",
       "      <td>0.24</td>\n",
       "      <td>0.2879</td>\n",
       "      <td>0.81</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2011-01-01</td>\n",
       "      <td>1</td>\n",
       "      <td>Clear, Few clouds, Partly cloudy, Partly cloudy</td>\n",
       "      <td>0.22</td>\n",
       "      <td>0.2727</td>\n",
       "      <td>0.80</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2011-01-01</td>\n",
       "      <td>2</td>\n",
       "      <td>Clear, Few clouds, Partly cloudy, Partly cloudy</td>\n",
       "      <td>0.22</td>\n",
       "      <td>0.2727</td>\n",
       "      <td>0.80</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2011-01-01</td>\n",
       "      <td>3</td>\n",
       "      <td>Clear, Few clouds, Partly cloudy, Partly cloudy</td>\n",
       "      <td>0.24</td>\n",
       "      <td>0.2879</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2011-01-01</td>\n",
       "      <td>4</td>\n",
       "      <td>Clear, Few clouds, Partly cloudy, Partly cloudy</td>\n",
       "      <td>0.24</td>\n",
       "      <td>0.2879</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      dteday  hr                                       weathersit  temp  \\\n",
       "0 2011-01-01   0  Clear, Few clouds, Partly cloudy, Partly cloudy  0.24   \n",
       "1 2011-01-01   1  Clear, Few clouds, Partly cloudy, Partly cloudy  0.22   \n",
       "2 2011-01-01   2  Clear, Few clouds, Partly cloudy, Partly cloudy  0.22   \n",
       "3 2011-01-01   3  Clear, Few clouds, Partly cloudy, Partly cloudy  0.24   \n",
       "4 2011-01-01   4  Clear, Few clouds, Partly cloudy, Partly cloudy  0.24   \n",
       "\n",
       "    atemp   hum  windspeed  \n",
       "0  0.2879  0.81        0.0  \n",
       "1  0.2727  0.80        0.0  \n",
       "2  0.2727  0.80        0.0  \n",
       "3  0.2879  0.75        0.0  \n",
       "4  0.2879  0.75        0.0  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "registered-diesel",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([104.81])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.predict(pd.DataFrame([[\n",
    "    pd.to_datetime(\"2012-11-01\"),\n",
    "    10,\n",
    "    \"Clear, Few clouds, Partly cloudy, Partly cloudy\",\n",
    "    0.3,\n",
    "    0.31,\n",
    "    0.8,\n",
    "    0.0,\n",
    "]], columns=[\n",
    "    'dteday',\n",
    "    'hr',\n",
    "    'weathersit',\n",
    "    'temp',\n",
    "    'atemp',\n",
    "    'hum',\n",
    "    'windspeed'\n",
    "]))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
