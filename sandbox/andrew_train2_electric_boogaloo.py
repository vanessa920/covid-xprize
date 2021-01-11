#!/usr/bin/env python
# coding: utf-8
# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.
# # Example Predictor: Linear Rollout Predictor
# 
# This example contains basic functionality for training and evaluating a linear predictor that rolls out predictions day-by-day.
# 
# First, a training data set is created from historical case and npi data.
# 
# Second, a linear model is trained to predict future cases from prior case data along with prior and future npi data.
# The model is an off-the-shelf sklearn Lasso model, that uses a positive weight constraint to enforce the assumption that increased npis has a negative correlation with future cases.
# 
# Third, a sample evaluation set is created, and the predictor is applied to this evaluation set to produce prediction results in the correct format.

# ## Training

# In[13]:


import pickle
import numpy as np
import pandas as pd
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split
from scipy.integrate import odeint
import seaborn as sns


# ### Copy the data locally

# In[3]:


# routine to remerge & save the data
df = pd.read_csv("data/merged_data.csv", index_col=0)


# In[4]:


# The SIR model differential equations
# y = (S, I, R)
def deriv(y, t, beta, gamma):
    S, I, R = y
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# In[ ]:


# https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788


# In[ ]:


df[df["GeoID"] == "United States__nan"]


# In[49]:


df[df["GeoID"] == "United States__nan"]["Infected"]


# In[5]:


df.update(df.groupby('GeoID')["SusceptibleDiff"].transform(lambda x: pd.concat([x.iloc[:-1], x.iloc[-2:-1]]) if x.iloc[-1] == 0 else x))
df.update(df.groupby('GeoID')["InfectedDiff"].transform(lambda x: pd.concat([x.iloc[:-1], x.iloc[-2:-1]]) if x.iloc[-1] == 0 else x))
df.update(df.groupby('GeoID')["DeathsDiff"].transform(lambda x: pd.concat([x.iloc[:-1], x.iloc[-2:-1]]) if x.iloc[-1] == 0 else x))
df.update(df.groupby('GeoID')["RecoveredDiff"].transform(lambda x: pd.concat([x.iloc[:-1], x.iloc[-2:-1]]) if x.iloc[-1] == 0 else x))
df.update(df.groupby('GeoID')["RemovedDiff"].transform(lambda x: pd.concat([x.iloc[:-1], x.iloc[-2:-1]]) if x.iloc[-1] == 0 else x))


# In[ ]:


pd.set_option('display_all')


# In[8]:


df.sample(5)


# In[ ]:


# should probably fill in the last difference with the same as the day prior


# In[ ]:


# can use ewm as well


# In[9]:


df["SDiffSmooth"] = df.groupby('GeoID')["SusceptibleDiff"].transform(lambda x: x.rolling(window=12).mean().fillna(0))
df["IDiffSmooth"] = df.groupby('GeoID')["InfectedDiff"].transform(lambda x: x.rolling(window=12).mean().fillna(0))
df["RDiffSmooth"] = df.groupby('GeoID')["RemovedDiff"].transform(lambda x: x.rolling(window=12).mean().fillna(0))


# In[10]:


sir_cols = ["Susceptible", "Infected", "Removed", "SusceptibleDiff", "InfectedDiff", "RemovedDiff"]


# In[11]:


data_df = df[['GeoID', 'Date'] + sir_cols]
us_data = data_df[data_df["GeoID"] == "United States__nan"]


# In[12]:


us_data.head()


# In[21]:


us_data.describe()


# In[23]:


us_data.info()


# In[18]:


sns.scatterplot(list(range(len(us_data["Date"]))), us_data["Infected"])


# In[19]:


sns.scatterplot(list(range(len(us_data["Date"]))), us_data["Susceptible"])


# In[20]:


sns.scatterplot(list(range(len(us_data["Date"]))), us_data["Removed"])


# ### early pandemic data might not be helpful in determining recent trends so just try to consider rows after 110 (~April 20)

# In[25]:


later_us_data = us_data.iloc[110:]
later_us_data.head()


# In[27]:


later_us_data.describe()


# In[31]:


sns.scatterplot(list(range(len(later_us_data["Date"]))), later_us_data["Infected"]);


# In[32]:


sns.scatterplot(list(range(len(later_us_data["Date"]))), later_us_data["Susceptible"]);


# In[33]:


sns.scatterplot(list(range(len(later_us_data["Date"]))), later_us_data["Removed"]);


# In[38]:


later_us_data[['Date','Removed']].tail(20)


# In[39]:


later_us_data[['Date','Infected']].tail(20)


# In[ ]:





# In[ ]:





# In[ ]:


odeint()


# In[10]:


# routine to forecast the next X days given an SIR model and params

def forecast_future(S, I, R, beta, gamma, days):
    y_i = S, I, R
    change = odeint(deriv, y_i, [i for i in range(days)], args=(beta, gamma))
    return change


# In[ ]:


# solve: beta = -N*dS/dt*SI


# In[ ]:


# get a beta/gamma for each timestep
# we'll want to do smoothing of some sort on the cases
# just get dSdt, dIdt, dRdt from smoothing of the S/I/R data
# then use that to make dbeta/dgamma dt, then use NPIs to predict dbeta/dgamma dt
# we assume beta/gamma evolve through some equation depending on the NPIs/previous values of S,I,R over the last X days
# then find the past NPI+ country(?) data that best predicts dbeta/dgamma dt
# then use that for the actual prediction


# In[13]:


# Set number of past days to use to make predictions
nb_lookback_days = 30

# Create training data across all countries for predicting one day ahead
X_cols = cases_col + npi_cols
y_col = cases_col
X_samples = []
y_samples = []
geo_ids = df.GeoID.unique()
for g in geo_ids:
    gdf = df[df.GeoID == g]
    all_case_data = np.array(gdf[cases_col])
    all_npi_data = np.array(gdf[npi_cols])

    # Create one sample for each day where we have enough data
    # Each sample consists of cases and npis for previous nb_lookback_days
    nb_total_days = len(gdf)
    for d in range(nb_lookback_days, nb_total_days - 1):
        X_cases = all_case_data[d-nb_lookback_days:d]

        # Take negative of npis to support positive
        # weight constraint in Lasso.
        X_npis = -all_npi_data[d - nb_lookback_days:d]

        # Flatten all input data so it fits Lasso input format.
        X_sample = np.concatenate([X_cases.flatten(),
                                   X_npis.flatten()])
        y_sample = all_case_data[d]
        X_samples.append(X_sample)
        y_samples.append(y_sample)

X_samples = np.array(X_samples)
y_samples = np.array(y_samples).flatten()


# In[14]:


# Helpful function to compute mae
def mae(pred, true):
    return np.mean(np.abs(pred - true))


# In[15]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                    y_samples,
                                                    test_size=0.2,
                                                    random_state=301)


# In[16]:


# Create and train Lasso model.
# Set positive=True to enforce assumption that cases are positively correlated
# with future cases and npis are negatively correlated.
model = Lasso(alpha=0.1,
              precompute=True,
              max_iter=10000,
              positive=True,
              selection='random')
# Fit model
model.fit(X_train, y_train)


# In[17]:


# Evaluate model
train_preds = model.predict(X_train)
train_preds = np.maximum(train_preds, 0) # Don't predict negative cases
print('Train MAE:', mae(train_preds, y_train))

test_preds = model.predict(X_test)
test_preds = np.maximum(test_preds, 0) # Don't predict negative cases
print('Test MAE:', mae(test_preds, y_test))


# In[18]:


# Inspect the learned feature coefficients for the model
# to see what features it's paying attention to.

# Give names to the features
x_col_names = []
for d in range(-nb_lookback_days, 0):
    x_col_names.append('Day ' + str(d) + ' ' + cases_col[0])
for d in range(-nb_lookback_days, 1):
    for col_name in npi_cols:
        x_col_names.append('Day ' + str(d) + ' ' + col_name)

# View non-zero coefficients
for (col, coeff) in zip(x_col_names, list(model.coef_)):
    if coeff != 0.:
        print(col, coeff)
print('Intercept', model.intercept_)


# In[19]:


# Save model to file
if not os.path.exists('models'):
    os.mkdir('models')
with open('models/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


# ## Evaluation
# 
# Now that the predictor has been trained and saved, this section contains the functionality for evaluating it on sample evaluation data.

# In[20]:


# Reload the module to get the latest changes
import predict
from importlib import reload
reload(predict)
from predict import predict_df


# In[21]:


get_ipython().run_cell_magic('time', '', 'preds_df = predict_df("2020-08-01", "2020-08-31", path_to_ips_file="../../../validation/data/2020-09-30_historical_ip.csv", verbose=True)')


# In[23]:


# Check the predictions
preds_df.head()


# # Validation
# This is how the predictor is going to be called during the competition.  
# !!! PLEASE DO NOT CHANGE THE API !!!

# In[ ]:


get_ipython().system('python predict.py -s 2020-08-01 -e 2020-08-04 -ip ../../../validation/data/2020-09-30_historical_ip.csv -o predictions/2020-08-01_2020-08-04.csv')


# In[ ]:


get_ipython().system('head predictions/2020-08-01_2020-08-04.csv')


# # Test cases
# We can generate a prediction file. Let's validate a few cases...

# In[26]:


import os
import sys
sys.path.append("../../../..")
from covid_xprize.validation.predictor_validation import validate_submission

def validate(start_date, end_date, ip_file, output_file):
    # First, delete any potential old file
    try:
        os.remove(output_file)
    except OSError:
        pass
    
    # Then generate the prediction, calling the official API
    get_ipython().system('python predict.py -s {start_date} -e {end_date} -ip {ip_file} -o {output_file}')
    
    # And validate it
    errors = validate_submission(start_date, end_date, ip_file, output_file)
    if errors:
        for error in errors:
            print(error)
    else:
        print("All good!")


# ## 4 days, no gap
# - All countries and regions
# - Official number of cases is known up to start_date
# - Intervention Plans are the official ones

# In[27]:


validate(start_date="2020-08-01",
         end_date="2020-08-04",
         ip_file="../../../validation/data/2020-09-30_historical_ip.csv",
         output_file="predictions/val_4_days.csv")


# ## 1 month in the future
# - 2 countries only
# - there's a gap between date of last known number of cases and start_date
# - For future dates, Intervention Plans contains scenarios for which predictions are requested to answer the question: what will happen if we apply these plans?

# In[28]:


get_ipython().run_cell_magic('time', '', 'validate(start_date="2021-01-01",\n         end_date="2021-01-31",\n         ip_file="../../../validation/data/future_ip.csv",\n         output_file="predictions/val_1_month_future.csv")')


# ## 180 days, from a future date, all countries and regions
# - Prediction start date is 1 week from now. (i.e. assuming submission date is 1 week from now)  
# - Prediction end date is 6 months after start date.  
# - Prediction is requested for all available countries and regions.  
# - Intervention plan scenario: freeze last known intervention plans for each country and region.  
# 
# As the number of cases is not known yet between today and start date, but the model relies on them, the model has to predict them in order to use them.  
# This test is the most demanding test. It should take less than 1 hour to generate the prediction file.

# ### Generate the scenario

# In[29]:


from datetime import datetime, timedelta

start_date = datetime.now() + timedelta(days=7)
start_date_str = start_date.strftime('%Y-%m-%d')
end_date = start_date + timedelta(days=180)
end_date_str = end_date.strftime('%Y-%m-%d')
print(f"Start date: {start_date_str}")
print(f"End date: {end_date_str}")


# In[30]:


from covid_xprize.validation.scenario_generator import get_raw_data, generate_scenario, NPI_COLUMNS
DATA_FILE = 'data/OxCGRT_latest.csv'
latest_df = get_raw_data(DATA_FILE, latest=True)
scenario_df = generate_scenario(start_date_str, end_date_str, latest_df, countries=None, scenario="Freeze")
scenario_file = "predictions/180_days_future_scenario.csv"
scenario_df.to_csv(scenario_file, index=False)
print(f"Saved scenario to {scenario_file}")


# ### Check it

# In[ ]:


get_ipython().run_cell_magic('time', '', 'validate(start_date=start_date_str,\n         end_date=end_date_str,\n         ip_file=scenario_file,\n         output_file="predictions/val_6_month_future.csv")')


# In[ ]:




