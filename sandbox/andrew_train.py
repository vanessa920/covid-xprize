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

# In[4]:


import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint


# ### Copy the data locally

# In[13]:


import seaborn as sns


# In[5]:


# routine to remerge & save the data
df = pd.read_csv("data/merged_data.csv", index_col=0)


# In[6]:


df[df["CountryName"]


# In[25]:


def country_data(country):
    return df[(df["CountryName"]==country)&(df["RegionName"].isna())]


# In[30]:


def fit_sir(data):
    date_range = pd.date_range(df['Date'].min(), df['Date'].max(), freq='D').strftime("%m-%d-%Y")
    window_size = 15
    extend = (int(window_size/2))-1
    for middle in range(extend, len(date_range)-start_idx):
        window_data = data.iloc[middle-extend:middle-extend+window_size]
        y_data = window_data[["Susceptible", "Infected"]]
        x_data = [i for i in range(window_size)]
        


# In[31]:


fit_sir(country_data("Sweden"))


# In[ ]:


def fit_sir_helper(x_data, y_data):
        


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

ydata = ['1e-06', '1.49920166169172e-06', '2.24595472686361e-06', '3.36377954575331e-06', '5.03793663882291e-06', '7.54533628058909e-06', '1.13006564683911e-05', '1.69249500601052e-05', '2.53483161761933e-05', '3.79636391699325e-05', '5.68567547875179e-05', '8.51509649182741e-05', '0.000127522555808945', '0.000189928392105942', '0.000283447055673738', '0.000423064043409294', '0.000631295993246634', '0.000941024110897193', '0.00140281896645859', '0.00209085569326554', '0.00311449589149717', '0.00463557784224762', '0.00689146863803467', '0.010227347567051', '0.0151380084180746', '0.0223233100045688', '0.0327384810150231', '0.0476330618585758', '0.0685260046667727', '0.0970432959143974', '0.134525888779423', '0.181363340075877', '0.236189247803334', '0.295374180276257', '0.353377036130714', '0.404138746080267', '0.442876028839178', '0.467273954573897', '0.477529937494976', '0.475582401936257', '0.464137179474659', '0.445930281787152', '0.423331710456602', '0.39821360956389', '0.371967226561944', '0.345577884704341', '0.319716449520481', '0.294819942458255', '0.271156813453547', '0.24887641905719', '0.228045466022105', '0.208674420183194', '0.190736203926912', '0.174179448652951', '0.158937806544529', '0.144936441326754', '0.132096533873646', '0.120338367115739', '0.10958340819268', '0.099755679236243', '0.0907826241267504', '0.0825956203546979', '0.0751302384111894', '0.0683263295744258', '0.0621279977639921', '0.0564834809370572', '0.0513449852139111', '0.0466684871328814', '0.042413516167789', '0.0385429293775096', '0.035022685071934', '0.0318216204865132', '0.0289112368382048', '0.0262654939162707', '0.0238606155312519', '0.021674906523588', '0.0196885815912485', '0.0178836058829335', '0.0162435470852779', '0.0147534385851646', '0.0133996531928511', '0.0121697868544064', '0.0110525517526551', '0.0100376781867076', '0.00911582462544914', '0.00827849534575178', '0.00751796508841916', '0.00682721019158058', '0.00619984569061827', '0.00563006790443123', '0.00511260205894446', '0.00464265452957236', '0.00421586931435123', '0.00382828837833139', '0.00347631553734708', '0.00315668357532714', '0.00286642431380459', '0.00260284137520731', '0.00236348540287827', '0.00214613152062159', '0.00194875883295343']
xdata = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101']

ydata = np.array(ydata, dtype=float)
xdata = np.array(xdata, dtype=float)

def sir_model(y, x, beta, gamma):
    S = -beta * y[0] * y[1] / N
    R = gamma * y[1]
    I = -(S + R)
    return S, I, R

def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:,1]

N = 1.0
I0 = ydata[0]
S0 = N - I0
R0 = 0.0

popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
fitted = fit_odeint(xdata, *popt)

plt.plot(xdata, ydata, 'o')
plt.plot(xdata, fitted)
plt.show()


# In[ ]:


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


df.update(df.groupby('GeoID')["SusceptibleDiff"].transform(lambda x: pd.concat([x.iloc[:-1], x.iloc[-2:-1]]) if x.iloc[-1] == 0 else x))
df.update(df.groupby('GeoID')["InfectedDiff"].transform(lambda x: pd.concat([x.iloc[:-1], x.iloc[-2:-1]]) if x.iloc[-1] == 0 else x))
df.update(df.groupby('GeoID')["DeathsDiff"].transform(lambda x: pd.concat([x.iloc[:-1], x.iloc[-2:-1]]) if x.iloc[-1] == 0 else x))
df.update(df.groupby('GeoID')["RecoveredDiff"].transform(lambda x: pd.concat([x.iloc[:-1], x.iloc[-2:-1]]) if x.iloc[-1] == 0 else x))
df.update(df.groupby('GeoID')["RemovedDiff"].transform(lambda x: pd.concat([x.iloc[:-1], x.iloc[-2:-1]]) if x.iloc[-1] == 0 else x))


# In[ ]:


# should probably fill in the last difference with the same as the day prior


# In[ ]:


# can use ewm as well


# In[ ]:


df["SDiffSmooth"] = df.groupby('GeoID')["SusceptibleDiff"].transform(lambda x: x.rolling(window=12).mean().fillna(0))
df["IDiffSmooth"] = df.groupby('GeoID')["InfectedDiff"].transform(lambda x: x.rolling(window=12).mean().fillna(0))
df["RDiffSmooth"] = df.groupby('GeoID')["RemovedDiff"].transform(lambda x: x.rolling(window=12).mean().fillna(0))


# In[ ]:


sir_cols = ["Susceptible", "Infected", "Removed", "SusceptibleDiff", "InfectedDiff", "RemovedDiff"]


# In[1]:


data_df = df[['GeoID', 'Date'] + sir_cols]
us_data = data_df[data_df["GeoID"] == "United States__nan"]


# In[ ]:





# In[2]:


sns.plot(us_data["Date"], us_data["Infected"])


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




