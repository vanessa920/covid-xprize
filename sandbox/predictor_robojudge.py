#!/usr/bin/env python
# coding: utf-8
# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.
# In[1]:


import os
import pandas as pd
import numpy as np


# # Load data set

# In[2]:


LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'


# In[3]:


def load_dataset(url):
    latest_df = pd.read_csv(url,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)
    latest_df["RegionName"] = latest_df["RegionName"].fillna("")
    return latest_df


# In[4]:


latest_df = load_dataset(LATEST_DATA_URL)


# In[5]:


latest_df.sample(3)


# # Get NPIs

# In[6]:


NPI_COLUMNS = ['C1_School closing',
               'C2_Workplace closing',
               'C3_Cancel public events',
               'C4_Restrictions on gatherings',
               'C5_Close public transport',
               'C6_Stay at home requirements',
               'C7_Restrictions on internal movement',
               'C8_International travel controls',
               'H1_Public information campaigns',
               'H2_Testing policy',
               'H3_Contact tracing',
               'H6_Facial Coverings']


# In[7]:


npis_df = latest_df[["CountryName", "RegionName", "Date"] + NPI_COLUMNS]


# In[8]:


npis_df.sample(3)


# # Dates

# In[9]:


start_date_str = "2020-12-01"
end_date_str = "2020-12-31"


# In[10]:


start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')


# In[11]:


actual_npis_df = npis_df[(npis_df.Date >= start_date) & (npis_df.Date <= end_date)]
actual_npis_df.sample(3)


# # Get actual cases between these dates

# In[12]:


NUM_PREV_DAYS_TO_INCLUDE = 6
WINDOW_SIZE = 7


# In[13]:


def get_actual_cases(df, start_date, end_date):
    # 1 day earlier to compute the daily diff
    start_date_for_diff = start_date - pd.offsets.Day(WINDOW_SIZE)
    actual_df = df[["CountryName", "RegionName", "Date", "ConfirmedCases"]]
    # Filter out the data set to include all the data needed to compute the diff
    actual_df = actual_df[(actual_df.Date >= start_date_for_diff) & (actual_df.Date <= end_date)]
    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    # np.where usage: if A then B else C
    actual_df["GeoID"] = np.where(actual_df["RegionName"] == '',
                                  actual_df["CountryName"],
                                  actual_df["CountryName"] + ' / ' + actual_df["RegionName"])
    actual_df.sort_values(by=["GeoID","Date"], inplace=True)
    # Compute the diff
    actual_df["ActualDailyNewCases"] = actual_df.groupby("GeoID")["ConfirmedCases"].diff()
    # Compute the 7 day moving average
    actual_df["ActualDailyNewCases7DMA"] = actual_df.groupby(
        "GeoID")['ActualDailyNewCases'].rolling(
        WINDOW_SIZE, center=False).mean().reset_index(0, drop=True)
    return actual_df


# In[14]:


actual_df = get_actual_cases(latest_df, start_date, end_date)


# In[15]:


# Add population column.
from covid_xprize.scoring.predictor_scoring import add_population_column

actual_df = add_population_column(actual_df)


# In[19]:


actual_df.head(12)


# In[ ]:


ID_COLS = ['CountryName',
           'RegionName',
           'Date']


# In[ ]:


# Add empty rows until end date, if needed, to simplify merge with teams predictions
new_rows = []
for g in actual_df.GeoID.unique():
    gdf = actual_df[actual_df.GeoID == g]
    last_known_date = gdf.Date.max()
    current_date = last_known_date + np.timedelta64(1, 'D')
    while current_date <= end_date:
        new_row = [gdf.CountryName.unique()[0],
                   gdf.RegionName.unique()[0],
                   current_date,
                   np.nan,
                   g,
                   np.nan,
                   np.nan,
                   gdf.Population.unique()[0]
                  ]
        new_rows.append(new_row)
        # Move to next day
        current_date = current_date + np.timedelta64(1, 'D')
# Add the new rows
if new_rows:
    new_rows_df = pd.DataFrame(new_rows, columns=actual_df.columns)
    # Append the new rows
    actual_df = actual_df.append(new_rows_df)
# Sort
actual_df.sort_values(by=ID_COLS, inplace=True, ignore_index=True)


# In[ ]:


actual_df.head(40)


# # Get historical data for 7 days moving average calculation
# In order to compute the 7 days moving average, we need to get the historical true new cases for the last 7 days before start date

# In[ ]:


ma_df = actual_df[actual_df["Date"] < start_date]
ma_df = ma_df[["CountryName", "RegionName", "Date", "ActualDailyNewCases"]]
ma_df = ma_df.rename(columns={"ActualDailyNewCases": "PredictedDailyNewCases"})
ma_df.head()


# # Run the predictions
# Evaluate some example submissions.  
# __NOTE: Please run the corresponding example notebooks first in order to train the models that are used in this section.__

# In[ ]:


IP_FILE = "predictions/robojudge_test_scenario.csv"
predictions = {}


# In[ ]:


from covid_xprize.validation.scenario_generator import generate_scenario

countries = None
scenario_df = generate_scenario(start_date_str, end_date_str, latest_df, countries, scenario="Freeze")
# Remove countries that weren't there at the beginning of the challenge or duplicates
ignore_countries = ["Tonga", "United States Virgin Islands"]
scenario_df = scenario_df[scenario_df.CountryName.isin(ignore_countries) == False]
# IP_FILE = "covid_xprize/validation/data/2020-09-30_historical_ip.csv"
scenario_df.to_csv(IP_FILE, index=False)


# ## Linear

# In[ ]:


# Check a model has been trained
if not os.path.isfile("covid_xprize/examples/predictors/linear/models/model.pkl"):
    print("ERROR: Please run the notebook in 'covid_xprize/examples/predictors/linear' in order to train a model!")


# In[ ]:


linear_output_file = "covid_xprize/examples/predictors/linear/predictions/robojudge_test.csv"


# In[ ]:


get_ipython().system('python covid_xprize/examples/predictors/linear/predict.py -s {start_date_str} -e {end_date_str} -ip {IP_FILE} -o {linear_output_file}')


# In[ ]:


predictions["Linear"] = linear_output_file


# ## LSTM

# In[ ]:


# Check a model has been trained
if not os.path.isfile("covid_xprize/examples/predictors/lstm/models/trained_model_weights.h5"):
    print("ERROR: Please run the notebook in 'covid_xprize/examples/predictors/lstm' in order to train a model!")


# In[ ]:


lstm_output_file = "covid_xprize/examples/predictors/lstm/predictions/robojudge_test.csv"


# In[ ]:


get_ipython().system('python covid_xprize/examples/predictors/lstm/predict.py -s {start_date_str} -e {end_date_str} -ip {IP_FILE} -o {lstm_output_file}')


# In[ ]:


predictions["LSTM"] = lstm_output_file


# # Get predictions from submissions

# In[ ]:


def get_predictions_from_file(predictor_name, predictions_file, ma_df):
    preds_df = pd.read_csv(predictions_file,
                           parse_dates=['Date'],
                           encoding="ISO-8859-1",
                           error_bad_lines=False)
    preds_df["RegionName"] = preds_df["RegionName"].fillna("")
    preds_df["PredictorName"] = predictor_name
    preds_df["Prediction"] = True
    
    # Append the true number of cases before start date
    ma_df["PredictorName"] = predictor_name
    ma_df["Prediction"] = False
    preds_df = ma_df.append(preds_df, ignore_index=True)

    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    # np.where usage: if A then B else C
    preds_df["GeoID"] = np.where(preds_df["RegionName"] == '',
                                 preds_df["CountryName"],
                                 preds_df["CountryName"] + ' / ' + preds_df["RegionName"])
    # Sort
    preds_df.sort_values(by=["GeoID","Date"], inplace=True)
    # Compute the 7 days moving average for PredictedDailyNewCases
    preds_df["PredictedDailyNewCases7DMA"] = preds_df.groupby(
        "GeoID")['PredictedDailyNewCases'].rolling(
        WINDOW_SIZE, center=False).mean().reset_index(0, drop=True)

    # Put PredictorName first
    preds_df = preds_df[["PredictorName"] + [col for col in preds_df.columns if col != "PredictorName"]]
    return preds_df


# In[ ]:


test_predictor_name = "Linear"
temp_df = get_predictions_from_file(test_predictor_name, predictions[test_predictor_name], ma_df.copy())
temp_df.head(12)


# In[ ]:


actual_df.head(8)


# In[ ]:


from covid_xprize.validation.predictor_validation import validate_submission

ranking_df = pd.DataFrame()
for predictor_name, predictions_file in predictions.items():
    print(f"Getting {predictor_name}'s predictions from: {predictions_file}")
    errors = validate_submission(start_date_str, end_date_str, IP_FILE, predictions_file)
    if not errors:
        preds_df = get_predictions_from_file(predictor_name, predictions_file, ma_df)
        merged_df = actual_df.merge(preds_df, on=['CountryName', 'RegionName', 'Date', 'GeoID'], how='left')
        ranking_df = ranking_df.append(merged_df)
    else:
        print(f"Predictor {predictor_name} did not submit valid predictions! Please check its errors:")
        print(errors)


# In[ ]:


# Keep only predictions (either Prediction == True) or on or after start_date
ranking_df = ranking_df[ranking_df["Date"] >= start_date]


# In[ ]:


from covid_xprize.scoring.predictor_scoring import add_predictor_performance_columns

# Compute performance measures
ranking_df = add_predictor_performance_columns(ranking_df)


# In[ ]:


ranking_df.head(4*2)


# In[ ]:


ranking_df[(ranking_df.CountryName == "United States") &
           (ranking_df.Date == '2020-12-02')]


# In[ ]:


# Save to file
# ranking_df.to_csv("/Users/m_754337/workspace/esp-demo/xprize/tests/fixtures/ranking.csv", index=False)


# # Ranking

# In[ ]:


# Filter to get the metrics for the final day
last_known_date = ranking_df[ranking_df["Cumul-7DMA-MAE-per-100K"].notnull()].Date.max()
leaderboard_df = ranking_df[ranking_df.Date == last_known_date][[
        "CountryName",
        "RegionName",
        "PredictorName",
        "Cumul-7DMA-MAE-per-100K",
        "PredictorRank"]]


# ## Global

# In[ ]:


leaderboard_df.groupby(["PredictorName"]).mean()     .sort_values(by=["Cumul-7DMA-MAE-per-100K"]).reset_index()


# ## Countries

# In[ ]:


leaderboard_df.sort_values(by=['CountryName', 'RegionName'])


# ## Specific country

# In[ ]:


leaderboard_df[(leaderboard_df.CountryName == "Italy") &
               (leaderboard_df.RegionName == "")]


# In[ ]:


ranking_df[ranking_df.CountryName == "Italy"]


# ## Specific region

# In[ ]:


leaderboard_df[(leaderboard_df.CountryName == "United States") &
               (leaderboard_df.RegionName == "California")]


# ## Continent

# In[ ]:


NORTH_AMERICA = ["Canada", "United States", "Mexico"]


# In[ ]:


leaderboard_df[(leaderboard_df.CountryName.isin(NORTH_AMERICA)) &
               (leaderboard_df.RegionName == "")].groupby('PredictorName').mean() \
    .sort_values(by=["Cumul-7DMA-MAE-per-100K"]).reset_index()


# In[ ]:


leaderboard_df[(leaderboard_df.CountryName.isin(NORTH_AMERICA)) &
               (leaderboard_df.RegionName == "")]


# # Plots

# In[ ]:


ALL_GEO = "Overall"
DEFAULT_GEO = ALL_GEO


# ## Prediction vs actual

# In[ ]:


predictor_names = list(ranking_df.PredictorName.dropna().unique())
geoid_names = list(ranking_df.GeoID.unique())


# ## Filter by country

# In[ ]:


all_df = ranking_df.groupby(["PredictorName", "Date"])[["GeoID", "PredictorName", "PredictedDailyNewCases7DMA"]].sum().     sort_values(by=["PredictorName", "Date"]).reset_index()
all_df


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure(layout=dict(title=dict(text=f"{DEFAULT_GEO} Daily New Cases 7-day Average ",
                                       y=0.9,
                                       x=0.5,
                                       xanchor='center',
                                       yanchor='top'
                                       ),
                             plot_bgcolor='#f2f2f2',
                             xaxis_title="Date",
                             yaxis_title="Daily new cases 7-day average"
                             ))

# Keep track of trace visibility by geo ID name
geoid_plot_names = []

all_df = ranking_df.groupby(["PredictorName", "Date"])[["GeoID", "PredictorName", "PredictedDailyNewCases7DMA"]].sum().     sort_values(by=["PredictorName", "Date"]).reset_index()

# Add 1 trace per predictor, for all geos
for predictor_name in predictor_names:
    all_geo_df = all_df[all_df.PredictorName == predictor_name]
    fig.add_trace(go.Scatter(x=all_geo_df.Date,
                             y=all_geo_df.PredictedDailyNewCases7DMA,
                             name=predictor_name,
                             visible=(ALL_GEO == DEFAULT_GEO))
                 )
    geoid_plot_names.append(ALL_GEO)

# Add 1 trace per predictor, per geo id
for predictor_name in predictor_names:
    for geoid_name in geoid_names:
        pred_geoid_df = ranking_df[(ranking_df.GeoID == geoid_name) &
                                   (ranking_df.PredictorName == predictor_name)]
        fig.add_trace(go.Scatter(x=pred_geoid_df.Date,
                                 y=pred_geoid_df.PredictedDailyNewCases7DMA,
                                 name=predictor_name,
                                 visible=(geoid_name == DEFAULT_GEO))
                     )
        geoid_plot_names.append(geoid_name)

# For each geo
# Add 1 trace for the true number of cases
for geoid_name in geoid_names:
    # Only plot data that is known, for ground truth
    last_known_date = actual_df[actual_df.ActualDailyNewCases7DMA.notnull()].Date.max()
    geo_actual_df = actual_df[(actual_df.GeoID == geoid_name) &
                                  (actual_df.Date >= start_date)]
    fig.add_trace(go.Scatter(x=geo_actual_df[actual_df.Date <= last_known_date].Date,
                             y=geo_actual_df[actual_df.Date <= last_known_date].ActualDailyNewCases7DMA,
                             name="Ground Truth",
                             visible= (geoid_name == DEFAULT_GEO),
                             line=dict(color='orange', width=4, dash='dash'))
                  )
    geoid_plot_names.append(geoid_name)
    
# Add 1 trace for the overall ground truth
overall_actual_df = actual_df[actual_df.Date >= start_date].groupby(["Date"])[["GeoID", "ActualDailyNewCases7DMA"]].sum().     sort_values(by=["Date"]).reset_index()
overall_last_known_date = overall_actual_df[overall_actual_df.ActualDailyNewCases7DMA > 0].Date.max()
fig.add_trace(go.Scatter(x=overall_actual_df[overall_actual_df.Date <= overall_last_known_date].Date,
                         y=overall_actual_df[overall_actual_df.Date <= overall_last_known_date].ActualDailyNewCases7DMA,
                         name="Ground Truth",
                         visible= (ALL_GEO == DEFAULT_GEO),
                         line=dict(color='orange', width=4, dash='dash'))
                  )
geoid_plot_names.append(geoid_name)

# Format x axis
fig.update_xaxes(
dtick="D1",  # Means 1 day
tickformat="%d\n%b")

# Filter
buttons=[]
for geoid_name in ([ALL_GEO] + geoid_names):
    buttons.append(dict(method='update',
                        label=geoid_name,
                        args = [{'visible': [geoid_name==r for r in geoid_plot_names]},
                                {'title': f"{geoid_name} Daily New Cases 7-day Average "}]))
fig.update_layout(showlegend=True,
                  updatemenus=[{"buttons": buttons,
                                "direction": "down",
                                "active": ([ALL_GEO] + geoid_names).index(DEFAULT_GEO),
                                "showactive": True,
                                "x": 0.1,
                                "y": 1.15}])

fig.show()


# ## Rankings: by cumulative 7DMA error per 100K

# In[ ]:


ranking_fig = go.Figure(layout=dict(title=dict(text=f'{DEFAULT_GEO} submission rankings',
                                               y=0.9,
                                               x=0.5,
                                               xanchor='center',
                                               yanchor='top'
                                               ),
                                    plot_bgcolor='#f2f2f2',
                                    xaxis_title="Date",
                                    yaxis_title="MAE per 100K"
                                    ))

# Keep track of trace visibility by geo name
ranking_geoid_plot_names = []

all_df = ranking_df.groupby(["PredictorName", "Date"])[["GeoID", "PredictorName", "Cumul-7DMA-MAE-per-100K"]].mean().     sort_values(by=["PredictorName", "Date"]).reset_index()

# Add 1 trace per predictor, for all geos
for predictor_name in predictor_names:
    ranking_geoid_df = all_df[all_df.PredictorName == predictor_name]
    ranking_fig.add_trace(go.Scatter(x=ranking_geoid_df.Date,
                             y=ranking_geoid_df['Cumul-7DMA-MAE-per-100K'],
                             name=predictor_name,
                             visible=(ALL_GEO == DEFAULT_GEO))
                 )
    ranking_geoid_plot_names.append(ALL_GEO)


# Add 1 trace per predictor, per country
for predictor_name in predictor_names:
    for geoid_name in geoid_names:
        ranking_geoid_df = ranking_df[(ranking_df.GeoID == geoid_name) &
                                        (ranking_df.PredictorName == predictor_name)]
        ranking_fig.add_trace(go.Scatter(x=ranking_geoid_df.Date,
                                 y=ranking_geoid_df['Cumul-7DMA-MAE-per-100K'],
                                 name=predictor_name,
                                 visible= (geoid_name == DEFAULT_GEO))
                     )
        ranking_geoid_plot_names.append(geoid_name)

# Format x axis
ranking_fig.update_xaxes(
dtick="D1",  # Means 1 day
tickformat="%d\n%b")

# Filter
buttons=[]
for geoid_name in ([ALL_GEO] + geoid_names):
    buttons.append(dict(method='update',
                        label=geoid_name,
                        args = [{'visible': [geoid_name==r for r in ranking_geoid_plot_names]},
                                {'title': f'{geoid_name} submission rankings'}]))
ranking_fig.update_layout(showlegend=True,
                          updatemenus=[{"buttons": buttons,
                                        "direction": "down",
                                        "active": ([ALL_GEO] + geoid_names).index(DEFAULT_GEO),
                                        "showactive": True,
                                        "x": 0.1,
                                        "y": 1.15}])

ranking_fig.show()


# In[ ]:




