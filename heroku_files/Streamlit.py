import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from PIL import Image
import pickle

# #preprocessing
# import spacy
# import re
# import string


#intro
# st.sidebar.write("This is an application for predicting COVID cases around the country!")
# st.sidebar.button("Predict")

from HTML_snippets import Title_html


st.markdown(Title_html, unsafe_allow_html=True) #Title rendering

#Source: https://www.analyticsvidhya.com/blog/2020/10/create-interactive-dashboards-with-streamlit-and-python/

st.markdown("The dashboard will visualize the Covid-19 cases worldwide")
st.markdown("Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment. This app gives you the real-time predicted daily new cases of COVID-19")
st.sidebar.title("Visualization Selector")
st.sidebar.markdown("Select the Charts/Plots accordingly:")

# DATA_URL=('E:\Data science Projects\NIELIT project\covid_19_world.csv')
# For different use case where the data does not change often
# @st.cache(persist=True)


def load_data():
    data=pd.read_csv("./../2020-08-01_2020-08-04_predictions_example.csv")
    # data = pd.read_csv("https://github.com/OxCGRT/covid-policy-tracker/blob/master/data/OxCGRT_latest.csv")
    return data

df=load_data()
if st.checkbox('Show dataframe'):
    st.write(df)

# st.sidebar.checkbox("Show Analysis by Country", True, key=1)
select = st.sidebar.selectbox('Select a Country',df['CountryName'].unique())

#get the country selected in the selectbox
country_data = df[df['CountryName'] == select]

# select_status = st.sidebar.radio("Covid-19 patient's status", ('Confirmed',
# 'Active', 'Recovered', 'Deceased'))

# def get_total_dataframe(dataset):
#     total_dataframe = pd.DataFrame({
#     'Status':['Confirmed', 'Recovered', 'Deaths','Active'],
#     'Number of cases':(dataset.iloc[0]['confirmed'],
#     dataset.iloc[0]['recovered'], 
#     dataset.iloc[0]['deaths'],dataset.iloc[0]['active'])})
#     return total_dataframe

# def get_total_dataframe(dataset):
#     total_dataframe = pd.DataFrame({
#     'Date':['Predicted Daily New Cases'],
#     # 'Date':(dataset.iloc[0]['Date']),
#     'Number of cases':(dataset.iloc[0]['PredictedDailyNewCases'])
#     })
#     return total_dataframe

# country_total = get_total_dataframe(country_data)

if st.sidebar.checkbox("Show Analysis by Country", True, key=2):
    st.markdown("## **Country level analysis**")
    st.markdown("### Overall Predicted Daily New Cases in %s " % (select))
    if not st.checkbox('Hide Graph', False, key=1):
        country_total_graph = px.line(
            country_data,
            x='Date',
            y='PredictedDailyNewCases',
            labels={
                'PredictedDailyNewCases':'<b>Number of Cases (per 100k?)</b>',
                'Date':'<b>Date</b>'
            },
            title=f'<b>Overall Predicted Daily New Cases in {select}</b>')
            #color='Date')
        country_total_graph.update_layout(
            xaxis_tickformat="%b %d %Y",
            xaxis_nticks=len(list(country_data['Date'])),
            yaxis_range = [0,max(list(country_data['PredictedDailyNewCases']))]
        )
#         country_total_graph.update_xaxes(tickformat="%b %d %Y", nticks=len(list(country_data['Date'])))
        country_total_graph.update_yaxes(tick0 = 0)
        st.plotly_chart(country_total_graph)
        #st.write(country_data)
