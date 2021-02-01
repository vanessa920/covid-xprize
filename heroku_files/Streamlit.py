import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from PIL import Image
import pickle

#preprocessing
import spacy
import re
import string


#intro
# st.sidebar.write("This is an application for predicting COVID cases around the country!")
# st.sidebar.button("Predict")


Title_html = """
    <style>
        .title h1{
          user-select: none;
          font-size: 43px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 600vw 600vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
        .reportview-container .main .block-container{
            padding-top: 3em;
        }
        body {
            background-color: white;
            background-position-y: -200px;
        }
        @media (max-width: 1800px) {
            body {
                background-position-x: -500px;
            }
        }
        .Widget.stTextArea, .Widget.stTextArea textarea {
        height: 586px;
        width: 400px;
        }
        h1{
            color: black
        }
        h2{
            color: black
        }
        p{
            color: black
        }
        .sidebar-content {
            width:25rem !important;
        }
        .Widget.stTextArea, .Widget.stTextArea textarea{
        }
        .sidebar.--collapsed .sidebar-content {
         margin-left: -25rem;
        }
        .streamlit-button.small-button {
        padding: .5rem 9.8rem;
        }
        .streamlit-button.primary-button {
        background-color: white;
        }
    </style> 
    
    <div>
        <h1>Welcome to the COVID Prediction App!</h1>
    </div>
    """
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

def get_total_dataframe(dataset):
    total_dataframe = pd.DataFrame({
    'Date':['Predicted Daily New Cases'],
    # 'Date':(dataset.iloc[0]['Date']),
    'Number of cases':(dataset.iloc[0]['PredictedDailyNewCases'])
    })
    return total_dataframe

country_total = get_total_dataframe(country_data)

if st.sidebar.checkbox("Show Analysis by Country", True, key=2):
    st.markdown("## **Country level analysis**")
    st.markdown("### Overall Predicted Daily New Cases in %s " % (select))
    if not st.checkbox('Hide Graph', False, key=1):
        country_total_graph = px.scatter(
        country_total, 
        x='Date',
        y='Number of cases',
        labels={'Number of cases':'Number of cases in %s' % (select)},
        color='Date')
        st.plotly_chart(country_total_graph)
