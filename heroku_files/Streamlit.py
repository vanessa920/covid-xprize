import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime

from PIL import Image
import pickle


#intro
# st.sidebar.write("This is an application for predicting COVID cases around the country!")
# st.sidebar.button("Predict")

from HTML_snippets import Title_html
st.markdown(Title_html, unsafe_allow_html=True) #Title rendering

Body_html = """
  <style>
 img{
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%;
    #width: 300px;
    height: auto;
    #opacity:0.9;
}
   body{
    background-image: url(https://i.stack.imgur.com/HCfU2.png);
    #https://i.stack.imgur.com/9WYxT.png
    #opacity: 0.5;
}
</style>
"""
st.markdown(Body_html, unsafe_allow_html=True) #Body rendering

# st.markdown("The dashboard will visualize the Covid-19 cases worldwide")
st.markdown("Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment. This app gives you the real-time predicted daily new cases of COVID-19")
st.sidebar.title("Select a Country and NPI stringency level")
# st.sidebar.markdown("Select the Charts/Plots accordingly:")

# DATA_URL=('E:\Data science Projects\NIELIT project\covid_19_world.csv')
# For different use case where the data does not change often
# @st.cache(persist=True)


def load_data():
    data=pd.read_csv("2020-08-01_2020-08-04_predictions_example.csv")
    # data = pd.read_csv("https://github.com/OxCGRT/covid-policy-tracker/blob/master/data/OxCGRT_latest.csv")
    return data

df=load_data()
# if st.checkbox('Show Data'):
#     st.write(df)

# st.sidebar.checkbox("Show Analysis by Country", True, key=1)
select = st.sidebar.selectbox('Country',df['CountryName'].unique())

#get the country selected in the selectbox
country_data = df[df['CountryName'] == select]

#Removed checkbox
# if st.sidebar.checkbox("Show Analysis by Country", True, key=2):
# st.markdown("## **Country level analysis**")
st.markdown(f"## **Overall Predicted Daily New Cases in {select}**")

#Removed checkbox
# if not st.checkbox('Hide Graph', False, key=1):
country_total_graph = px.line(
    country_data,
    x='Date',
    y='PredictedDailyNewCases',
    labels={
        'PredictedDailyNewCases':'<b>Number of Cases (per 100k)</b>',
        'Date':'<b>Date</b>'
    },)
    # title=f'<b>Overall Predicted Daily New Cases in {select}</b>')
    #color='Date')
country_total_graph.update_layout(
    xaxis_tickformat="%b %d %Y",
    xaxis_nticks=len(list(country_data['Date'])),
    yaxis_range = [0,max(list(country_data['PredictedDailyNewCases']))]
)
country_total_graph.update_yaxes(tick0 = 0)
st.plotly_chart(country_total_graph)
#st.write(country_data)

def load_prescribe_df():
    data=pd.read_csv("heroku_files/all_2021q1_test_task.csv")
    return data

stringency = st.sidebar.slider('NPIs', 0, 9)
# prescribe_df = pd.read_csv("all_2021q1_test_task.csv")
prescribe_df = load_prescribe_df()
prescribe_df = prescribe_df[prescribe_df['CountryName'] == select] #select the country
prescribe_df = prescribe_df[pd.to_datetime(prescribe_df['Date']) >= datetime.datetime.today()] #select today and future dates
prescribe_df = prescribe_df[prescribe_df['PrescriptionIndex'] == stringency] #select the relevant prescription index
# st.write(prescribe_df)

# all_npis = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings',
# 'C5_Close public transport', 'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
# 'C8_International travel controls', 'H1_Public information campaigns', 'H2_Testing policy',
# 'H3_Contact tracing', 'H6_Facial Coverings', 'Date', 'CountryName', 'RegionName', 'PrescriptionIndex']

# `npis` is in reverse order of `all_npis` because of the way the matrix ends up when it's transposed

# npis = ['H6_Facial Coverings', 'H3_Contact tracing', 'H2_Testing policy',
# 'H1_Public information campaigns', 'C8_International travel controls',
# 'C7_Restrictions on internal movement', 'C6_Stay at home requirements',
# 'C5_Close public transport', 'C4_Restrictions on gatherings', 'C3_Cancel public events',
# 'C2_Workplace closing', 'C1_School closing']

first_date = datetime.datetime.today() - datetime.timedelta(days=1)
last_date = pd.to_datetime(prescribe_df['Date'].values[-1])
dates = [first_date + datetime.timedelta(days=x) for x in range((last_date-first_date).days + 2)] #adding the extra day for setting the color scale to 0-4

prescribe_df = np.transpose(np.array(prescribe_df.drop(axis=1, columns=['Date', 'CountryName', 'RegionName', 'PrescriptionIndex'])))
prescribe_df = np.concatenate(
    (np.transpose(np.array([[0,1,2,3,4,4,4,4,4,4,4,4]])), # this is here to make sure that the color scale actually goes from 0 to 4
    prescribe_df), axis=1)

fig2 = go.Figure(data=go.Heatmap(
        z=prescribe_df,
        x=dates,
        ygap=10,
        # colorscale=[#this isn't working properly and scales continuously if not all npi values are present (i.e. 4 is missing)
        #         [0,"grey"],
        #         [0.2,"grey"],
        #         [0.2,"blue"],
        #         [0.4,"blue"],
        #         [0.4,'green'],#can also use rgba()
        #         [0.6,'green'],
        #         [0.6,"yellow"],
        #         [0.8,"yellow"],
        #         [0.8,"red"],
        #         [1,"red"]
        # ],
        colorscale=[#this isn't working properly and scales continuously if not all npi values are present (i.e. 4 is missing)
                [0,"#e6eeff"],
                [0.2,"#e6eeff"],
                [0.2,"#99bbff"],
                [0.4,"#99bbff"],
                [0.4,"#4d88ff"],#can also use rgba()
                [0.6,"#4d88ff"],
                [0.6,"#0055ff"],
                [0.8,"#0055ff"],
                [0.8,"#003cb3"],
                [1,"#003cb3"]
        ],
        y=[
        'Mask Mandate',
        'Contact Tracing',
        'Testing Policy',
        'Public Information Campaign',
        'International Travel Restrictions',
        'Local Travel Restrictions',
        'Stay at Home Order',
        'Suspend Public Transportation',
        'Restrict Social Gatherings',
        'Cancel Public Events',
        'Close Workplaces',
        'Close Schools'],
        hovertemplate='%{x}<br>' + '%{y}<br>' + 'Restriction Level: %{z}'#see https://plotly.com/python/hover-text-and-formatting/
))
fig2.update_layout(
    title='Recommended Intervention Plan',
    xaxis_nticks=len(dates)//4, # make it more frequent without flipping it 90 degrees
    xaxis_tickformat='%d \n %B', # For more time formatting types, see: https://github.com/d3/d3-time-format/blob/master/README.md
    # ^^ from https://plotly.com/javascript/tick-formatting/
    hovermode="y",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    width=775,
    xaxis_range = [datetime.datetime.today(),last_date])
fig2.update_traces(
    showscale=False
)
st.plotly_chart(fig2)

Description = """
### Containment and closure policies

| Name | Description |
| --- | --- |
|Close Schools| Record closings of schools and universities |
|Close Workplaces| Record closings of workplaces | 
|Cancel Public Events| Record cancelling public events |  
|Restrict Social gatherings| Record limits on gatherings | 
|Suspend Public Transportation| Record closing of public transport |
|Stay at Home Order| Record orders to "shelter-in-place" and otherwise confine to the home | 
|Local Travel Restrictions| Record restrictions on internal movement between cities/regions | 
|International Travel Restrictions| Record restrictions on international travel <br/><br/>Note: this records policy for foreign travellers, not citizens |

### Health system policies

| Name | Description |
| --- | --- |
| Public Information Campaign | Record presence of public info campaigns | 
| Testing Policy | Record government policy on who has access to testing <br/><br/>Note: this records policies about testing for current infection (PCR tests) not testing for immunity (antibody test) | 
| Contact Tracing | Record government policy on contact tracing after a positive diagnosis <br/><br/>Note: we are looking for policies that would identify all people potentially exposed to Covid-19; voluntary bluetooth apps are unlikely to achieve this |
| Mask Mandate | Record policies on the use of facial coverings outside the home <br/> | 

"""
st.markdown(Description, unsafe_allow_html=True) #Body rendering

# put in a legend with the actual colors somehow. Maybe an SVG?
# make the hover text actually useful. definitely hide the x=..., y=... part. Keep z for now until something better can be put in
# put in plain english explanations of what the NPIs actually mean and a link to the official explanations.


