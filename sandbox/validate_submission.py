#!/usr/bin/env python
# coding: utf-8

# In[1]:


from predictor_validation import validate_submission
import pandas as pd
# start_date_str = "2020-12-22"
# end_date_str = "2021-06-19"
# IP_FILE = "~/covid-xprize-robotasks-main/ips/live/20201220_080002_20201222_20210619_ips.csv"
# predictions_file = "~/work/predictions/20201220_080002_robojudge_live.csv"


# In[2]:


task_csv = pd.read_csv("~/covid-xprize-robotasks-main/tasks/tasks.csv")
start_date_str = task_csv["StartDate"][0]
end_date_str = task_csv["EndDate"][0]
IP_FILE = task_csv["IpFile"][0]
predictions_file = task_csv["OutputFile"][0]
print(IP_FILE, predictions_file)


# In[3]:


validate_submission(start_date_str, end_date_str, IP_FILE, predictions_file)


# In[ ]:


# sanity checks
# doesn't go over the total population
# run it with 0s for all NPIs from now on, and see whether it gives ridiculous values


# In[6]:


df = pd.read_csv("example_ips.csv", 
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)


# In[8]:


df.columns


# In[9]:


npi_cols = ['C1_School closing',
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


# In[10]:


for col in npi_cols:
    df[col] = 0.0


# In[11]:


#df.to_csv("do_nothing_ips.csv")


# In[ ]:


os.mkdir()


# In[75]:


get_ipython().system('python predict.py -s {start_date_str} -e {end_date_str} -ip do_nothing_ips.csv -o dumpsterfire/do_nothing_outputs.csv')


# In[70]:


start_date_str = df["StartDate"][0]
end_date_str = df["EndDate"][0]
IP_FILE = "do_nothing_ips.csv"
predictions_file = df["OutputFile"][0]
print(IP_FILE, predictions_file)


# In[76]:


validate_submission(start_date_str, end_date_str, "do_nothing_ips.csv", "dumpsterfire/do_nothing_outputs.csv")


# In[ ]:





# In[ ]:





# In[77]:


import numpy as np

country_info = pd.read_csv("data/Additional_Context_Data_Global.csv")
country_info["RegionName"] = np.NaN
country_info = country_info.astype({'RegionName': 'object'})

df_2 = pd.read_csv("dumpsterfire/do_nothing_outputs.csv")

df_2['GeoID'] = df_2['CountryName'] + '__' + df_2['RegionName'].astype(str)

summed = df_2.groupby(['CountryName', 'RegionName'], dropna=False)["PredictedDailyNewCases"].sum().reset_index()

country_info_less = country_info[["CountryName", "RegionName", "Population"]]

test_info = summed.merge(country_info_less, how='left', on=['CountryName', 'RegionName'])

test_info["prop"] = test_info["PredictedDailyNewCases"]/test_info["Population"]

test_info.sort_values(by="prop", ascending=False).head(20)


# In[78]:


test_info.to_csv("check_this.csv")


# In[11]:


import numpy as np


# In[25]:


country_info = pd.read_csv("data/Additional_Context_Data_Global.csv")
country_info["RegionName"] = np.NaN
country_info = country_info.astype({'RegionName': 'object'})


# In[26]:


df_2 = pd.read_csv("dumpsterfire/do_nothing_outputs.csv")


# In[14]:


df_2['GeoID'] = df_2['CountryName'] + '__' + df_2['RegionName'].astype(str)


# In[55]:


summed = df_2.groupby(['CountryName', 'RegionName'], dropna=False)["PredictedDailyNewCases"].sum().reset_index()


# In[56]:


country_info_less = country_info[["CountryName", "RegionName", "Population"]]


# In[58]:


test_info = summed.merge(country_info_less, how='left', on=['CountryName', 'RegionName'])


# In[62]:


test_info["prop"] = test_info["PredictedDailyNewCases"]/test_info["Population"]


# In[ ]:


bad_data = ["Turkey", "Uruguay", "Belize", "Denmark", "Estonia", "Lithuania", "Panama", "Cyprus", "Latvia", "Malaysia", "Bermuda"]


# In[65]:


test_info.sort_values(by="prop", ascending=False).head(20)


# In[59]:


test_info[test_info["PredictedDailyNewCases"]>test_info["Population"]]


# In[54]:


test = summed.merge(country_info[["CountryName", "RegionName", "Population"]], how='left', on=['CountryName', 'RegionName'])


# In[27]:


df_2 = df_2.merge(country_info, how='left', on=["CountryName", "RegionName"], suffixes=[None, "_jh"])
# perhaps use region name and region code to merge?
# currently inner merge - must be left (and should fix region clashes)


# In[28]:


a = df_2[["CountryName", "RegionName", "Date", "PredictedDailyNewCases", "Population"]]
#a.to_csv("check_this.csv")


# In[23]:


b = a.merge(a.groupby(["CountryName", "RegionName"], dropna=False)["PredictedDailyNewCases"].sum(), how="left", on=["CountryName", "RegionName"])


# In[24]:


b


# In[ ]:


merged_df = merged_df.merge(country_info, how='left', on=["CountryName", "RegionName"], suffixes=[None, "_jh"])


# In[ ]:


countries_regions


# In[ ]:


my_predicts.groupby()

