import pandas as pd

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

def write_solutions(prescriptions, country_name, region_name, start_date, output_csv):

#     df = pd.DataFrame(columns=["CountryName", "RegionName", "Date", "PrescriptionIndex"] + NPI_COLUMNS)
#     df.to_csv(output_csv, index=False)
    # cast to int
    for i in range(len(prescriptions)):

        df = pd.DataFrame(prescriptions[i], columns=NPI_COLUMNS)
        df[NPI_COLUMNS] = df[NPI_COLUMNS].astype('int32')
        df["Date"] = pd.date_range(start_date, periods=len(prescriptions[i]), freq='D')
        df["CountryName"] = country_name
        df["RegionName"] = region_name
        df["PrescriptionIndex"] = i

        df.to_csv(output_csv, mode="a", header=False, index=False)


#     def write_solutions(prescriptions, country_name, region_name, start_date, output_csv):
#     dfs = []

#     #= pd.DataFrame(columns=["CountryName", "RegionName", "Date", "PrescriptionIndex"] + NPI_COLUMNS)
#     #df.to_csv(output_csv)
#     # cast to int
#     for i in range(len(prescriptions)):

#         df_add = pd.DataFrame(prescriptions[i], columns=NPI_COLUMNS)

#         df["Date"] = pd.date_range(start_date, periods=len(prescriptions[i]), freq='D')
#         df["CountryName"] = country_name
#         df["RegionName"] = region_name
#         df["PrescriptionIndex"] = i

#         dfs.append(df)
#     big_df = pd.concat(dfs, axis=0)
#     big_df.to_csv(output_csv)
    
