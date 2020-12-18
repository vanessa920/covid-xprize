import pickle
import os
import urllib.request
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def run_model(start_date: str,
            end_date: str,
            path_to_ips_file: str,
            output_file_path):
    """
    This function actually makes the predictions
    Does NOT do anything with the inputs. Fix this
    """
    # Main source for the training data
    DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
    # Local files
    data_path = 'covid_xprize/examples/predictors/ryan_predictor/data'
    DATA_FILE = data_path + '/OxCGRT_latest.csv'

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    df = pd.read_csv(DATA_FILE, 
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     dtype={"RegionName": str,
                            "RegionCode": str},
                     error_bad_lines=False)

    HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-07-31")
    df = df[df.Date <= HYPOTHETICAL_SUBMISSION_DATE]

    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)


    # Add new cases column
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)

    # Keep only columns of interest
    id_cols = ['CountryName',
               'RegionName',
               'GeoID',
               'Date']
    cases_col = ['NewCases']
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
    df = df[id_cols + cases_col + npi_cols]

    # Fill any missing case values by interpolation and setting NaNs to 0
    df.update(df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))

    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in npi_cols:
        df.update(df.groupby('GeoID')[npi_col].ffill().fillna(0))

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    scores_df = pd.DataFrame(columns = ['Country', 'TrainMAE', 'TestMAE'])

    # Helpful function to compute mae
    def mae(pred, true):
        return np.mean(np.abs(pred - true))

    do_not_scale_list = ['India', 'Mauritania', 'Philippines', 'Costa Rica']
    forest_list = ['Italy', 'Egypt', 'Iraq', 'Singapore', 'Poland', 'Pakistan'
        'Germany', 'Peru', 'Central African Republic', 'Guinea', 'Palestine',
        'France', 'Ecuador', 'Tanzania', 'Kyrgyz Republic']
    # The models for these countries were found to perform significantly better using:
    # - unscaled data for a linear regression
    # and 
    # - random forest

    scaler = StandardScaler()
    for country in df['CountryName'].unique().tolist():

        country_df = df[df['CountryName'] == country]

        # Set number of past days to use to make predictions
        nb_lookback_days = 30

        # Create training data across all countries for predicting one day ahead
        X_cols = cases_col + npi_cols
        y_col = cases_col
        X_samples = []
        y_samples = []
        geo_ids = country_df.GeoID.unique()
        for g in geo_ids:
            gdf = country_df[country_df.GeoID == g]
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
                y_sample = all_case_data[d + 1]
                X_samples.append(X_sample)
                y_samples.append(y_sample)

        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples).flatten()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_samples, y_samples, test_size=0.2, random_state=42)

        if country in do_not_scale_list:
            model = Lasso(alpha=0.1, precompute=True, max_iter=10000, positive=True, selection='random')
            model.fit(X_train, y_train)
        elif country in forest_list:
            model = RandomForestRegressor(max_depth=2, random_state=0)
            model.fit(X_train, y_train)
        else:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = Lasso(alpha=0.1, precompute=True, max_iter=10000, positive=True, selection='random')
            model.fit(X_train, y_train)

        # Evaluate model
        train_preds = model.predict(X_train)
        train_preds = np.maximum(train_preds, 0) # Don't predict negative cases
    #     print('Train MAE:', mae(train_preds, y_train))


        test_preds = model.predict(X_test)
        test_preds = np.maximum(test_preds, 0) # Don't predict negative cases
    #     print('Test MAE:', mae(test_preds, y_test))

        score_df = pd.DataFrame([[country,
                                  mae(train_preds, y_train),
                                  mae(test_preds, y_test)]],
                                columns=['Country', 'TrainMAE', 'TestMAE'])
        scores_df = scores_df.append(score_df)

    og_df = pd.read_csv(DATA_FILE, 
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     dtype={"RegionName": str,
                            "RegionCode": str},
                     error_bad_lines=False)
    og_df['GeoID'] = og_df['CountryName'] + '__' + og_df['RegionName'].astype(str)
    geoid_cases = og_df.groupby('GeoID').agg({'ConfirmedCases':np.median}).reset_index()
    geoid_cases = geoid_cases.merge(og_df[['GeoID','CountryName']], how='left', left_on='GeoID', right_on='GeoID')
    geoid_cases = geoid_cases.groupby('CountryName').agg({'ConfirmedCases':np.sum}).reset_index()

#     scores_df = scores_df.merge(geoid_cases, how='left', left_on='Country', right_on='CountryName').drop(['CountryName'], axis=1)

#     scores_df['TrainMPE'] = 100*scores_df['TrainMAE']/scores_df['ConfirmedCases']
#     scores_df['TestMPE'] = 100*scores_df['TestMAE']/scores_df['ConfirmedCases']

#     scores_df.sort_values(by='TestMPE').reset_index().to_csv('case_pred_errors_as_percent.csv', index=False)

#     scores_df = scores_df.sort_values(by='TestMPE').reset_index()