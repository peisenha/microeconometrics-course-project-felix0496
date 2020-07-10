import numpy as np
import pandas as pd

FILE_PATH_CENSUS80_EXTRACT = "data/QOB.txt"
FILE_PATH_FULL_CENSUS7080 = "data/NEW7080.dta"

def get_df_census80_extract():

    cols = [0, 1, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 23, 24, 26]

    cols_names = ['AGE', 'AGEQ', 'EDUC', 'ENOCENT', 'ESOCENT', 'LWKLYWGE', \
                'MARRIED', 'MIDATL', 'MT', 'NEWENG', 'CENSUS', 'STATE', 'QOB', \
                'RACE', 'SMSA', 'SOATL', 'WNOCENT', 'WSOCENT', 'YOB']

    df = pd.read_csv(FILE_PATH_CENSUS80_EXTRACT, sep = " ", usecols = cols, names = cols_names)

    # correct AGEQ
    df.loc[df['CENSUS'] == 80, 'AGEQ'] = df['AGEQ'] - 1900

    return df

def get_df_full_census7080():

    cols = ['v1', 'v2', 'v4', 'v5', 'v6', 'v9', 'v10', 'v11', 'v12', 'v13', 'v16', \
            'v17', 'v18', 'v19', 'v20', 'v21', 'v24', 'v25', 'v27']

    cols_names = ['AGE', 'AGEQ', 'EDUC', 'ENOCENT', 'ESOCENT', 'LWKLYWGE', \
                'MARRIED', 'MIDATL', 'MT', 'NEWENG', 'CENSUS', 'STATE', 'QOB', \
                'RACE', 'SMSA', 'SOATL', 'STATE', 'WNOCENT', 'WSOCENT', 'YOB']
    
    df = pd.read_stata(FILE_PATH_FULL_CENSUS7080, columns = cols)

    df = df.rename(columns = dict(zip(cols, cols_names)))

    return df

def add_quarter_of_birth_dummies(df):

    df['DUMMY_QOB_1'] = [1 if x == 1 else 0 for x in df['QOB']]
    df['DUMMY_QOB_2'] = [1 if x == 2 else 0 for x in df['QOB']]
    df['DUMMY_QOB_3'] = [1 if x == 3 else 0 for x in df['QOB']]

    return df

def add_year_of_birth_dummies(df):

    df['DUMMY_YOB_0'] = [1 if x % 10 == 0 else 0 for x in df['YOB']]
    df['DUMMY_YOB_1'] = [1 if x % 10 == 1 else 0 for x in df['YOB']]
    df['DUMMY_YOB_2'] = [1 if x % 10 == 2 else 0 for x in df['YOB']]
    df['DUMMY_YOB_3'] = [1 if x % 10 == 3 else 0 for x in df['YOB']]
    df['DUMMY_YOB_4'] = [1 if x % 10 == 4 else 0 for x in df['YOB']]
    df['DUMMY_YOB_5'] = [1 if x % 10 == 5 else 0 for x in df['YOB']]
    df['DUMMY_YOB_6'] = [1 if x % 10 == 6 else 0 for x in df['YOB']]
    df['DUMMY_YOB_7'] = [1 if x % 10 == 7 else 0 for x in df['YOB']]
    df['DUMMY_YOB_8'] = [1 if x % 10 == 8 else 0 for x in df['YOB']]
    df['DUMMY_YOB_9'] = [1 if x % 10 == 9 else 0 for x in df['YOB']]

    return df

def add_state_of_birth_dummies(df):

    for i in set(df['STATE']):

        column_name = f'DUMMY_STATE_{i}'
        df[column_name] = [1 if x == i else 0 for x in df['STATE']]

    return df

def add_education_dummies(df):

    # dummy variable high school degree (12 or more years of education)
    df['DUMMY_HIGH_SCHOOL'] = [1 if x >= 12 else 0 for x in df['EDUC']]

    # dummy variable college degree (16 or more years of education)
    df['DUMMY_COLLEGE'] = [1 if x >= 16 else 0 for x in df['EDUC']]

    # dummy variable master's degree (18 or more years of education)
    df['DUMMY_MASTER'] = [1 if x >= 18 else 0 for x in df['EDUC']]

    # dummy variable doctoral degree (20 or more years of education)
    df['DUMMY_DOCTOR'] = [1 if x >= 20 else 0 for x in df['EDUC']]

    return df

def add_detrended_educational_variables(df, educ_vars=['EDUC']):

    for ev in educ_vars:

        mean_ev = df.groupby(['YOB', 'QOB'])[ev].mean().to_frame()
        mean_ev['MV_AVG'] = two_sided_moving_average(mean_ev.values)

        for yob in set(df['YOB']):
            for qob in set(df['QOB']):
                df.loc[(df['YOB'] == yob) & (df['QOB'] == qob), f'MV_AVG_{ev}'] = mean_ev.loc[(yob, qob), 'MV_AVG']

        df[f'DTRND_{ev}'] = df[ev] - df[f'MV_AVG_{ev}']

    return df

def two_sided_moving_average(x):

    ma = np.full_like(x, np.nan)

    for i in range(2, len(x) - 2):
        ma[i] = (x[i - 2] + x[i - 1] + x[i + 1] + x[i + 2]) / 4

    return ma    
