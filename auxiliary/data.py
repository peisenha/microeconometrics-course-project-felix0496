import pandas as pd

def get_df_compulsory_school_attendance():

    df = pd.DataFrame({'STATE': list(range(1, 57))})

    school_leaving_16 = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, \
                         17, 18, 19, 20, 22, 24, 25, 26, 27, 29,  \
                         30, 31, 33, 34, 36, 37, 44, 46, 50, 54, 55]

    df['1960'] = [16 if x in school_leaving_16 else pd.NA for x in df['STATE']]
    df['1970'] = [16 if x in school_leaving_16 else pd.NA for x in df['STATE']]
    df['1980'] = [16 if x in school_leaving_16 else pd.NA for x in df['STATE']]

    df.loc[df['STATE'] == 15, ['1960', '1970', '1980']] = [16, 18, 18]
    df.loc[df['STATE'] == 23, ['1960', '1970', '1980']] = [15, 17, 17]
    df.loc[df['STATE'] == 32, ['1960', '1970', '1980']] = [17, 17, 17] 
    df.loc[df['STATE'] == 35, ['1960', '1970', '1980']] = [16, 17, 17] 
    df.loc[df['STATE'] == 38, ['1960', '1970', '1980']] = [17, 16, 16] 
    df.loc[df['STATE'] == 39, ['1960', '1970', '1980']] = [18, 18, 18] 
    df.loc[df['STATE'] == 40, ['1960', '1970', '1980']] = [18, 18, 18]
    df.loc[df['STATE'] == 41, ['1960', '1970', '1980']] = [18, 18, 18]
    df.loc[df['STATE'] == 42, ['1960', '1970', '1980']] = [17, 17, 17]
    df.loc[df['STATE'] == 45, ['1960', '1970', '1980']] = [pd.NA, 16, 16]
    df.loc[df['STATE'] == 47, ['1960', '1970', '1980']] = [17, 17, 16]
    df.loc[df['STATE'] == 48, ['1960', '1970', '1980']] = [16, 17, 17]
    df.loc[df['STATE'] == 49, ['1960', '1970', '1980']] = [18, 18, 18]
    df.loc[df['STATE'] == 51, ['1960', '1970', '1980']] = [16, 17, 17]
    df.loc[df['STATE'] == 53, ['1960', '1970', '1980']] = [16, 18, 18]
    df.loc[df['STATE'] == 56, ['1960', '1970', '1980']] = [17, 16, 16]

    return df