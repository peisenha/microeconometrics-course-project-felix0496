from collections import OrderedDict
import pandas as pd
import statsmodels.formula.api as smf

from . import data_helper

def get_regression_results_educational_variables(educ_vars, cohorts):

    results = []

    for ev in educ_vars:
        for chrt_name, chrt in cohorts:
            results.append({'var': ev,
                            'cohort': chrt_name,
                            'mean': chrt[ev].mean(),
                            'ols': smf.ols(formula = f'DTRND_{ev} ~ DUMMY_QOB_1 + DUMMY_QOB_2 + DUMMY_QOB_3', data = chrt).fit()})
    
    return results

def get_regression_results_ols_tls(df, state_of_birth_dummies = False, race = True):

    # add dummies for quarter and year of birth
    df = data_helper.add_quarter_of_birth_dummies(df)
    df = data_helper.add_year_of_birth_dummies(df)
    
    if state_of_birth_dummies:
        df = data_helper.add_state_of_birth_dummies(df)
        state_lst = set(df['STATE'])
        state_lst.remove(1)

    # add AGESQ age squared
    df['AGESQ'] = df['AGEQ'].pow(2)

    # regression (1) OLS
    formula_1 = 'LWKLYWGE ~ EDUC + '
    formula_1 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)])

    if state_of_birth_dummies:
        formula_1 += ' + '
        formula_1 += ' + '.join([f'DUMMY_STATE_{i}' for i in state_lst])

    ols_1 = smf.ols(formula = formula_1, data = df).fit()

    # regression (2) TSLS
    formula_1st_stage_2 = 'EDUC ~ ' 
    formula_1st_stage_2 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)]) 
    formula_1st_stage_2 += ' + '
    formula_1st_stage_2 += ' + '.join([f'DUMMY_YOB_{i} : DUMMY_QOB_{j}' for j in range(1,4) for i in range(0, 10)])

    if state_of_birth_dummies:
        formula_1st_stage_2 += ' + '
        formula_1st_stage_2 += ' + '.join([f'DUMMY_STATE_{i}' for i in state_lst])
        formula_1st_stage_2 += ' + '
        formula_1st_stage_2 += ' + '.join([f'DUMMY_STATE_{i} : DUMMY_QOB_{j}' for j in range(1,4) for i in state_lst])

    df['EDUC_pred_2'] = smf.ols(formula = formula_1st_stage_2, data = df).fit().predict()

    formula_2nd_stage_2 = 'LWKLYWGE ~ EDUC_pred_2 +' 
    formula_2nd_stage_2 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)]) 
    
    if state_of_birth_dummies:
        formula_2nd_stage_2 += ' + '
        formula_2nd_stage_2 += ' + '.join([f'DUMMY_STATE_{i}' for i in state_lst])

    tsls_2 = smf.ols(formula = formula_2nd_stage_2, data = df).fit()

    # regression (3) OLS
    formula_3 = 'LWKLYWGE ~ EDUC + AGEQ + AGESQ + '
    formula_3 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)])

    if state_of_birth_dummies:
        formula_3 += ' + '
        formula_3 += ' + '.join([f'DUMMY_STATE_{i}' for i in state_lst])

    ols_3 = smf.ols(formula = formula_3, data = df).fit()

    # regression (4) TSLS
    formula_1st_stage_4 = 'EDUC ~ AGEQ + AGESQ + ' 
    formula_1st_stage_4 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)]) 
    formula_1st_stage_4 += ' + '
    formula_1st_stage_4 += ' + '.join([f'DUMMY_YOB_{i} : DUMMY_QOB_{j}' for j in range(1,4) for i in range(0, 10)])

    if state_of_birth_dummies:
        formula_1st_stage_4 += ' + '
        formula_1st_stage_4 += ' + '.join([f'DUMMY_STATE_{i}' for i in state_lst])
        formula_1st_stage_4 += ' + '
        formula_1st_stage_4 += ' + '.join([f'DUMMY_STATE_{i} : DUMMY_QOB_{j}' for j in range(1,4) for i in state_lst])

    df['EDUC_pred_4'] = smf.ols(formula = formula_1st_stage_4, data = df).fit().predict()

    formula_2nd_stage_4 = 'LWKLYWGE ~ EDUC_pred_4 + AGEQ + AGESQ + ' 
    formula_2nd_stage_4 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)]) 

    if state_of_birth_dummies:
        formula_2nd_stage_4 += ' + '
        formula_2nd_stage_4 += ' + '.join([f'DUMMY_STATE_{i}' for i in state_lst])

    tsls_4 = smf.ols(formula = formula_2nd_stage_4, data = df).fit()

    # regression (5) OLS
    formula_5 = 'LWKLYWGE ~ EDUC + MARRIED + SMSA + NEWENG + MIDATL + '
    formula_5 += 'ENOCENT + WNOCENT + SOATL + ESOCENT + WSOCENT + MT + '
    formula_5 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)])

    if race:
        formula_5 += ' + RACE'

    if state_of_birth_dummies:
        formula_5 += ' + '
        formula_5 += ' + '.join([f'DUMMY_STATE_{i}' for i in state_lst])

    ols_5 = smf.ols(formula = formula_5, data = df).fit()
    
    # regression (6) TSLS
    formula_1st_stage_6 = 'EDUC ~ MARRIED + SMSA + NEWENG + MIDATL + '
    formula_1st_stage_6 += 'ENOCENT + WNOCENT + SOATL + ESOCENT + WSOCENT + MT + ' 
    formula_1st_stage_6 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)]) 
    formula_1st_stage_6 += ' + '
    formula_1st_stage_6 += ' + '.join([f'DUMMY_YOB_{i} : DUMMY_QOB_{j}' for j in range(1,4) for i in range(0, 10)])

    if race:
        formula_1st_stage_6 += ' + RACE'

    if state_of_birth_dummies:
        formula_1st_stage_6 += ' + '
        formula_1st_stage_6 += ' + '.join([f'DUMMY_STATE_{i}' for i in state_lst])
        formula_1st_stage_6 += ' + '
        formula_1st_stage_6 += ' + '.join([f'DUMMY_STATE_{i} : DUMMY_QOB_{j}' for j in range(1,4) for i in state_lst])

    df['EDUC_pred_6'] = smf.ols(formula = formula_1st_stage_6, data = df).fit().predict()

    formula_2nd_stage_6 = 'LWKLYWGE ~ EDUC_pred_6 + MARRIED + SMSA + NEWENG + MIDATL + '
    formula_2nd_stage_6 += 'ENOCENT + WNOCENT + SOATL + ESOCENT + WSOCENT + MT + '
    formula_2nd_stage_6 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)]) 

    if race:
        formula_2nd_stage_6 += ' + RACE'

    if state_of_birth_dummies:
        formula_2nd_stage_6 += ' + '
        formula_2nd_stage_6 += ' + '.join([f'DUMMY_STATE_{i}' for i in state_lst])

    tsls_6 = smf.ols(formula = formula_2nd_stage_6, data = df).fit()

    # regression (7) OLS
    formula_7 = 'LWKLYWGE ~ EDUC + AGEQ + AGESQ + MARRIED + SMSA + NEWENG + MIDATL + '
    formula_7 += 'ENOCENT + WNOCENT + SOATL + ESOCENT + WSOCENT + MT + '
    formula_7 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)])

    if race:
        formula_7 += ' + RACE'

    if state_of_birth_dummies:
        formula_7 += ' + '
        formula_7 += ' + '.join([f'DUMMY_STATE_{i}' for i in state_lst])

    ols_7 = smf.ols(formula = formula_7, data = df).fit()

    # regression (8) TSLS
    formula_1st_stage_8 = 'EDUC ~ AGEQ + AGESQ + MARRIED + SMSA + NEWENG + MIDATL + '
    formula_1st_stage_8 += 'ENOCENT + WNOCENT + SOATL + ESOCENT + WSOCENT + MT + ' 
    formula_1st_stage_8 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)]) 
    formula_1st_stage_8 += ' + '
    formula_1st_stage_8 += ' + '.join([f'DUMMY_YOB_{i} : DUMMY_QOB_{j}' for j in range(1,4) for i in range(0, 10)])

    if race:
        formula_1st_stage_8 += ' + RACE'

    if state_of_birth_dummies:
        formula_1st_stage_8 += ' + '
        formula_1st_stage_8 += ' + '.join([f'DUMMY_STATE_{i}' for i in state_lst])
        formula_1st_stage_8 += ' + '
        formula_1st_stage_8 += ' + '.join([f'DUMMY_STATE_{i} : DUMMY_QOB_{j}' for j in range(1,4) for i in state_lst])

    df['EDUC_pred_8'] = smf.ols(formula = formula_1st_stage_8, data = df).fit().predict()

    formula_2nd_stage_8 = 'LWKLYWGE ~ EDUC_pred_8 + AGEQ + AGESQ + MARRIED + SMSA + NEWENG + MIDATL + '
    formula_2nd_stage_8 += 'ENOCENT + WNOCENT + SOATL + ESOCENT + WSOCENT + MT + '
    formula_2nd_stage_8 += ' + '.join([f'DUMMY_YOB_{i}' for i in range(0, 9)]) 

    if race:
        formula_2nd_stage_8 += ' + RACE'

    if state_of_birth_dummies:
        formula_2nd_stage_8 += ' + '
        formula_2nd_stage_8 += ' + '.join([f'DUMMY_STATE_{i}' for i in set(df['STATE'])])

    tsls_8 = smf.ols(formula = formula_2nd_stage_8, data = df).fit()

    return OrderedDict([('ols_1', ols_1),
                        ('tsls_2', tsls_2),
                        ('ols_3', ols_3),
                        ('tsls_4', tsls_4),
                        ('ols_5', ols_5),
                        ('tsls_6', tsls_6),
                        ('ols_7', ols_7),
                        ('tsls_8', tsls_8)])
