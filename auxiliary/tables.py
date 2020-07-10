import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from operator import itemgetter

from . import regressions

def create_table_qob(results, outcome_variables = None):

    table = """
    <table>
        <thead>
            <tr>
                <th>Outcome variable</th>
                <th>Birth cohort</th>
                <th>Mean</th>
                <th colspan="3">Quarter-of-birth effect</th>
                <th>F-test</th>
            </tr>
            <tr>
                <th></th><th></th><th></th>
                <th>I</th>
                <th>II</th>
                <th>III</th>
                <th>[P-value]</th>
            </tr>
        </thead>
        <hline>
        <tbody>
    """

    if outcome_variables:
        for out_var, rslt in zip(outcome_variables, results):
            table += create_table_row_qob(out_var, rslt['cohort'], rslt['mean'], rslt['ols'])

    else:
        for rslt in results:
            table += create_table_row_qob(rslt['var'], rslt['cohort'], rslt['mean'], rslt['ols'])
    
    table += """
        </tbody>
    </table>
    """

    return table
    
def create_table_row_qob(outcome_variable, cohort, mean, ols):

    table_row = f"""
    <tr>
        <td>{outcome_variable}</td>
        <td>{cohort}</td>
        <td>{mean:5.2f}</td>
        <td>{ols.params['DUMMY_QOB_1']:6.3f}</td>
        <td>{ols.params['DUMMY_QOB_2']:6.3f}</td>
        <td>{ols.params['DUMMY_QOB_3']:6.3f}</td>
        <td>{ols.fvalue:6.1f}</td>
    </tr>
    <tr>
        <td></td><td></td><td></td>
        <td>({ols.bse['DUMMY_QOB_1']:5.3f})</td>
        <td>({ols.bse['DUMMY_QOB_2']:5.3f})</td>
        <td>({ols.bse['DUMMY_QOB_3']:5.3f})</td>
        <td>[{ols.f_pvalue:6.4f}]</td>
    </tr>
    """
    return table_row

def create_table_ols_tls(results, state_of_birth_dummies = 0, race = True):

    educ_keys = ['EDUC', 'EDUC_pred_2', 'EDUC', 'EDUC_pred_4', \
                 'EDUC', 'EDUC_pred_6', 'EDUC', 'EDUC_pred_8']

    table = """
    <table>
        <thead>
            <tr>
                <th>
            </tr>
            <tr>
                <td>Independent variable</td>
                
            </tr>
        </thead>
        <hline>
        <tbody>
        </tbody>
    </table>
    
    """


    table = """
    <table>
        <thead>
            <tr>
                <th></th>
    """
    table += '\n'.join([f'<th>({i})</th>' for i in range(1, 9)])
    table += """
            </tr>
            <tr>
                <th>Independent variable</th>
    """
    table += '\n'.join([f'<th>{method}</th>' for method in ['OLS', 'TSLS'] * 4])
    table += """
            </tr>
        </thead>
        <hline>
        <tbody>
            <tr>
                <td>Years of education</td>
    """
    table += '\n'.join([f'<td>{rslt.params.get(key):6.4f}</td>' for key, rslt in zip(educ_keys, results.values())])
    table += """
            </tr>
            <tr>
                <td></td>
    """
    table += '\n'.join([f'<td>({rslt.bse.get(key):5.4f})</td>' for key, rslt in zip(educ_keys, results.values())])
    
    if race:
        table +="""
                </tr>
                <tr>
                    <td>Race (1 = black)</td>
        """
        table += '\n'.join(['<td>-</td>'] * 4)
        table += '\n'.join([f'<td>{rslt.params.get("RACE"):6.4f}</td>' for rslt in list(results.values())[4:]])
        table += """
                </tr>
                <tr>
        """
        table += '\n'.join(['<td></td>'] * 5)
        table += '\n'.join([f'<td>({rslt.bse.get("RACE"):5.4f})</td>' for rslt in list(results.values())[4:]])

    table +="""
            </tr>
            <tr>
                <td>SMSA (1 = center city)</td>
    """
    table += '\n'.join(['<td>-</td>'] * 4)
    table += '\n'.join([f'<td>{rslt.params.get("SMSA"):6.4f}</td>' for rslt in list(results.values())[4:]])
    table += """
            </tr>
            <tr>
    """
    table += '\n'.join(['<td></td>'] * 5)
    table += '\n'.join([f'<td>({rslt.bse.get("SMSA"):5.4f})</td>' for rslt in list(results.values())[4:]])
    table +="""
            </tr>
            <tr>
                <td>Married (1 = married)</td>
    """
    table += '\n'.join(['<td>-</td>'] * 4)
    table += '\n'.join([f'<td>{rslt.params.get("MARRIED"):6.4f}</td>' for rslt in list(results.values())[4:]])
    table += """
            </tr>
            <tr>
    """
    table += '\n'.join(['<td></td>'] * 5)
    table += '\n'.join([f'<td>({rslt.bse.get("MARRIED"):5.4f})</td>' for rslt in list(results.values())[4:]])
    table += """
            </tr>
            <tr>
                <td>9 Year-of-birth dummies</td>
    """
    table += '\n'.join(['<td>Yes</td>'] * 8)
    table += """
            </tr>
            <tr>
                <td>8 Region-of-residence dummies</td>
    """
    table += '\n'.join(['<td>No</td>'] * 4 + ['<td>Yes</td>'] * 4)
    
    if state_of_birth_dummies:
        table += f"""
        </tr>
        <tr>
            <td>{state_of_birth_dummies} State-of-birth dummies</td>
        """
        table += '\n'.join(['<td>Yes</td>'] * 8)

    table += """
            </tr>
            <tr>
                <td>Age</td>
                <td>-</td>
                <td>-</td>
    """
    table += '\n'.join([f'<td>{rslt.params.get("AGEQ"):6.4f}</td>' for rslt in itemgetter(2, 3)(list(results.values()))])
    table += '<td>-</td><td>-</td>'
    table += '\n'.join([f'<td>{rslt.params.get("AGEQ"):6.4f}</td>' for rslt in itemgetter(6, 7)(list(results.values()))])
    table +="""
            </tr>
            <tr>
                <td></td>
                <td></td>
                <td></td>
    """
    table += '\n'.join([f'<td>({rslt.bse.get("AGEQ"):5.4f})</td>' for rslt in itemgetter(2, 3)(list(results.values()))])
    table += '<td>-</td><td>-</td>'
    table += '\n'.join([f'<td>({rslt.bse.get("AGEQ"):5.4f})</td>' for rslt in itemgetter(6, 7)(list(results.values()))])
    table += """
            </tr>
            <tr>
                <td>Age Squared</td>
                <td>-</td>
                <td>-</td>
    """
    table += '\n'.join([f'<td>{rslt.params.get("AGESQ"):6.4f}</td>' for rslt in itemgetter(2, 3)(list(results.values()))])
    table += '<td>-</td><td>-</td>'
    table += '\n'.join([f'<td>{rslt.params.get("AGESQ"):6.4f}</td>' for rslt in itemgetter(6, 7)(list(results.values()))])
    table += """
        </tr>
            <td></td>
            <td></td>
            <td></td>
    """
    table += '\n'.join([f'<td>({rslt.bse.get("AGESQ"):5.4f})</td>' for rslt in itemgetter(2, 3)(list(results.values()))])
    table += '<td>-</td><td>-</td>'
    table += '\n'.join([f'<td>({rslt.bse.get("AGESQ"):5.4f})</td>' for rslt in itemgetter(6, 7)(list(results.values()))])
    table += """
            </tr>
        </tbody>
    </table>
    """

    return table