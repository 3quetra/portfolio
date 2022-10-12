import numpy as np

def map_to_numbers(survey):
    num_survey = survey.copy()

    cols_to_map_q7 = [
        'q0007_0001', 'q0007_0002', 'q0007_0003', 'q0007_0004', 'q0007_0005', 'q0007_0006', 
        'q0007_0007', 'q0007_0008', 'q0007_0009', 'q0007_0010', 'q0007_0011'
        ]
    cols_to_map_q8 = [
        'q0008_0001', 'q0008_0002', 'q0008_0003', 'q0008_0004', 'q0008_0005', 'q0008_0006', 
        'q0008_0007', 'q0008_0008', 'q0008_0009', 'q0008_0010', 'q0008_0011', 'q0008_0012'
        ]

    map_q7 = {'Often': 4, 'Sometimes': 3, 'Rarely': 2, 'Never, but open to it': 1, 'Never, and not open to it': 0, 'No answer': np.nan}
    map_q1 = {'Very masculine': 3, 'Somewhat masculine': 2, 'Not very masculine': 1, 'Not at all masculine': 0, 'No answer': np.nan}
    map_q2 = {'Very important': 3, 'Somewhat important': 2, 'Not too important': 1, 'Not at all important': 0, 'No answer': np.nan}


    map_income = {
        '$0-$9,999': 1,
        '$10,000-$24,999': 2,
        '$25,000-$49,999': 3, 
        '$50,000-$74,999': 4,
        '$75,000-$99,999': 5,
        '$100,000-$124,999': 6,
        '$125,000-$149,999': 7,
        '$150,000-$174,999': 8,
        '$175,000-$199,999': 9,   
        '$200,000+': 10,
        'Prefer not to answer': np.nan,
        np.nan:  np.nan
    }

    map_q8 = {
        'None of the above': 1,
        'Your height': 1,
        'Appearance of your genitalia': 1,
        'Your weight': 1,
        'Your hair or hairline': 1,
        'Your physique': 1,
        'Your clothing or style': 1,
        'Sexual performance or amount of sex': 1,
        'Your mental health': 1, 
        'Your physical health': 1,
        'Your finances, including your current or future income, assets, or debt': 1,
        'Your ability to provide for your family, current or anticipated': 1,
        'Not selected': 0,
        np.nan:  np.nan
    }

    map_race = {
        'Non-white': 1,
        'White': 2,
    }

    map_age = {
        '18 - 34': 1,
        '35 - 64': 2,
        '65 and up': 3,
        np.nan:  np.nan
    }

    map_educ4 = {
        'High school or less': 1,
        'Some college': 2,  
        'College or more': 3,
        'Post graduate degree': 4,
        np.nan:  np.nan
    }

    map_orient = {
        'Straight': 1,
        'Other': 2,
        'Gay/Bisexual': 3,
        
        'No answer': np.nan,
        np.nan:  np.nan 
    }

    map_kids = {
        'No children': 0,
        'Has children': 1,
        np.nan:  np.nan 
    }

    num_survey[cols_to_map_q7] = survey[cols_to_map_q7].apply(lambda column: column.map(lambda value: map_q7[value]))
    num_survey[cols_to_map_q8] = survey[cols_to_map_q8].apply(lambda column: column.map(lambda value: map_q8[value]))
    num_survey['q0001'] = survey['q0001'].apply(lambda value: map_q1[value])
    num_survey['q0002'] = survey['q0002'].apply(lambda value: map_q2[value])
    num_survey['Income'] = survey['Income'].apply(lambda value: map_income[value])
    num_survey['race2'] = survey['race2'].apply(lambda value: map_race[value])
    num_survey['educ4'] = survey['educ4'].apply(lambda value: map_educ4[value])
    num_survey['orientation'] = survey['orientation'].apply(lambda value: map_orient[value])
    num_survey['kids'] = survey['kids'].apply(lambda value: map_kids[value])
    num_survey['age3'] = survey['age3'].apply(lambda value: map_age[value])

    return num_survey, map_q7, map_income, map_race, map_educ4, map_orient, map_kids, map_age, map_q8
