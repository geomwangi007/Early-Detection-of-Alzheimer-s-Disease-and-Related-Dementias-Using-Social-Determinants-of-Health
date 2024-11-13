import pandas as pd

def apply_mappings(df):
    mappings = {
    'age_03': {
        '1. 50–59': 1,
        '3. 70–79': 3,
        '2. 60–69': 2,
        '0. 49 or younger': 0,
        '4. 80+': 4
    },
    'urban_03': {
        '1. 100,000+': 1,
        '0. <100,000': 0
    },
    'married_03': {
        '3. Widowed': 3,
        '1. Married or in civil union': 1,
        '4. Single': 4,
        '2. Separated or divorced': 2
    },
    'edu_gru_03': {
        '3. 7–9 years': 3,
        '1. 1–5 years': 1,
        '0. No education': 0,
        '2. 6 years': 2,
        '4. 10+ years': 4
    },
    'n_living_child_03': {
        '1. 1 or 2': 1,
        '3. 5 or 6': 3,
        '0. No children': 0,
        '2. 3 or 4': 2,
        '4. 7+': 4
    },
    'glob_hlth_03': {
        '4. Fair': 4,
        '5. Poor': 5,
        '3. Good': 3,
        '1. Excellent': 1,
        '2. Very good': 2
    },
    'employment_03': {
        '3. Dedicated to household chores': 3,
        '1. Currently Working': 1,
        '2. Currently looking for work': 2,
        '4. Retired, incapacitated, or does not work': 4
    },
    'age_12': {
        '2. 60–69': 2,
        '1. 50–59': 1,
        '4. 80+': 4,
        '3. 70–79': 3,
        '0. 49 or younger': 0
    },
    'urban_12': {
        '0. <100,000': 0,
        '1. 100,000+': 1
    },
    'married_12': {
        '1. Married or in civil union': 1,
        '3. Widowed': 3,
        '2. Separated or divorced': 2,
        '4. Single': 4
    },
    'edu_gru_12': {
        '0. No education': 0,
        '3. 7–9 years': 3,
        '1. 1–5 years': 1,
        '2. 6 years': 2,
        '4. 10+ years': 4
    },
    'n_living_child_12': {
        '1. 1 or 2': 1,
        '3. 5 or 6': 3,
        '0. No children': 0,
        '2. 3 or 4': 2,
        '4. 7+': 4
    },
    'glob_hlth_12': {
        '4. Fair': 4,
        '3. Good': 3,
        '2. Very good': 2,
        '5. Poor': 5,
        '1. Excellent': 1
    },
    'bmi_12': {
        '3. Overweight': 3,
        '4. Obese': 4,
        '2. Normal weight': 2,
        '1. Underweight': 1,
        '5. Morbidly obese': 5
    },
    'decis_famil_12': {
        '2. Approximately equal weight': 2,
        '3. Spouse': 3,
        '1. Respondent': 1
    },
    'decis_personal_12': {
        '1. A lot': 1,
        '2. A little': 2,
        '3. None': 3
    },
    'employment_12': {
        '1. Currently Working': 1,
        '2. Currently looking for work': 2,
        '3. Dedicated to household chores': 3,
        '4. Retired, incapacitated, or does not work': 4
    },
    'satis_ideal_12': {
        '3. Disagrees': 3,
        '1. Agrees': 1,
        '2. Neither agrees nor disagrees': 2
    },
    'satis_excel_12': {
        '3. Disagrees': 3,
        '2. Neither agrees nor disagrees': 2,
        '1. Agrees': 1
    },
    'satis_fine_12': {
        '1. Agrees': 1,
        '3. Disagrees': 3,
        '2. Neither agrees nor disagrees': 2
    },
    'cosas_imp_12': {
        '1. Agrees': 1,
        '2. Neither agrees nor disagrees': 2,
        '3. Disagrees': 3
    },
    'wouldnt_change_12': {
        '3. Disagrees': 3,
        '1. Agrees': 1,
        '2. Neither agrees nor disagrees': 2
    },
    'memory_12': {
        '2. Very good': 2,
        '4. Fair': 4,
        '3. Good': 3,
        '5. Poor': 5,
        '1. Excellent': 1
    },
    'ragender': {
        '1.Man': 1,
        '2.Woman': 2
    },
    'rameduc_m': {
        '1.None': 1,
        '2.Some primary': 2,
        '3.Primary': 3,
        '4.More than primary': 4
    },
    'rafeduc_m': {
        '1.None': 1,
        '2.Some primary': 2,
        '4.More than primary': 4,
        '3.Primary': 3
    },
    'sgender_03': {
        '2.Woman': 2,
        '1.Man': 1
    },
    'rrelgimp_03': {
        '1.very important': 1,
        '2.somewhat important': 2,
        '3.not important': 3
    },
    'sgender_12': {
        '2.Woman': 2,
        '1.Man': 1
    },
    'rjlocc_m_12': {
        '6.Workers in Agriculture, Livestock, Forestry, and Fishing': 6,
        '18.Safety and Security Personnel': 18,
        '8.Artisans and Workers in Production, Repair, Maintenance': 8,
        '14.Merchants and Sales Representatives': 14,
        '17.Domestic Workers': 17,
        '9.Operators of Fixed Machinery and Equipment for Ind. Production': 9,
        '15.Traveling Salespeople and Traveling Salespeople of Services': 15,
        '16.Workers in the Service Industry': 16,
        '4.Workers in Art, Shows, and Sports': 4,
        '3.Educators': 3,
        '12.Department Heads/Coordinators/Supervisors in Admin and Service Activities': 12,
        '10.Asst/Laborers etc in Ind. Production, Repair, Maintenance': 10,
        '1.Professionals': 1,
        '13.Administrative Support Staff': 13,
        '11.Drivers and Asst Drivers of Mobile Machinery and Transport Vehicles': 11,
        '7.Bosses/Supervisors etc in Artistic, Ind. Production, Repair, Maintenance Activities': 7,
        '2.Technicians': 2,
        '5.Officials and Directors Public, Private, and Social Sectors': 5
    },
    'rrelgimp_12': {
        '2.somewhat important': 2,
        '1.very important': 1,
        '3.not important': 3
    },
    'rrfcntx_m_12': {
        '9.Never': 9,
        '6.2 or 3 times a month': 6,
        '4.Once a week': 4,
        '3.2 or 3 times a week': 3,
        '8.Almost Never, sporadic': 8,
        '1.Almost every day': 1,
        '2.4 or more times a week': 2,
        '7.Once a month': 7,
        '5.4 or more times a month': 5
    },
    'rsocact_m_12': {
        '9.Never': 9,
        '1.Almost every day': 1,
        '2.4 or more times a week': 2,
        '3.2 or 3 times a week': 3,
        '4.Once a week': 4,
        '8.Almost Never, sporadic': 8,
        '5.2 or 3 times a month': 5,
        '6.Once a month': 6,
        '7.4 or more times a month': 7
    },
    'rrelgwk_12': {
        '0.No': 0,
        '1.Yes': 1
    },
    'a34_12': {
        'No 2': 0,
        'Yes 1': 1
    },
    'j11_12': {
        'Wood, mosaic, or other covering 1': 1,
        'Concrete 2': 2,
        'Mud 3': 3
    }
    }

    
    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping)
    return df

def reshape_data(df):
    stubnames = [col.rsplit('_', 1)[0] for col in df.columns if col.endswith('_03') or col.endswith('_12')]
    stubnames = list(set(stubnames))
    data_long = pd.wide_to_long(
        df,
        stubnames=stubnames,
        i=['uid', 'year', 'composite_score'],
        j='time',
        sep='_',
        suffix='\\d+'
    ).reset_index()
    return data_long

def split_features_labels(data_long):
    X = data_long.drop(['composite_score', 'year'], axis=1)
    y = data_long['composite_score']
    return X, y
