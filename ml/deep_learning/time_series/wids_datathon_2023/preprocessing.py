import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Save target 
target = 'contest-tmp2m-14d__tmp2m'

def location_nom(train, test):
    # Truncate values
    train.lat = train.lat.round(6)
    train.lon = train.lon.round(6)
    test.lat = test.lat.round(6)
    test.lon = test.lon.round(6)

    # Concatenate datasets
    all_df = pd.concat([train, test], axis=0)
    # Create column with groups
    all_df['loc_group'] = all_df.groupby(['lat','lon']).ngroup()
    # Put datasets back together
    train = all_df.iloc[:len(train)]
    test = all_df.iloc[len(train):].drop(target, axis=1)
    
    return train, test

def with_nans(dataset):
    # Find columns with NaN values and count such values
    with_nan = dataset.isna().sum()
    # Return appropriate message if no NaNs are found
    if with_nan[with_nan > 0].empty:
        print('No NaN values found.')
    else:
        return with_nan[with_nan > 0]


def parse_start_date(df):
    df['startdate'] =pd.to_datetime(df.startdate) 
    df['year'] = df.startdate.dt.year
    df['month'] = df.startdate.dt.month
    df['day_of_year'] = df.startdate.dt.dayofyear

    return df

def categorical_encode(train, test):
    le = LabelEncoder()
    train['climateregions__climateregion'] = le.fit_transform(train['climateregions__climateregion'])
    test['climateregions__climateregion'] = le.transform(test['climateregions__climateregion'])
    return train, test


group_na = [
    'nmme0-tmp2m-34w__ccsm30', 
    'nmme-tmp2m-56w__ccsm3', 
    'nmme-prate-34w__ccsm3', 
    'nmme0-prate-56w__ccsm30', 
    'nmme0-prate-34w__ccsm30', 
    'nmme-prate-56w__ccsm3', 
    'nmme-tmp2m-34w__ccsm3']

group_means =  ['nmme0-tmp2m-34w__nmme0mean', 
 'nmme-tmp2m-56w__nmmemean', 
 'nmme-prate-34w__nmmemean', 
 'nmme0-prate-56w__nmme0mean', 
 'nmme0-prate-34w__nmme0mean', 
 'nmme-prate-56w__nmmemean', 
 'nmme-tmp2m-34w__nmmemean']


group_1 = ['nmme0-tmp2m-34w__cancm30',
'nmme0-tmp2m-34w__cancm40',
'nmme0-tmp2m-34w__ccsm40',
'nmme0-tmp2m-34w__cfsv20',
'nmme0-tmp2m-34w__gfdlflora0',
'nmme0-tmp2m-34w__gfdlflorb0',
'nmme0-tmp2m-34w__gfdl0',
'nmme0-tmp2m-34w__nasa0']

group_2 = ['nmme-tmp2m-56w__cancm3',
'nmme-tmp2m-56w__cancm4',
'nmme-tmp2m-56w__ccsm4',
'nmme-tmp2m-56w__cfsv2',
'nmme-tmp2m-56w__gfdl',
'nmme-tmp2m-56w__gfdlflora',
'nmme-tmp2m-56w__gfdlflorb',
'nmme-tmp2m-56w__nasa']

group_3 = ['nmme-prate-34w__cancm3',
'nmme-prate-34w__cancm4',
'nmme-prate-34w__ccsm4',
'nmme-prate-34w__cfsv2',
'nmme-prate-34w__gfdl',
'nmme-prate-34w__gfdlflora',
'nmme-prate-34w__gfdlflorb',
'nmme-prate-34w__nasa']

group_4 = [ 'nmme0-prate-56w__cancm30',
'nmme0-prate-56w__cancm40',
'nmme0-prate-56w__ccsm40',
'nmme0-prate-56w__cfsv20',
'nmme0-prate-56w__gfdlflora0',
'nmme0-prate-56w__gfdlflorb0',
'nmme0-prate-56w__gfdl0',
'nmme0-prate-56w__nasa0']

group_5 = ['nmme0-prate-34w__cancm30',
'nmme0-prate-34w__cancm40',
'nmme0-prate-34w__ccsm40',
'nmme0-prate-34w__cfsv20',
'nmme0-prate-34w__gfdlflora0',
'nmme0-prate-34w__gfdlflorb0',
'nmme0-prate-34w__gfdl0',
'nmme0-prate-34w__nasa0']

group_6 = ['nmme-prate-56w__cancm3',
'nmme-prate-56w__cancm4',
'nmme-prate-56w__ccsm4',
'nmme-prate-56w__cfsv2',
'nmme-prate-56w__gfdl',
'nmme-prate-56w__gfdlflora',
'nmme-prate-56w__gfdlflorb',
'nmme-prate-56w__nasa']

group_7 = ['nmme-tmp2m-34w__cancm3',
'nmme-tmp2m-34w__cancm4',
'nmme-tmp2m-34w__ccsm4',
'nmme-tmp2m-34w__cfsv2',
'nmme-tmp2m-34w__gfdl',
'nmme-tmp2m-34w__gfdlflora',
'nmme-tmp2m-34w__gfdlflorb',
'nmme-tmp2m-34w__nasa']

groups = [group_1, group_2, group_3, group_4, group_5, group_6, group_7]