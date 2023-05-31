import pandas as pd

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
