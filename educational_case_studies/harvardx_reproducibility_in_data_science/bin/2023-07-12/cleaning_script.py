# Import libraries related data wrangling
import pandas as pd
import re
import datetime
datetime.timedelta

def resolve_age(df: pd.DataFrame) -> pd.DataFrame:
    copy = df.copy()
    # Convert type object to int and all that is not int to NA
    copy['Age of patient'] = pd.to_numeric(copy['Age of patient'], 'coerce')
    copy['Age of patient'] = copy['Age of patient'].astype('Int64')
    return copy


def resolve_sex(df: pd.DataFrame) -> pd.DataFrame:
    # Replace every value that has "f" in it with 0 and with 1 if it has "m"
    def replace_sex(value):
        value = value.lower()
        if 'f' in value:
            return 0
        if 'm' in value:
            return 1
        return None
    
    copy = df.copy()
    copy['Patient gender'] = copy['Patient gender'].apply(replace_sex)
    copy['Patient gender'] = copy['Patient gender'].astype('Int64')

    return copy


def resolve_complicat(df: pd.DataFrame) -> pd.DataFrame:
    # Replace every value that has "n" in it with 0 and with 1 if it has "y"
    def replace_complicat(value):
        value = value.lower()
        if 'n' in value:
            return 0
        if 'y' in value:
            return 1
        return None
    
    copy = df.copy()
    # Remove "?" from the column name
    copy.rename(columns={
        'Complications?':  'Complications'},
        inplace=True)
    copy['Complications'] = copy['Complications'].apply(replace_complicat)
    copy['Complications'] = copy['Complications'].astype('Int64')

    return copy


def resolve_tstage(df: pd.DataFrame) -> pd.DataFrame:
    # Map Roman numbers with Arabic
    map_stage = {
    'I': 1,
    'II' : 2,
    'III' : 3,
    'IV' : 4,
    }

    copy = df.copy()
    # Replace roman numbers with arabic and convert existing arabic to int64
    copy['Tumor stage'] = copy['Tumor stage'].apply(lambda value: map_stage[value] if value in map_stage else pd.to_numeric(value, 'coerce'))


    return copy


def resolve_date(df: pd.DataFrame) -> pd.DataFrame:
    # Parse strings to datetime format
    df['Date enrolled'] = pd.to_datetime(df['Date enrolled'], errors='coerce', format='%m/%d/%Y').fillna(
    pd.to_datetime(df['Date enrolled'], errors='coerce', format='%m/%d/%y')).fillna(
    pd.to_datetime(df['Date enrolled'], errors='coerce', format='%m-%Y')).fillna(
    pd.to_datetime(df['Date enrolled'], errors='coerce', format='%m/%Y')).fillna(
    pd.to_datetime(df['Date enrolled'], errors='coerce', format='%b %Y'))

    # Count the difference from the start date of the reseach to the enrollment date
    def month_num(value):
        if value is not None:
            return (value - datetime.datetime(1999, 1, 12)).days
        
    copy = df.copy()
    copy['Date enrolled'] = copy['Date enrolled'].apply(month_num)
    # Rename date column
    copy.rename(columns={
            'Date enrolled':  'Days enrolled'},
            inplace=True)
    
    return copy

def resolve_height(df: pd.DataFrame) -> pd.DataFrame:
    def convert_height(value):
        # Check if the value is in cm, if yes - return it as it is
        if isinstance(value, float):
            return value
        # Check if the value is in inches (") or feet (') and create groups to deal with each variation
        if "'" in value or '"' in value:
            match = re.search(r'((\d+)\')?((\d+)")?', value)
            # Convert feet to inches
            inches = (int(match.group(2)) if match.group(2) is not None else 0) * 12 + (int(match.group(4)) if match.group(4) is not None else 0)
            # Convert inches to cm
            return inches * 2.54
        # Remove "cm"
        if "cm" in value:
            return float(value.replace('cm', ''))
        return value

    copy = df.copy()
    copy['Height'] = copy['Height'].apply(convert_height)
    copy['Height'] = copy['Height'].astype('float64')

    return copy



def clean_up(df: pd.DataFrame) -> pd.DataFrame:
    # Find a row with "Treatment B" 
    second_table_index = df[df['Treatment A'] == 'Treatment B'].index[0]

    # Make 2 slices of the dataset based on location of the row with "Treatment B" 
    treat_a = df.iloc[:second_table_index].copy()
    treat_b = df.iloc[second_table_index + 1:].copy()

    # Code treatment types with 0 and 1 
    treat_a.insert(1, 'Treatment type', 0)
    treat_b.insert(1, 'Treatment type', 1)

    # Remove "Treatment A" column
    df = pd.concat([treat_a, treat_b])[[
        'Treatment type', 
        'Age of patient', 
        'Patient gender',
        'Height', 
        'Tumor stage', 
        'Date enrolled', 
        'Complications?']]
    
    # Add "id" column
    df.insert(0, 'id', range(1, len(df)+1))

    # Run cleaning functions and save the result to DataFrames
    df = resolve_age(df)
    df = resolve_sex(df)
    df = resolve_complicat(df)
    df = resolve_tstage(df)
    df = resolve_height(df)
    df = resolve_date(df)
    return df
    
# Clean the dataset
clean_df = clean_up(pd.read_csv('./data/2023-07-12/dataset_1.csv'))

# Display cleaned dataset
print(clean_df)
# Display cleaned dataset's data types
print(clean_df.dtypes)
# Save cleaned dataset to csv
clean_df.to_csv('./data/2023-07-13/cleaned_dataset_1.csv')