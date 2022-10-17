import numpy as np

def map_to_numbers(profiles):
    mix_profiles = profiles.copy()

    map_body_t = {
        'used up': 1,
        'skinny': 2,
        'thin': 3,
        'average': 4,
        'fit': 5,
        'athletic': 6,
        'jacked': 7,
        'curvy': 8,
        'a little extra': 9,
        'full figured': 10,
        'overweight': 11,
        'rather not say': 0,
        np.nan: np.nan,       
    }

    map_diet = {
        'anything': 1,
        'mostly anything': 1,
        'strictly anything': 1,
        'vegetarian': 2,
        'mostly vegetarian': 2,
        'strictly vegetarian': 2,
        'vegan': 3,
        'mostly vegan': 3,
        'strictly vegan': 3,
        'halal': 4,
        'mostly halal': 4,
        'strictly halal': 4,
        'kosher': 5,
        'mostly kosher': 5,
        'strictly kosher': 5,
        'other': 6,
        'mostly other': 6,
        'strictly other': 6, 
        np.nan: np.nan
    }
        
    map_drinks = {
        'not at all': 0,
        'rarely': 1,
        'socially': 2,
        'often': 3, 
        'very often': 4,
        'desperately': 5,
        np.nan: np.nan   
    }
        
    map_drugs = {
        'never': 0, 
        'sometimes': 1,
        'often': 2, 
        np.nan: np.nan
        
    }

    map_edu = {
        np.nan: np.nan,
        'dropped out of high school': 1,
        'high school': 1,
        'working on high school': 1,
        'graduated from high school': 1,
        'dropped out of space camp': 2,
        'space camp': 2, 
        'working on space camp': 2,
        'graduated from space camp': 2,
        'dropped out of two-year college': 3,
        'working on two-year college': 3,
        'two-year college': 3,
        'graduated from two-year college': 3,
        'dropped out of college/university': 4,
        'college/university': 4,  
        'working on college/university': 4, 
        'graduated from college/university': 4,
        'dropped out of law school': 4,
        'law school': 4,
        'working on law school': 4,
        'graduated from law school': 4,
        'dropped out of med school': 4,
        'med school': 4,
        'working on med school': 4,
        'graduated from med school': 4,
        'dropped out of masters program': 5,
        'masters program': 5,
        'working on masters program': 5,
        'graduated from masters program': 5,
        'dropped out of ph.d program': 6,
        'ph.d program': 6,
        'working on ph.d program': 6,
        'graduated from ph.d program': 6,
    }
           
    
        
    map_kids = {
        np.nan: np.nan,
        'doesn&rsquo;t want kids': 0,
        'doesn&rsquo;t have kids, and doesn&rsquo;t want any': 0, 
        'doesn&rsquo;t have kids': 0,
        'doesn&rsquo;t have kids, but might want them': 1, 
        'doesn&rsquo;t have kids, but wants them': 1,
        'might want kids': 1,
        'wants kids': 1, 
        'has a kid': 2, 
        'has a kid, but doesn&rsquo;t want more': 2, 
        'has kids': 3,
        'has kids, but doesn&rsquo;t want more': 3,
        'has a kid, and might want more': 4,
        'has a kid, and wants more': 4,
        'has kids, and might want more': 4,
        'has kids, and wants more': 4
    }
        
    map_smoking = {
        np.nan: np.nan,
        'no': 0,
        'sometimes': 1, 
        'when drinking': 1, 
        'trying to quit': 3,
        'yes': 4, 
    }
    
       
    map_relig = {
    0:'agnosticism',
    1: 'atheism',
    2: 'christianity',
    3: 'other',
    4: 'catholicism',
    5: 'buddhism',
    6: 'judaism',
    7: 'hinduism',
    8: 'islam'
}
        
        
    mix_profiles['body_type'] = profiles['body_type'].apply(lambda value: map_body_t[value])
    mix_profiles['diet'] = profiles['diet'].apply(lambda value: map_diet[value])
    mix_profiles['drinks'] = profiles['drinks'].apply(lambda value: map_drinks[value])
    mix_profiles['smokes'] = profiles['smokes'].apply(lambda value: map_smoking[value])
    mix_profiles['offspring'] = profiles['offspring'].apply(lambda value: map_kids[value])
    mix_profiles['education'] = profiles['education'].apply(lambda value: map_edu[value])
    mix_profiles['drugs'] = profiles['drugs'].apply(lambda value: map_drugs[value])


    return mix_profiles, map_body_t, map_diet, map_drinks, map_smoking, map_kids, map_edu, map_drugs, map_relig


def age_to_num(value): 
    if value < 30:
        return 1
    elif value < 40:
        return 2
    elif value < 60:
        return 3
    elif value > 60:
        return 4
            
def age_to_str(value): 
    if value == 1:
        return '18-29'
    elif value == 2:
        return '30-39'
    elif value == 3:
        return '40-59'
    elif value == 4:
        return '60 and up'





            
       
       
        
        
        
    
