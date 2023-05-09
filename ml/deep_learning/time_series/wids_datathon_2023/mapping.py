import numpy as np
import pandas as pd


def map_to_def(regions):
    mix_regions = regions.copy()

    map_to_region = {
        'BSh': 'Hot semi-arid climate',
        'BSk': 'Cold semi-arid climate',
        'BWh': 'Hot desert climate',
        'BWk': 'Cold desert climate',
        'Cfa': 'Humid subtropical climate',
        'Cfb': 'Temperate oceanic climate or subtropical highland climate',
        'Csa': 'Hot-summer Mediterranean climate',
        'Csb': 'Warm-summer Mediterranean climate',
        'Dfa': 'Hot-summer humid continental climate',
        'Dfb': 'Warm-summer humid continental climate',
        'Dfc': 'Subarctic climate',
        'Dsb': 'Mediterranean-influenced warm-summer humid continental climate',
        'Dsc': 'Mediterranean-influenced subarctic climate',
        'Dwa': 'Monsoon-influenced hot-summer humid continental climate',
        'Dwb': 'Monsoon-influenced warm-summer humid continental climate',
        np.nan: np.nan       
    }

    mix_regions['Definition'] = regions.index.map(lambda value: map_to_region[value])

    return mix_regions