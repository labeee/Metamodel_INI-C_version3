import chaospy
import numpy as np
import random, copy, io
import pandas as pd
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

dfdbtm = pd.read_csv('input/dbt_epws.csv')[['file', 'month', 'dbt_mean']]
dbtmn = dfdbtm['dbt_mean'].min()
dbtmx = dfdbtm['dbt_mean'].max()

config = {

    'dbt': ['c', dbtmn, dbtmx],                                                 # OK!

    'azimuth': ['d', 0, 45, 90, 135, 180, 225, 270, 315],                       # OK!
    'ceiling_height': ['c', 2.3, 10],                                           # OK!
    'building_xlen': ['c', 10, 40],                                             # OK!
    'wwr': ['c', 0.0, 0.99],                                                    # OK!
    'shading': ['h', [0, 0.30], 5],                                             # OK!
    'vert_shading': ['h', [0, 0.30], 45],                                       # OK!
    'pilotis': ['h', [0, 0.8], [1, 1]],                                         # OK!

    'floor_u': ['c', 1.75, 3.75],                                               # OK!
    'floor_ct': ['c', 130, 400],                                                # OK!

    'roof_u': ['c', 0.30, 5.2],                                                 # OK!
    'roof_ct': ['c', 0.15, 450],                                                # OK!
    'roof_absorptance': ['c', 0.1, 0.98],                                       # OK!

    'extwall_u': ['c', 0.25, 5],                                                # OK!
    'extwall_ct': ['c', 20, 450],                                               # OK!
    'extwall_absorptance': ['c', 0.1, 0.98],                                    # OK!

    'intwall_u': ['c', 0.31, 5],                                                # OK!
    'intwall_ct': ['c', 25, 400],                                               # OK!

    'shgc_jan': ['c', 0.2, 0.9],                                                # OK!
    'u_jan': ['c', 1.5, 6],                                                     # OK!
    'shgc_zen': ['c', 0.2, 0.9],                                                # OK!
    'u_zen': ['c', 1.5, 6],                                                     # OK!

    'int_zone_type': ['d', 'apt', 'app'],                                       # OK!
    'schedule': ['d', 'SCH_8H', 'SCH_10H', 'SCH_12H', 'SCH_16H', 'SCH_24H'],    # OK!
    'people': ['c', 1/30, 1],                                                   # OK!
    'lights': ['c', 3, 40],                                                     # OK!
    'equip': ['c', 3, 60],                                                      # OK!

    'prop_entorno': ['h', [0, 0.3], 2],                                         # OK!
    'paz': ['h', [0, 0.75], 5]                                                  # OK!

}

def gerar_amostra(nomeout='sample.csv', numsample=100, seed=1):

    sample = chaospy.create_sobol_samples(order=numsample, dim=len(config.keys()), seed=seed).T
    sample_str = np.array(sample, dtype='object')

    for icol, col in enumerate(config.keys()):
        cod, *values = config[col]

        if cod == 'c':
            sample_str[:, icol] = (values[0] + (values[1] - values[0])*sample[:, icol]).round(2)
            if col == 'dbt': xindexes = [(dbtm - dfdbtm['dbt_mean']).abs().sort_values().index[0] for dbtm in sample_str[:, icol]]

        elif cod == 'd':
            num_values = len(values)
            limits = np.linspace(0, 1, num_values+1)[1:-1]
            lower_value = 0
            for i, upper_value in enumerate(limits):
                sample_str[(lower_value <= sample[:,icol]) & (sample[:,icol] < upper_value), icol] = values[i]
                lower_value = upper_value
            sample_str[sample[:,icol] >= upper_value, icol] = values[-1]

        elif cod == 'h':
            lower_value = 0
            for i, each_value in enumerate(values):
                if type(each_value) == list:
                    value_h = each_value[0]
                    upper_value = each_value[1]
                    sample_str[(lower_value <= sample[:,icol]) & (sample[:,icol] < upper_value), icol] = value_h
                    lower_value = upper_value

                elif (type(each_value) == int) or (type(each_value) == float):
                    xsample = minmax_scale(sample[sample[:,icol] >= lower_value, icol], feature_range=(0, 1))
                    sample_str[sample[:,icol] >= lower_value, icol] = (value_h + (each_value - value_h)*xsample).round(2)

    sample = pd.DataFrame(sample_str)
    sample.columns = config.keys()
    sample = sample.reset_index().rename({'index': 'id'}, axis=1)

    sample['shell_depth'] = 4.5
    sample['building_ratio'] = 1
    sample['nfloor'] = 5
    sample['shading_ceiling_height'] = 0

    sample.loc[sample.wwr<=0.01, 'wwr'] = 0
    sample.loc[sample.shading<=0.1, 'shading'] = 0
    sample.loc[sample.vert_shading<=1, 'vert_shading'] = 0
    sample.loc[sample.prop_entorno<=0.01, 'prop_entorno'] = 0
    sample.loc[sample.paz<=1, 'paz'] = 0

    sample.loc[sample.wwr==0, ['shading', 'vert_shading']] = 0

    sample[['month', 'dbt_mean', 'file']] = dfdbtm.loc[xindexes, ['month', 'dbt_mean', 'file']].values
    sample = sample.drop('dbt', axis=1)

    sample.to_csv(nomeout, index=None)

gerar_amostra('sample_inic.csv', 300_000, 1)
