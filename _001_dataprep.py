'''
Preparing data and creating test data for:
- RA created summaries
'''

import sys
import os
os.chdir('/Users/fogellmcmuffin/Dropbox/ai_irpd_coding/data')   # Working Dir.

# Modules & Packages
sys.path.append('/Users/fogellmcmuffin/git_repos/irpd_ai/')
import pandas as pd
import functions as f
import importlib
importlib.reload(f)


###################
## Summary data ##
###################

## Merging raw summary data ##

def MergeRawData(summary_type: str, RA: str):
    '''
    Function that merges the treatments for summary data
    '''
    if RA != 'both':
        ra_noise_raw = pd.read_csv(f'raw/{summary_type}_noise_{RA}.csv')
        ra_no_noise_raw = pd.read_csv(f'raw/{summary_type}_no_noise_{RA}.csv')
    else:
        ra1_noise_raw = pd.read_csv(f'raw/{summary_type}_noise_eli.csv')
        ra1_no_noise_raw = pd.read_csv(f'raw/{summary_type}_no_noise_eli.csv')
        ra2_noise_raw = pd.read_csv(f'raw/{summary_type}_noise_thi.csv')
        ra2_no_noise_raw = pd.read_csv(f'raw/{summary_type}_no_noise_thi.csv')
        ra_noise_raw = pd.merge(ra1_noise_raw, ra2_noise_raw, 'outer')
        ra_no_noise_raw = pd.merge(ra1_no_noise_raw, ra2_no_noise_raw, 'outer')

    # Merging
    ra_no_noise_raw['treatment'] = 0
    ra_noise_raw['treatment'] = 1
    merged_raw = pd.concat([ra_no_noise_raw, ra_noise_raw], ignore_index=True, sort=False)

    # Export merged raw data
    merged_raw.to_csv(f'raw/{summary_type}_merged_{RA}.csv', index=False)


## Trimming data ##

def TrimData(summary_type: str, RA: str):
    '''
    Function that trims ra summary data
    '''
    ra_noise_trim = pd.read_csv(f'raw/{summary_type}_noise_{RA}.csv')
    ra_no_noise_trim = pd.read_csv(f'raw/{summary_type}_no_noise_{RA}.csv')
    ra_merged_trim = pd.read_csv(f'raw/{summary_type}_merged_{RA}.csv')
    
    if summary_type == 'first' or summary_type == 'switch':
        keep_columns = ['summary_1', 'summary_2', 'window_number', 'cooperate', 'treatment']
    else:
        keep_columns = ['summary_1', 'summary_2', 'window_number', 'unilateral_cooperate', 'unilateral_defect', 'treatment']
    
    ra_noise_trim = ra_noise_trim[ra_noise_trim.intersection(keep_columns)]
    ra_no_noise_trim = ra_no_noise_trim[ra_no_noise_trim.intersection(keep_columns)]
    ra_merged_trim = ra_merged_trim[ra_merged_trim.intersection(keep_columns)]
    
    ra_noise_trim = f.remove_summary_commas(ra_noise_trim)
    ra_no_noise_trim = f.remove_summary_commas(ra_no_noise_trim)
    ra_merged_trim = f.remove_summary_commas(ra_merged_trim)
    
    ra_no_noise_trim.to_csv(f'trim/{summary_type}_no_noise_{RA}.csv', index=False)
    ra_noise_trim.to_csv(f'trim/{summary_type}_noise_{RA}.csv', index=False)
    ra_merged_trim.to_csv(f'trim/{summary_type}_merged_{RA}.csv', index=False)


## Creating test dataframes ##

# Importing trim data
ra_no_noise = pd.read_csv(f'trim_data/RAsum_no_noise_v{version}.csv')
ra_noise = pd.read_csv(f'trim_data/RAsum_noise_v{version}.csv')
ra_merge = pd.read_csv(f'trim_data/RAsum_merged_v{version}.csv')
ra_merge = ra_merge.drop(['treatment'], axis=1)

# Creating ucoop and udef test data
ra_no_noise_coop, ra_no_noise_def = f.test_summaries(ra_no_noise, type=summary_type)   # ucoop & udef no-noise test data
ra_noise_coop, ra_noise_def = f.test_summaries(ra_noise, type=summary_type)    # ucoop & udef noise test data
ra_merge_coop, ra_merge_def = f.test_summaries(ra_merge, type=summary_type)    # ucoop & udef merged test data

# Export
if summary_type != 'FAR':
    ra_no_noise_coop.to_csv(f'test_data/RAsum_no_noise_ucoop_v{version}.csv', index=False)
    ra_no_noise_def.to_csv(f'test_data/RAsum_no_noise_udef_v{version}.csv', index=False)
    ra_noise_coop.to_csv(f'test_data/RAsum_noise_ucoop_v{version}.csv', index=False)
    ra_noise_def.to_csv(f'test_data/RAsum_noise_udef_v{version}.csv', index=False)
    ra_merge_coop.to_csv(f'test_data/RAsum_merged_ucoop_v{version}.csv', index=False)
    ra_merge_def.to_csv(f'test_data/RAsum_merged_udef_v{version}.csv', index=False)
else:
    ra_no_noise_coop.to_csv(f'test_data/RAsum_no_noise_coop_v{version}.csv', index=False)
    ra_no_noise_def.to_csv(f'test_data/RAsum_no_noise_def_v{version}.csv', index=False)
    ra_noise_coop.to_csv(f'test_data/RAsum_noise_coop_v{version}.csv', index=False)
    ra_noise_def.to_csv(f'test_data/RAsum_noise_def_v{version}.csv', index=False)
    ra_merge_coop.to_csv(f'test_data/RAsum_merged_coop_v{version}.csv', index=False)
    ra_merge_def.to_csv(f'test_data/RAsum_merged_def_v{version}.csv', index=False)

ra_noise.to_csv(f'test_data/RAsum_merged_noise_v{version}.csv', index=False)
ra_no_noise.to_csv(f'test_data/RAsum_merged_no_noise_v{version}.csv', index=False)