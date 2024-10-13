'''
Preparing data and creating test data for:
- RA created summaries
'''

import sys
import os
os.chdir('/Users/fogellmcmuffin/Dropbox/ai_irpd_coding/data/')   # Working Dir.

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
        ra_noise_raw.to_csv(f'raw/{summary_type}_noise_both.csv', index=False)
        ra_no_noise_raw.to_csv(f'raw/{summary_type}_no_noise_both.csv', index=False)

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
    
    ra_noise_trim = ra_noise_trim[ra_noise_trim.columns.intersection(keep_columns)]
    ra_no_noise_trim = ra_no_noise_trim[ra_no_noise_trim.columns.intersection(keep_columns)]
    ra_merged_trim = ra_merged_trim[ra_merged_trim.columns.intersection(keep_columns)]
    
    ra_noise_trim = f.remove_summary_commas(ra_noise_trim)
    ra_no_noise_trim = f.remove_summary_commas(ra_no_noise_trim)
    ra_merged_trim = f.remove_summary_commas(ra_merged_trim)
    
    ra_no_noise_trim.to_csv(f'trim/{summary_type}_no_noise_{RA}.csv', index=False)
    ra_noise_trim.to_csv(f'trim/{summary_type}_noise_{RA}.csv', index=False)
    ra_merged_trim.to_csv(f'trim/{summary_type}_merged_{RA}.csv', index=False)


## Creating test dataframes ##

def Test_summaries(summary_type: str, RA: str):
    '''
    Function to seperate the trim data into ucoop/udef or coop/def test dfs based on summary type
    '''
    ra_no_noise = pd.read_csv(f'trim/{summary_type}_no_noise_{RA}.csv')
    ra_noise = pd.read_csv(f'trim/{summary_type}_noise_{RA}.csv')
    ra_merge = pd.read_csv(f'trim/{summary_type}_merged_{RA}.csv')
    ra_merge = ra_merge.drop(['treatment'], axis=1)
    
    trim_dfs = [ra_no_noise, ra_noise, ra_merge]
    test_dfs = f.create_instance_dfs(trim_dfs, summary_type)
    
    type_1, type_2 = f.get_window_types(summary_type)
    test_dfs[0].to_csv(f'test/{summary_type}_no_noise_{RA}_{type_1}.csv', index=False)
    test_dfs[1].to_csv(f'test/{summary_type}_no_noise_{RA}_{type_2}.csv', index=False)
    test_dfs[2].to_csv(f'test/{summary_type}_noise_{RA}_{type_1}.csv', index=False)
    test_dfs[3].to_csv(f'test/{summary_type}_noise_{RA}_{type_2}.csv', index=False)
    test_dfs[4].to_csv(f'test/{summary_type}_merged_{RA}_{type_1}.csv', index=False)
    test_dfs[5].to_csv(f'test/{summary_type}_merged_{RA}_{type_2}.csv', index=False)
