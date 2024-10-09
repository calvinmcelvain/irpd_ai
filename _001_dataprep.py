'''
Preparing data and creating test data for:
- Reaw experimental data
- RA created summaries
'''

import sys
import os
os.chdir('/Users/fogellmcmuffin/Dropbox/ai_irpd_coding/')   # Working Dir.

# Modules & Packages
sys.path.append('/Users/fogellmcmuffin/git_repos/irpd_ai/')
import pandas as pd
import functions as f
import importlib
importlib.reload(f)


###################
## Summary data ##
###################

# Set this before running
summary_type = 'FAR'

## Merging raw summary data ##

# Import raw data
version = f.get_summary_version()
ra_noise_raw = pd.read_csv(f'raw_data/RAsum_noise_v{version}.csv')
ra_no_noise_raw = pd.read_csv(f'raw_data/RAsum_no_noise_v{version}.csv')

# Merging
merged_raw = pd.concat([ra_no_noise_raw, ra_noise_raw], ignore_index=True, sort=False)

# Export merged raw data
merged_raw.to_csv(f'raw_data/RAsum_merged_v{version}.csv', index=False)

## Trimming data ##

# Import raw data
ra_noise_trim = pd.read_csv(f'raw_data/RAsum_noise_v{version}.csv')
ra_no_noise_trim = pd.read_csv(f'raw_data/RAsum_no_noise_v{version}.csv')

# Trimming columns
keep_columns = ['summary', 'window_number', 'unilateral_cooperate', 'unilateral_defect'] if summary_type != 'FAR' else ['summary', 'window_number', 'cooperate']
ra_noise_trim = ra_noise_trim[keep_columns]
ra_no_noise_trim = ra_no_noise_trim[keep_columns]

# Creating merge df for RA summaries
ra_no_noise_trim['treatment'] = 1
ra_noise_trim['treatment'] = 0
ra_merged_trim = pd.concat([ra_no_noise_trim, ra_noise_trim], ignore_index=True, sort=False)

# Removing commas from summaries
ra_noise_trim = f.remove_summary_commas(ra_noise_trim)
ra_no_noise_trim = f.remove_summary_commas(ra_no_noise_trim)
ra_merged_trim = f.remove_summary_commas(ra_merged_trim)

# Removing 'treatment' var from ind. treatment dfs
ra_no_noise_trim = ra_no_noise_trim.drop(['treatment'], axis=1)
ra_noise_trim = ra_noise_trim.drop(['treatment'], axis=1)

# Export trim data
ra_no_noise_trim.to_csv(f'trim_data/RAsum_no_noise_v{version}.csv', index=False)
ra_noise_trim.to_csv(f'trim_data/RAsum_noise_v{version}.csv', index=False)
ra_merged_trim.to_csv(f'trim_data/RAsum_merged_v{version}.csv', index=False)


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

###########################
## Raw Experimental Data ##
###########################

## Trimming data ##

# Import raw data
no_noise = pd.read_csv('raw_data/no_noise.csv')
noise = pd.read_csv('raw_data/noise.csv')

# Dropping columns & trimming 'noise' & 'no_noise' dfs
common_drops = ['original_subject', 'super_game_match', 'time_message', 'Unnamed: 0']   # Columns to drop
noise = f.trim(noise, common_drops)
no_noise = f.trim(no_noise, common_drops + ['critical']) # Also dropping 'critical' column

# Creating merged df using 'noise' & 'no_noise' dfs
no_noise['treatment'] = 0
noise['treatment'] = 1
merged = pd.concat([no_noise, noise], ignore_index=True, sort=False)

# Export trim data
no_noise.to_csv('trim_data/no_noise.csv', index=False)
noise.to_csv('trim_data/noise.csv', index=False)
merged.to_csv('trim_data/merged.csv', index=False)

## Creating test dataframes ##

# Importing trim data
merged = pd.read_csv('trim_data/merged.csv')
no_noise = pd.read_csv('trim_data/no_noise.csv')
noise = pd.read_csv('trim_data/noise.csv')

# Creating ucoop and udef test data
no_noise_ucoop, no_noise_udef = f.ucoop_udef_windows(no_noise)   # ucoop & udef no-noise test data
noise_ucoop, noise_udef = f.ucoop_udef_windows(noise)    # ucoop & udef noise test data
merge_ucoop, merge_udef = f.ucoop_udef_windows(merged)    # ucoop & udef merged test data

# Export
no_noise_ucoop.to_csv(f'test_data/no_noise_ucoop.csv', index=False)
no_noise_udef.to_csv(f'test_data/no_noise_udef.csv', index=False)
noise_ucoop.to_csv(f'test_data/noise_ucoop.csv', index=False)
noise_udef.to_csv(f'test_data/noise_udef.csv', index=False)
merge_ucoop.to_csv(f'test_data/merged_ucoop.csv', index=False)
merge_udef.to_csv(f'test_data/merged_udef.csv', index=False)