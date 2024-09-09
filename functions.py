docs = '''
Functions module
'''

import os
os.chdir('/Users/fogellmcmuffin/Documents/ra/team_discussions/AI/')   # Working Dir.

# Modules & Packages
import pandas as pd
import ast
import re 


#############################
## Simple Helper Functions ##
#############################

def file_to_string(file_path):
  '''
  Reads file and returns its contents as a string
  '''
  with open(file_path, 'r') as file:
    k = file.read()
  return str(k)


def write_file(file_path, file_write):
  '''
  Writes files to path
  '''
  with open(file_path, 'w') as file:
    file.write(file_write)     


def get_summary_version():
  '''
  Returns the current summary data version from raw_data directory (string integer)
  '''
  summary_data = [int(k[-5:-4]) for k in os.listdir('raw_data/') if k.startswith('RAsum')]
  version = max(summary_data)
  return str(version)


def get_test_name(test_type='test', previous=False):
  '''
  Returns the test name depending on:
  - test_type: The type of test, either a 'test' or 'subtest'
  - previous: If True, it returns the current/latest test name. If False, it returns the name for the next test (latest test number + 1)
  '''
  if test_type == 'test':
    test_path = 'output/'
    
    tests = os.listdir(test_path)
    test_numbers = [int(re.findall(r'\d+', name)[0]) for name in tests if name.startswith('test_') and re.findall(r'\d+', name)]
    new_test_number = max(test_numbers) + 1 if previous == False else max(test_numbers)
    
    test_name = f"test_{new_test_number}"
  elif test_type == 'subtest':
    subtest_path = 'output/_subtests/'
    
    subtests = [int(k) for k in os.listdir(subtest_path) if k.isdigit()]
    new_test_number = max(subtests) + 1 if previous == False else max(subtests)
    
    test_name = str(new_test_number)
  return test_name


def get_cat_number(stage_dir, previous=False):
  '''
  Returns the next/latest category name (if previous is False, it returns next category name)
  '''
  cats = [i[4:] for i in os.listdir(stage_dir) if i.startswith('cat')]
  cat_numbers = [int(k) for k in cats]
  cat_numbers.append(0)
  
  new_cat_number = max(cat_numbers) + 1 if previous == False else max(cat_numbers)
  
  cat_number = f"cat_{new_cat_number}"
  return cat_number


def create_system_prompt(approach, treatment, stage, window_type, general_task_overview=True, experiment_context=True, summary_context=True, task=True, constraints=True, output_format=True, data_var=True):
  '''
  Function to create the system prompt based on the approach, stage, and treatment
  '''
  if stage == 'stage_2' or stage == 'stage_1_2':
    data_var = False
  elif stage == 'stage_1_1':
    experiment_context = False
  elif stage == 'stage_1r':
    data_var = False
    experiment_context = False
  
  title = '# Task Description and Context'
  s1 = file_to_string(file_path=f'prompts/{approach}/{stage}/general_task_overview.md') if general_task_overview == True else ''
  experiment_context
  s2 = file_to_string(file_path=f'prompts/experiment_context_{treatment}.md') if experiment_context == True else ''
  s3 = file_to_string(file_path=f'prompts/summary_context.md') if summary_context == True else ''
  s4 = file_to_string(file_path=f'prompts/{approach}/{stage}/task.md') if task == True else ''
  s5 = file_to_string(file_path=f'prompts/{approach}/{stage}/constraints.md') if constraints == True else ''
  s6 = file_to_string(file_path=f'prompts/{approach}/{stage}/output_format.md') if output_format == True else ''
  s7 = file_to_string(file_path=f'prompts/data_variable_{window_type}.md') if data_var == True else ''
  sections = [s1, s2, s3, s4, s5, s6, s7]
  
  sys_prompt = title + '\n\n'
  for section in sections:
    sys_prompt += section + '\n' if section != '' else section
  
  return sys_prompt
    


######################################
## Data Prep and Trimming Functions ##
######################################

def remove_summary_commas(df):
  '''
  Function to remove commas in the summaries for better GPT readability
  '''
  df[f'summary'] = df[f'summary'].str.replace(',', '', regex=False)
  return df


def trim(df, cols):
  '''
  Takes raw experimental data, drops columns (defined by cols arg), and trims rows to only cooperate & defect windows
  '''
  df = df.drop(cols, axis=1)
  df = df.loc[(df['window_ucoop'] == 1) | (df['window_udef'] == 1)]
  return df


def test_summaries(df, type):
  '''
  Function to seperate the trim data into ucoop/udef or coop/def test dfs based on summary type
  '''
  df_coop = df.loc[(df['unilateral_cooperate'] == 1)] if type != 'FAR' else df.loc[(df['cooperate'] == 1)]
  df_coop = df_coop.drop(['unilateral_cooperate', 'unilateral_defect'], axis=1) if type != 'FAR' else df_coop.drop(['cooperate'], axis=1)
  
  df_def = df.loc[(df['unilateral_defect'] == 1)] if type != 'FAR' else df.loc[(df['cooperate'] == 0)]
  df_def = df_def.drop(['unilateral_defect', 'unilateral_cooperate'], axis=1) if type != 'FAR' else df_def.drop(['cooperate'], axis=1)
  
  return df_coop, df_def


def ucoop_udef_windows(df):
  '''
  Function to seperate unilateral cooperate and defect windows into 2 different dataframes.
  '''
  df_ucoop = df.loc[df['window_ucoop'] == 1]
  df_ucoop = df_ucoop.drop(['window_ucoop', 'window_udef'], axis=1)
  
  df_udef = df.loc[df['window_udef'] == 1]
  df_udef = df_udef.drop(['window_ucoop', 'window_udef'], axis=1)
  
  return df_ucoop, df_udef


#################################
## Final Output Data Functions ##
#################################

def extract_dict_from_file(file_path):
  '''
  Takes Stage 2 GPT reponses and extracts the Python dictionary. Then creates a new dictionary that includes a binary variables for each assigned category and a variable for the GPT reasoning
  '''
  with open(file_path, 'r', encoding='utf-8') as file:    # Opening response file
      lines = file.readlines()
  
  text = ''.join(lines)
  
  # Extracting dictionary
  start = text.find('{')
  end = text.find('}') + 1
  dict_text = text[start:end]
  
  cat_dict = ast.literal_eval(dict_text) # Turning dictionary string into python dictionary
  
  for i in cat_dict['assigned_categories']:  # Making a binary key for each assigned category
      cat_dict[i] = 1
  
  # Extracting GPT reasoning
  start_keyword = "Step-by-step Reasoning: "
  end_keyword = "Python Dictionary:"

  start_index = text.find(start_keyword) + len(start_keyword)
  end_index = text.find(end_keyword)
  reasoning = text[start_index:end_index].strip()
  
  data_dict = {}
  data_dict['gpt_reasoning'] = reasoning
  
  for key, value in cat_dict.items(): # Making sure gpt_reasoning is the first key
      data_dict[key] = value
  
  return data_dict


def response_df(response_dir, test_df):
  '''
  Takes each GPT response for Stage 2 and runs it through extract_dict_from_file funtion. Returns a dataframe of the category codings and GPT reasoning (to do this, also need the test data used for stage 2, i.e., test_df)
  '''
  resp_list = []
  
  for file in os.listdir(response_dir):
      file_path = os.path.join(response_dir, file)
      reponse_dict = extract_dict_from_file(file_path)
      resp_list.append(reponse_dict)
  
  df = pd.DataFrame.from_records(resp_list)
  df = df.drop(['assigned_categories'], axis=1)
  df = df.fillna(0)
  
  df = pd.merge(test_df, df, on='window_number', how='outer')
  
  return df


def ucoop_udef_rename(df, prefix):
  '''
  Adds 'ucoop' and 'udef' prefixes to categories names in the output of the response_df
  '''
  remove_columns = ['summary', 'unilateral_cooperation', 'window_number', 'gpt_reasoning']
  df_dropped = df.drop(columns=remove_columns)
  category_columns = df_dropped.columns.to_list()
  
  rename_dict = {col: f'{prefix}_{col}' for col in category_columns}
  df = df.rename(columns=rename_dict)
      
  return df


def coop_def_rename(df, prefix):
  '''
  Adds 'coop' and 'def' prefixes to categories names in the output of the response_df
  '''
  remove_columns = ['summary', 'cooperation', 'window_number', 'gpt_reasoning']
  df_dropped = df.drop(columns=remove_columns)
  category_columns = df_dropped.columns.to_list()
  
  rename_dict = {col: f'{prefix}_{col}' for col in category_columns}
  df = df.rename(columns=rename_dict)
      
  return df


def final_merge_df(final_df, og_df):
  '''
  Function to get the original test dataframe variables with gpt codings & rationale
  '''
  final_df = final_df.drop(['summary', 'unilateral_cooperation'], axis=1)
  merged_df = pd.merge(og_df, final_df, on='window_number')
  return merged_df


def final_merge_df_FAR(final_df, og_df):
  '''
  Function to get the original test dataframe variables with gpt codings & rationale for FAR coding
  '''
  final_df = final_df.drop(['summary', 'cooperation'], axis=1)
  merged_df = pd.merge(og_df, final_df, on='window_number')
  return merged_df