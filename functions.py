docs = '''
Functions module
'''

import os

# Modules & Packages
import pandas as pd
from datetime import datetime
import ast
import re 


#############################
## Simple Helper Functions ##
#############################

def file_to_string(file_path: str):
  '''
  Reads file and returns its contents as a string
  '''
  with open(file_path, 'r') as file:
    k = file.read()
  return str(k)


def write_file(file_path: str, file_write: any):
  '''
  Writes files to path
  '''
  with open(file_path, 'w') as file:
    file.write(file_write)     


def get_test_name(summary_type: str, test_type='test', previous=False):
  '''
  Returns the test name depending on:
  - summary_type: The type of summary being used
  - test_type: The type of test, either a 'test' or 'subtest'
  - previous: If True, it returns the current/latest test name. If False, it returns the name for the next test (latest test number + 1)
  '''
  if test_type == 'test':
    test_path = f'output/{summary_type}/'
    
    tests = os.listdir(test_path) if os.path.exists(test_path) else []
    test_numbers = [int(re.findall(r'\d+', name)[0]) for name in tests if name.startswith('test_') and re.findall(r'\d+', name)]
    
    if not test_numbers:
      return 'test_1'
    
    new_test_number = max(test_numbers) + 1 if previous == False else max(test_numbers)
    
    test_name = f"test_{new_test_number}"
  elif test_type == 'subtest':
    subtest_path = 'output/_subtests/'
    
    subtests = [int(k) for k in os.listdir(subtest_path) if k.isdigit()]
    new_test_number = max(subtests) + 1 if previous == False else max(subtests)
    
    test_name = str(new_test_number)
  return test_name


def get_window_types(summary_type: str):
  '''
  Get window types based on summary_type.
  '''
  if summary_type == 'first' or summary_type == 'switch':
    return 'coop', 'def'
  else:
    return 'ucoop', 'udef'


def get_test_directory(summary_type: str, test_type: str, test: str):
  '''
  Create a test directory based on the test and summary type.
  '''
  if test_type == 'test':
    test_dir = os.path.join(f'output/{summary_type}/', test)
  else:
    test_dir = os.path.join('output/_subtests/', test)
  os.makedirs(test_dir, exist_ok=True)
  return test_dir


def write_test_info(test_info: dict, directory: str, test_number: str, model_info: any, stage_number: str, data_file=''):
  '''
  Function to write test info (for each Stage)
  '''
  test_info_file = "MODEL INFORMATION: \n\n"
  test_info_file += f" Model: {model_info.MODEL} \n"
  test_info_file += f" Termperature: {model_info.TEMPERATURE} \n"
  test_info_file += f" Max-tokens: {model_info.MAX_TOKENS} \n"
  test_info_file += f" Seed: {model_info.SEED} \n"
  test_info_file += f" Top-p: {model_info.TOP_P} \n"
  test_info_file += f" Frequency penalty: {model_info.FREQUENCY_PENALTY} \n"
  test_info_file += f" Presence penalty: {model_info.PRESENCE_PENALTY} \n\n\n"
  
  # Initialize a string to store formatted test info content
  test_info_file += "TEST INFORMATION:\n\n"
  
  # Getting test time
  first_key = next(iter(test_info))
  test_info_file += f' Test date/time: {datetime.fromtimestamp(test_info[first_key].created).strftime('%Y-%m-%d %H:%M:%S')} \n'
  test_info_file += f' Data file: {data_file} \n'
  test_info_file += f' System fingerprint: {test_info[first_key].system_fingerprint}\n'

  # Loop through each window
  for key, value in test_info.items():
    test_info_file += f" {key.upper()} PROMPT USAGE: \n"
    test_info_file += f"   Completion tokens: {value.usage.completion_tokens} \n"
    test_info_file += f"   Prompt tokens: {value.usage.prompt_tokens} \n"
    test_info_file += f"   Total tokens: {value.usage.total_tokens} \n"
    

  # Define the directory and file path for the test info file
  info_dir = os.path.join(directory, f't{test_number}_stg{stage_number}_test_info.txt')

  # Write the formatted test info to the file
  write_file(file_path=info_dir, file_write=test_info_file)


######################################
## Data Prep and Trimming Functions ##
######################################

def remove_summary_commas(df):
  '''
  Function to remove commas in the summaries for better GPT readability
  '''
  summary_columns = ['summary_1', 'summary_2']
  df_column = df.columns.intersection(summary_columns)
  df[df_column] = df[df_column].replace(',', '', regex=False)
  return df


def create_instance_dfs(df_set, summary_type: str):
  '''
  Function that takes a df set (no-noise, noise, and merged trim dfs), and returns instances for each df type
  '''
  instance_set = []
  if summary_type == 'first' or summary_type == 'switch':
    for df in df_set:
      df_type1 = df.loc[(df['cooperate'] == 1)].drop(['cooperate'], axis=1)
      df_type2 = df.loc[(df['cooperate'] == 0)].drop(['cooperate'], axis=1)
      instance_set.append(df_type1)
      instance_set.append(df_type2)
  else:
    for df in df_set:
      df_type1 = df.loc[(df['unilateral_cooperate'] == 1)].drop(['unilateral_cooperate', 'unilateral_defect'], axis=1)
      df_type2 = df.loc[(df['unilateral_defect'] == 1)].drop(['unilateral_cooperate', 'unilateral_defect'], axis=1)
      instance_set.append(df_type1)
      instance_set.append(df_type2)
  
  return instance_set


#################################
## Final Output Data Functions ##
#################################

def extract_dict_from_file(file_path: str):
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


def response_df(response_dir: str, test_df: str):
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


def category_prefix(df, summary_type: str, prefix: str):
  '''
  Adds prefixes to categories names in the output of the response_df
  '''
  if summary_type == 'first' or summary_type == 'switch':
    remove_columns = ['summary_1', 'summary_2', 'cooperation', 'window_number', 'gpt_reasoning']
  else:
    remove_columns = ['summary_1', 'summary_2', 'unilateral_cooperation', 'window_number', 'gpt_reasoning']
  
  df_remove_cols = df.columns.intersection(remove_columns)
  df_dropped = df.drop(columns=df_remove_cols)
  category_columns = df_dropped.columns.to_list()
  
  rename_dict = {col: f'{prefix}_{col}' for col in category_columns}
  df = df.rename(columns=rename_dict)
      
  return df


def final_merge_df(final_df, og_df, summary_type: str):
  '''
  Function to get the original test dataframe variables with gpt codings & rationale
  '''
  if summary_type == 'first' or summary_type == 'switch':
    final_df = final_df.drop(columns=final_df.columns.intersection(['summary_1', 'summary_2', 'cooperation']))
  else:
    final_df = final_df.drop(columns=final_df.columns.intersection(['summary_1', 'summary_2', 'unilateral_cooperation']))
  
  merged_df = pd.merge(og_df, final_df, on='window_number')
  return merged_df
