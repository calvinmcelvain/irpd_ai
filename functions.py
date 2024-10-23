docs = '''
Functions module
'''

import os

# Modules & Packages
import pandas as pd
from datetime import datetime
from markdown_pdf import MarkdownPdf, Section
import json
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


#######################################
## GPT Response Formatting Functions ##
#######################################

def stage_1_response_format(responses: list, cat_types: list, file_path: str, stage_1r = False):
  '''
  Function to convert the raw response (structured) of Stage 1 to an easy-to-read PDF and save it to a given file path. Also serves as a function to return a string format for Stage 1r user prompt
  
  Args:
  - responses (dict): A dictionary containing the raw JSON responses.
  - cat_types (list): A list of category types.
  - file_path (str): The file path where the PDF should be saved.
  '''
  # Create a PDF object
  pdf = MarkdownPdf(toc_level=1)
  
  # Adding title
  text = "# Stage 1 Categories \n\n" if stage_1r != True else ''
  
  # Making response to literal JSON object
  for i in range(len(cat_types)):
    response_json = json.loads(responses[i])
    
    # Adding type category sections
    text += f"## {cat_types[i].capitalize()} Categories \n\n"
    cats = response_json['categories']
    for category in range(len(cats)):
      text += f"### {cats[category]['category_name']} \n\n"
      text += f"**Definition**: {cats[category]['definition']}\n\n"
      text += f"**Examples**:\n\n"
      examples = cats[category]['examples']
      for example in range(len(examples)):
        text += f"{example}. Window number: {examples[example]['window_number']}, Reasoning: {examples[example]['reasoning']}\n\n"

  if stage_1r == False:
    pdf.add_section(Section(text, toc=False))
    pdf.save(file_path)
  else:
    return(text)


def stage_1r_response_format(stage_1_responses: list, responses: list, cat_types: list, file_path: str, stage_2 = False):
  '''
  Function to convert the raw response (structured) of Stage 1r to an easy-to-read PDF and save it to a given file path. Also serves as a function to return a string format for Stage 1r user prompt. Also serves as a function to return a string format for Stage 2 system prompt
  '''
  # Create a PDF object
  pdf = MarkdownPdf(toc_level=1)
  
  # Adding title
  text = "# Stage 1 Refined Categories \n\n" if stage_2 == False else ''
  
  # Making response to literal JSON object
  for i in range(len(cat_types)):
    response_json_og = json.loads(stage_1_responses[i])
    response_json = json.loads(responses[i])
    
    # Adding type category sections
    text += f"## {cat_types[i].capitalize()} Final Categories \n\n"
    ref_cats = response_json['final_categories']
    temp_og_cats = response_json_og['categories']
    temp_dict = {item['category_name']: item for item in temp_og_cats}
    og_cats = [temp_dict[item['category_name']] for item in ref_cats]
    final_cats = [temp_dict[item['category_name']] for item in ref_cats if item['keep_decision'] == True]
    if stage_2 == False:
      for category in range(len(ref_cats)):
        text += f"### {ref_cats[category]['category_name']} \n\n"
        text += f"**Definition**: {og_cats[category]['definition']}\n\n"
        if ref_cats[category]['keep_decision'] == True:
          text += f"**Examples**:\n\n"
          examples = og_cats[category]['examples']
          for example in range(len(examples)):
            text += f"{example}. Window number: {examples[example]['window_number']}, Reasoning: {examples[example]['reasoning']}\n\n"
        text += f"**Kept**: {ref_cats[category]['keep_decision']}\n\n"
        text += f"  - *Reasoning*: {ref_cats[category]['reasoning']} \n\n"
    else:
      for category in range(len(final_cats)):
        text += f"### {final_cats[category]['category_name']} \n\n"
        text += f"**Definition**: {final_cats[category]['definition']}\n\n"
        text += f"**Examples**:\n\n"
        examples = final_cats[category]['examples']
        for example in range(len(examples)):
          text += f"{example}. Window number: {examples[example]['window_number']}, Reasoning: {examples[example]['reasoning']}\n\n"

  if stage_2 == False:
    pdf.add_section(Section(text, toc=False))
    pdf.save(file_path)
  else:
    return text

#################################
## Final Output Data Functions ##
#################################

def extract_dict_from_file(file_path: str):
  '''
  Takes Stage 2 GPT reponses and returning json object.
  '''
  response = file_to_string(file_path=file_path)
  response_data = json.loads(response)
  
  for i in response_data['assigned_categories']:
    response_data[i] = 1
  
  del response_data['assigned_categories']
  response_data['window_number'] = int(response_data['window_number'])
  
  return response_data


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
    remove_columns = ['summary_1', 'summary_2', 'unilateral_cooperation', 'window_number', 'reasoning']
  
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
