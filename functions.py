import os
import pandas as pd
import numpy as np
import ast 

# Working Dir.
os.chdir('/Users/fogellmcmuffin/Documents/ra/team_discussions/AI/')


###########################
       ## Functions ##
###########################

### Helper Functions ###

def file_to_string(file_path):  # File read-to-string functions
  with open(file_path, 'r') as file:
    k = file.read()
  return str(k)


def write_file(file_path, file_write):  # File write function
  with open(file_path, 'w') as file:
    file.write(file_write)     


def get_summary_version():    # Function to get the version number of summary data
  summary_data = [k for k in os.listdir('raw_data/') if k.startswith('RAsum')]
  version = summary_data[0][-5:-4]
  return version


def get_test_dir(test_type='test', cycle=False):  # Function to get test directory
  if test_type == 'test':
    test_path = 'output/'
    tests = [i for i in os.listdir(test_path) if i.startswith('test')]
    
    test_numbers = [int(k[5:]) for k in tests]
    new_test_number = max(test_numbers) + 1 if cycle == False else max(test_numbers)
    
    test_dir = f"test_{new_test_number}"
  elif test_type == 'subtest':
    subtest_path = 'output/_subtests/'
    
    subtests = [int(k) for k in os.listdir(subtest_path) if k.isdigit()]
    new_test_number = max(subtests) + 1 if cycle == False else max(subtests)
    
    test_dir = str(new_test_number)
  return test_dir


def get_cat_number(stage_dir, previous=False):      # Function to get category number
    cats = [i[4:] for i in os.listdir(stage_dir) if i.startswith('cat')]
    cat_numbers = [int(k) for k in cats]
    cat_numbers.append(0)
    
    new_cat_number = max(cat_numbers) + 1 if previous == False else max(cat_numbers)
    
    cat_number = f"cat_{new_cat_number}"
    return cat_number


### Final output functions ###

def extract_dict_from_file(file_path):  # Extracting info from GPT response text files
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


def response_df(response_dir, test_df):  # Turning dictionary list into GPT coded dataframe
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


def ucoop_udef_rename(df, prefix):  # Function to add ucoop or udef prefix to created category columns
    remove_columns = ['summary', 'unilateral_cooperation', 'window_number', 'gpt_reasoning']
    df_dropped = df.drop(columns=remove_columns)
    category_columns = df_dropped.columns.to_list()
    
    rename_dict = {col: f'{prefix}_{col}' for col in category_columns}
    df = df.rename(columns=rename_dict)
        
    return df