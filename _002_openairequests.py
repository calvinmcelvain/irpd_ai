docs = '''
Test functions
'''

import sys
import os
os.chdir('/Users/fogellmcmuffin/Dropbox/ai_irpd_coding/')   # Working Dir.

# Modules & Packages
sys.path.append('/Users/fogellmcmuffin/git_repos/irpd_ai/')
from gpt_key import key
import functions as f
import gpt_module
import pandas as pd
import importlib
importlib.reload(f)
importlib.reload(gpt_module)


########################
## GPT Model Settings ##
########################


model = gpt_module.GPT(
    api_key=key,
    organization='org-WLFAmqjnKmywM0wd6loMyGJq',
    project='proj_vOr6WeeCFk5IjZLCdksFLWUd',
)

# Set general (non-changing) settings
model.set_model('gpt-4o-2024-05-13')
model.set_temperature(0)
model.set_frequency_penalty(0)
model.set_presence_penalty(0)
model.set_top_p(1)
model.set_seed(1240034)


####################
## Test functions ##
####################

def stage_1_output(treatment: str, summary_type: str, RA: str, test_type='test'):
    '''
    Stage 1 function
    '''
    # Getting prefixes base don summary type
    type_1, type_2 = f.get_window_types(summary_type=summary_type)
    
    # System prompts
    sys_typ1 = f.file_to_string(file_path=f'prompts/{summary_type}/stg_1_{treatment}_{type_1}.md')
    sys_typ2 = f.file_to_string(file_path=f'prompts/{summary_type}/stg_1_{treatment}_{type_2}.md')

    # Summary data (User prompts)
    df_typ1 = pd.read_csv(f'data/test/{summary_type}_{treatment}_{RA}_{type_1}.csv')
    df_typ2 = pd.read_csv(f'data/test/{summary_type}_{treatment}_{RA}_{type_2}.csv')
    data_file = f'{summary_type}_{treatment}_{RA}_{type_1}.csv & {summary_type}_{treatment}_{RA}_{type_2}.csv'    # For test info

    df_typ1['window_number'] = df_typ1['window_number'].astype(int)   # Making sure window number is an integer
    df_typ2['window_number'] = df_typ2['window_number'].astype(int)

    user_typ1 = str(df_typ1.to_dict('records')) # Turning data into a list of dictionaries, then to string
    user_typ2 = str(df_typ2.to_dict('records'))

    # Aggregating prompts
    window_prompts = [[type_1, sys_typ1, user_typ1], [type_2, sys_typ2, user_typ2]]

    # Making test directory
    test = f.get_test_name(summary_type=summary_type, test_type=test_type)
    test_number = test[5:] if test_type == 'test' else test
    test_dir = f.get_test_directory(summary_type=summary_type, test_type=test_type, test=test)

    # GPT requests
    info_data = {type_1: 0, type_2: 0}      # Initializing info data
    for i in window_prompts:  # Requests for instances
        inst_dir = os.path.join(test_dir, f'stage_1_{i[0]}') # Creating ind. instance directory
        os.makedirs(inst_dir, exist_ok=False)

        # Prompts
        sys_prmpt = i[1]
        user_prmpt = i[2]

        # GPT request output
        model.set_max_tokens(2000)
        response, response_data = model.GPT_response(sys=sys_prmpt, user=user_prmpt)
        info_data[i[0]] = response_data     # Appending response data
        
        # Creating paths for prompts & GPT response
        sys_prmpt_path = os.path.join(inst_dir, f't{test_number}_stg_1_{i[0]}_sys_prmpt.txt')
        user_prmpt_path = os.path.join(inst_dir, f't{test_number}_stg_1_{i[0]}_user_prmpt.txt')
        response_path = os.path.join(inst_dir, f't{test_number}_stg_1_{i[0]}_response.txt')

        # Writing .txt files for prompts & GPT response
        f.write_file(file_path=sys_prmpt_path, file_write=sys_prmpt)
        f.write_file(file_path=user_prmpt_path, file_write=user_prmpt)
        f.write_file(file_path=response_path, file_write=str(response))
    
    f.write_test_info(test_info=info_data, directory=test_dir, test_number=test_number, model_info=model, stage_number = '1', data_file=data_file)    # Writing test information
    return print("Stage 1 Complete")


def stage_1r_output(treatment: str, summary_type: str, test_type='test'):
    '''
    Stage 1 refinement function
    '''
    # Getting prefixes base don summary type
    type_1, type_2 = f.get_window_types(summary_type=summary_type)
    
    # Getting test directory
    test = f.get_test_name(summary_type=summary_type, test_type=test_type, previous=True)
    test_number = test[5:] if test_type == 'test' else test
    test_dir = f.get_test_directory(summary_type=summary_type, test_type=test_type, test=test)
    
    # System prompts
    sys_typ1 = f.file_to_string(file_path=f'prompts/{summary_type}/stg_1r_{treatment}_{type_1}.md')
    sys_typ2 = f.file_to_string(file_path=f'prompts/{summary_type}/stg_1r_{treatment}_{type_2}.md')
    
    # User prompts
    stg_1_typ1_dir = os.path.join(test_dir, f'stage_1_{type_1}/')
    stg_1_typ2_dir = os.path.join(test_dir, f'stage_1_{type_2}/')
    typ1_response_path = os.path.join(stg_1_typ1_dir, f't{test_number}_stg_1_{type_1}_response.txt')
    typ2_response_path = os.path.join(stg_1_typ2_dir, f't{test_number}_stg_1_{type_2}_response.txt')
    user_typ1 = f.file_to_string(file_path=typ1_response_path)
    user_typ2 = f.file_to_string(file_path=typ2_response_path)
    
    # Aggregating prompts
    window_prompts = [[type_1, sys_typ1, user_typ1], [type_2, sys_typ2, user_typ2]]
    
    # GPT requests
    info_data = {type_1: 0, type_2: 0}      # Initializing info data
    for i in window_prompts:  # Requests for instances
        inst_dir = os.path.join(test_dir, f'stage_1r_{i[0]}') # Creating ind. instance directory
        os.makedirs(inst_dir, exist_ok=False)

        # Prompts
        sys_prmpt = i[1]
        user_prmpt = i[2]

        # GPT request output
        model.set_max_tokens(2000)
        response, response_data = model.GPT_response(sys=sys_prmpt, user=user_prmpt)
        info_data[i[0]] = response_data     # Appending response data

        # Creating paths for prompts & GPT response
        sys_prmpt_path = os.path.join(inst_dir, f't{test_number}_stg_1r_{i[0]}_sys_prmpt.txt')
        user_prmpt_path = os.path.join(inst_dir, f't{test_number}_stg_1r_{i[0]}_user_prmpt.txt')
        response_path = os.path.join(inst_dir, f't{test_number}_stg_1r_{i[0]}_response.txt')

        # Writing .txt files for prompts & GPT response
        f.write_file(file_path=sys_prmpt_path, file_write=sys_prmpt)
        f.write_file(file_path=user_prmpt_path, file_write=user_prmpt)
        f.write_file(file_path=response_path, file_write=str(response))
    
    f.write_test_info(test_info=info_data, directory=test_dir, test_number=test_number, model_info=model, stage_number = '1r')    # Writing test information
    return print("Stage 1r Complete")


def stage_2_output(treatment: str, summary_type: str, RA: str, max_windows=None, test_type='test', refinement=True):
    '''
    Stage 2 function
    '''
    # Getting prefixes base don summary type
    type_1, type_2 = f.get_window_types(summary_type=summary_type)
    
    # Getting test directory
    test = f.get_test_name(summary_type=summary_type, test_type=test_type, previous=True)
    test_number = test[5:] if test_type == 'test' else test
    test_dir = f.get_test_directory(summary_type=summary_type, test_type=test_type, test=test)

    # System Prompts
    stg_2_typ1_sys = f.file_to_string(file_path=f'prompts/{summary_type}/stg_2_{treatment}_{type_1}.md') # Getting system prompts for stage 2
    stg_2_typ2_sys = f.file_to_string(file_path=f'prompts/{summary_type}/stg_2_{treatment}_{type_2}.md')

    ## Getting Stage 1 response to merge with Stage 2 system prompt
    stage = '1' if refinement == False else '1r'    # Grabbing the response from either Stage 1 or Stage 1 refined
    stg_1_typ1_dir = os.path.join(test_dir, f'stage_{stage}_{type_1}/')
    stg_1_typ2_dir = os.path.join(test_dir, f'stage_{stage}_{type_2}/')
    typ1_response_path = os.path.join(stg_1_typ1_dir, f't{test_number}_stg_{stage}_{type_1}_response.txt')
    typ2_response_path = os.path.join(stg_1_typ2_dir, f't{test_number}_stg_{stage}_{type_2}_response.txt')

    typ1_response = f.file_to_string(file_path=typ1_response_path)
    typ2_response = f.file_to_string(file_path=typ2_response_path)

    sys_typ1 = stg_2_typ1_sys + '\n' + typ1_response   # Final system prompts
    sys_typ2 = stg_2_typ2_sys + '\n' + typ2_response

    # Summary data (User prompts)
    df_typ1 = pd.read_csv(f'data/test/{summary_type}_{treatment}_{RA}_{type_1}.csv')
    df_typ2 = pd.read_csv(f'data/test/{summary_type}_{treatment}_{RA}_{type_2}.csv')
    data_file = f'{summary_type}_{treatment}_{RA}_{type_1}.csv & {summary_type}_{treatment}_{RA}_{type_2}.csv'    # For test info

    df_typ1['window_number'] = df_typ1['window_number'].astype(int)   # Making sure window number is an integer
    df_typ2['window_number'] = df_typ2['window_number'].astype(int)

    df_typ1 = df_typ1[:max_windows] if max_windows != None else df_typ1  # Adjusting to max windows for Stage 2
    df_typ2 = df_typ2[:max_windows] if max_windows != None else df_typ2

    # Aggregating prompts
    window_prompts = [[type_1, sys_typ1, df_typ1], [type_2, sys_typ2, df_typ2]]
    
    info_data = {type_1: 0, type_2: 0}      # Initializing info data
    for i in window_prompts:
        # Requests for instances
        inst_dir = os.path.join(test_dir, f'stage_2_{i[0]}') # Creating ind. instance directory
        os.makedirs(inst_dir, exist_ok=False)

        sys_prmpt = i[1]    # System prompt for ucoop or udef data
        sys_prmpt_path = os.path.join(inst_dir, f't{test_number}_stg_2_{i[0]}_sys_prmpt.txt') 
        f.write_file(file_path=sys_prmpt_path, file_write=sys_prmpt)

        # Prompt & Response paths
        prompt_path = os.path.join(inst_dir, 'prompts')
        response_path = os.path.join(inst_dir, 'responses')
        os.makedirs(prompt_path, exist_ok=True)
        os.makedirs(response_path, exist_ok=True)
        
        # Initializing tokens to later average
        completion_tok = []
        prompt_tok = []
        total_tok = []

        # Requesting chat completion for each row
        df = i[2]   # Test data for ucoop or udef data
        for k in range(len(df)):
            row = df.iloc[k].to_dict()  # Creating a dictionary for each indv. row
            
            # GPT request output
            model.set_max_tokens(600)
            response, response_data = model.GPT_response(sys=sys_prmpt, user=str(row))
            if k == 1:
                info_data[i[0]] = response_data     # Appending response data
            
            # Appending all values of token usage
            completion_tok.append(response_data.usage.completion_tokens)
            prompt_tok.append(response_data.usage.prompt_tokens)
            total_tok.append(response_data.usage.total_tokens)
            
            # Creating paths for prompts & GPT responses using window_numbers
            window_number = row['window_number']
            user_prmpt_path = os.path.join(prompt_path, f't{test_number}_{window_number}_user_prmpt.txt')
            output_path = os.path.join(response_path, f't{test_number}_{window_number}_response.txt')
            
            # Writing .txt files for prompts & GPT response
            f.write_file(user_prmpt_path, str(row))
            f.write_file(output_path, str(response))

        # Gettting average (mean) of usage tokens and overwriting current value
        mcompletion_tok = sum(completion_tok)
        mprompt_tok = sum(prompt_tok)
        mtotal_tok = sum(total_tok)
        info_data[i[0]].usage.completion_tokens = mcompletion_tok
        info_data[i[0]].usage.prompt_tokens = mprompt_tok
        info_data[i[0]].usage.total_tokens = mtotal_tok
        
        # Prelimaries to final output
        if i[0] == type_1:
            if summary_type == 'first' or summary_type == 'switch':
                df['cooperation'] = 1
            else:
                df['unilateral_cooperation'] = 1
        else:
            if summary_type == 'first' or summary_type == 'switch':
                df['cooperation'] = 0
            else:
                df['unilateral_cooperation'] = 0
        
        typ1_df = f.response_df(response_dir=response_path, test_df=df)  # Coding GPT classifications for ucoop instances
        typ2_df = f.response_df(response_dir=response_path, test_df=df)
        typ1_df = f.category_prefix(df=typ1_df, summary_type=summary_type, prefix=type_1) # Adding ucoop prefix to categories
        typ2_df = f.category_prefix(df=typ1_df, summary_type=summary_type, prefix=type_2)

    # Final output dataframe
    GPT_df = pd.concat([typ1_df, typ2_df], ignore_index=True, sort=False)
    GPT_df = GPT_df.fillna(0)
    og_df = pd.read_csv(f'data/raw/{summary_type}_{treatment}_{RA}.csv')
    final_df = f.final_merge_df(final_df=GPT_df, og_df=og_df, summary_type=summary_type)
    final_out_path = os.path.join(test_dir, f"t{test[5:]}_final_output.csv" if test_type == 'test' else f"{test}_final_output.csv")
    final_df.to_csv(final_out_path, index=False)
    
    f.write_test_info(test_info=info_data, directory=test_dir, test_number=test_number, model_info=model, stage_number='2', data_file=data_file)    # Writing test information
    return print("Stage 2 Complete")


def run_full_test(treatment: str, summary_type: str, RA: str, test_type: str, max_windows: any, refinement: bool):
    stage_1_output(treatment=treatment, summary_type=summary_type, RA=RA, test_type=test_type)
    if refinement == True:
        stage_1r_output(treatment=treatment, summary_type=summary_type, test_type=test_type)
    stage_2_output(treatment=treatment, summary_type=summary_type, RA=RA, test_type=test_type, max_windows=max_windows, refinement=refinement)
    return print('Full Test Complete')
    

# stage_1_output(treatment=, summary_type=, RA=, test_type=)
# stage_1r_output(treatment=, summary_type=, test_type=)
# stage_2_output(treatment=, summary_type=, RA=, test_type=, max_windows=, refinement=)
# run_full_test(treatment=, summary_type=, RA=, test_type=, max_windows=, refinement=)