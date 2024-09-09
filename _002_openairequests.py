docs = '''
Test functions
'''

import sys
import os
os.chdir('/Users/fogellmcmuffin/Documents/ra/team_discussions/AI/')     # Working Dir.

# Modules & Packages
sys.path.append('/Users/fogellmcmuffin/Documents/ra/team_discussions/AI/code/')
from gpt_module import GPT
from gpt_key import key
import functions as f
import pandas as pd
import importlib
importlib.reload(f)


########################
## GPT Model Settings ##
########################


model = GPT(
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


####################
## Test functions ##
####################

def stage_1_output(treatment, summary_type, test_type='test'):
    '''
    Stage 1 function
    '''
    # Getting prefixes base don summary type
    if summary_type == 'FAR':
        type_1 = 'coop'
        type_2 = 'def'
    else:
        type_1 = 'ucoop'
        type_2 = 'udef'
    
    # System prompts
    sys_typ1 = f.create_system_prompt(approach='approach_1', treatment=treatment, stage='stage_1', window_type=type_1)
    sys_typ2 = f.create_system_prompt(approach='approach_1', treatment=treatment, stage='stage_1', window_type=type_2)

    # Summary data (User prompts)
    version = f.get_summary_version()
    df_typ1 = pd.read_csv(f'test_data/RAsum_{treatment}_{type_1}_v{version}.csv')
    df_typ2 = pd.read_csv(f'test_data/RAsum_{treatment}_{type_2}_v{version}.csv')

    df_typ1['window_number'] = df_typ1['window_number'].astype(int)   # Making sure window number is an integer
    df_typ2['window_number'] = df_typ2['window_number'].astype(int)

    user_typ1 = str(df_typ1.to_dict('records')) # Turning data into a list of dictionaries, then to string
    user_typ2 = str(df_typ2.to_dict('records'))

    # Aggregating prompts
    window_prompts = [[type_1, sys_typ1, user_typ1], [type_2, sys_typ2, user_typ2]]

    # Making test directory
    if test_type == 'test':
        test = f.get_test_name()
        test_dir = os.path.join('output/', test)
        info_path = os.path.join(test_dir, f't{test[5:]}_test_info.txt')  # Test info path
    elif test_type == 'subtest':
        test = f.get_test_name(test_type='subtest')
        test_dir = os.path.join('output/_subtests/', test)
        info_path = os.path.join(test_dir, f'{test}__subtest_info.txt')  # Test info path

    os.makedirs(test_dir, exist_ok=False)

    # Test info
    info = model.test_info(test=f"Test {test[5:]}" if test_type == 'test' else f"Subtest {test}", data_name=f'RAsum_{treatment}_{type_1}_v{version}.csv & RAsum_{treatment}_{type_2}_v{version}.csv')
    f.write_file(file_path=info_path, file_write=info)

    # GPT requests
    for i in window_prompts:  # Requests for instances
        inst_dir = os.path.join(test_dir, f'stage_1_{i[0]}') # Creating ind. instance directory
        os.makedirs(inst_dir, exist_ok=False)

        # Prompts
        sys_prmpt = i[1]
        user_prmpt = i[2]

        # GPT request output
        model.set_max_tokens(2000)
        output = model.GPT_response(sys=sys_prmpt, user=user_prmpt)

        # Creating paths for prompts & GPT response
        if test_type == 'test':
            sys_prmpt_path = os.path.join(inst_dir, f't{test[5:]}_stg_1_{i[0]}_sys_prmpt.txt')
            user_prmpt_path = os.path.join(inst_dir, f't{test[5:]}_stg_1_{i[0]}_user_prmpt.txt')
            response_path = os.path.join(inst_dir, f't{test[5:]}_stg_1_{i[0]}_response.txt')
        elif test_type == 'subtest':
            sys_prmpt_path = os.path.join(inst_dir, f'{test}_stg_1_{i[0]}_sys_prmpt.txt')
            user_prmpt_path = os.path.join(inst_dir, f'{test}_stg_1_{i[0]}_user_prmpt.txt')
            response_path = os.path.join(inst_dir, f'{test}_stg_1_{i[0]}_response.txt')

        # Writing .txt files for prompts & GPT response
        f.write_file(file_path=sys_prmpt_path, file_write=sys_prmpt)
        f.write_file(file_path=user_prmpt_path, file_write=user_prmpt)
        f.write_file(file_path=response_path, file_write=str(output))
    return print("Stage 1 Complete")


def stage_1r_output(treatment, summary_type, test_type='test'):
    '''
    Stage 1 refinement function
    '''
    # Getting prefixes base don summary type
    if summary_type == 'FAR':
        type_1 = 'coop'
        type_2 = 'def'
    else:
        type_1 = 'ucoop'
        type_2 = 'udef'
    
    # Getting test directory
    if test_type == 'test':
        test = f.get_test_name(previous=True)
        test_dir = os.path.join('output/', test)
    elif test_type == 'subtest':
        test = f.get_test_name(test_type='subtest', previous=True)
        test_dir = os.path.join('output/_subtests/', test)
    
    # System prompts
    sys_typ1 = f.create_system_prompt(approach='approach_1', treatment=treatment, stage='stage_1r', window_type=type_1)
    sys_typ2 = f.create_system_prompt(approach='approach_1', treatment=treatment, stage='stage_1r', window_type=type_2)
    
    # User prompts
    stg_1_typ1_dir = os.path.join(test_dir, f'stage_1_{type_1}/')
    stg_1_typ2_dir = os.path.join(test_dir, f'stage_1_{type_2}/')
    if test_type == 'test':
        typ1_response_path = os.path.join(stg_1_typ1_dir, f't{test[5:]}_stg_1_{type_1}_response.txt')
        typ2_response_path = os.path.join(stg_1_typ2_dir, f't{test[5:]}_stg_1_{type_2}_response.txt')
    elif test_type == 'subtest':
        typ1_response_path = os.path.join(stg_1_typ1_dir, f'{test}_stg_1_{type_1}_response.txt')
        typ2_response_path = os.path.join(stg_1_typ2_dir, f'{test}_stg_1_{type_2}_response.txt')

    user_typ1 = f.file_to_string(file_path=typ1_response_path)
    user_typ2 = f.file_to_string(file_path=typ2_response_path)
    
    # Aggregating prompts
    window_prompts = [[type_1, sys_typ1, user_typ1], [type_2, sys_typ2, user_typ2]]
    
    # GPT requests
    for i in window_prompts:  # Requests for instances
        inst_dir = os.path.join(test_dir, f'stage_1r_{i[0]}') # Creating ind. instance directory
        os.makedirs(inst_dir, exist_ok=False)

        # Prompts
        sys_prmpt = i[1]
        user_prmpt = i[2]

        # GPT request output
        model.set_max_tokens(2000)
        output = model.GPT_response(sys=sys_prmpt, user=user_prmpt)

        # Creating paths for prompts & GPT response
        if test_type == 'test':
            sys_prmpt_path = os.path.join(inst_dir, f't{test[5:]}_stg_1r_{i[0]}_sys_prmpt.txt')
            user_prmpt_path = os.path.join(inst_dir, f't{test[5:]}_stg_1r_{i[0]}_user_prmpt.txt')
            response_path = os.path.join(inst_dir, f't{test[5:]}_stg_1r_{i[0]}_response.txt')
        elif test_type == 'subtest':
            sys_prmpt_path = os.path.join(inst_dir, f'{test}_stg_1r_{i[0]}_sys_prmpt.txt')
            user_prmpt_path = os.path.join(inst_dir, f'{test}_stg_1r_{i[0]}_user_prmpt.txt')
            response_path = os.path.join(inst_dir, f'{test}_stg_1r_{i[0]}_response.txt')

        # Writing .txt files for prompts & GPT response
        f.write_file(file_path=sys_prmpt_path, file_write=sys_prmpt)
        f.write_file(file_path=user_prmpt_path, file_write=user_prmpt)
        f.write_file(file_path=response_path, file_write=str(output))
    return print("Stage 1r Complete")


def stage_2_output(treatment, summary_type, max_windows=None, test_type='test', refinement=True):
    '''
    Stage 2 function
    '''
    # Getting prefixes base don summary type
    if summary_type == 'FAR':
        type_1 = 'coop'
        type_2 = 'def'
    else:
        type_1 = 'ucoop'
        type_2 = 'udef'
    
    # Getting test directory
    if test_type == 'test':
        test = f.get_test_name(previous=True)
        test_dir = os.path.join('output/', test)
    elif test_type == 'subtest':
        test = f.get_test_name(test_type='subtest', previous=True)
        test_dir = os.path.join('output/_subtests/', test)

    # System Prompts
    stg_2_typ1_sys = f.create_system_prompt(approach='approach_1', treatment=treatment, stage='stage_2', window_type=type_1) # Getting system prompts for stage 2
    stg_2_typ2_sys = f.create_system_prompt(approach='approach_1', treatment=treatment, stage='stage_2', window_type=type_2)

    ## Getting Stage 1 response to merge with Stage 2 system prompt
    stage = '1' if refinement == False else '1r'    # Grabbing the response from either Stage 1 or Stage 1 refined
    stg_1_typ1_dir = os.path.join(test_dir, f'stage_{stage}_{type_1}/')
    stg_1_typ2_dir = os.path.join(test_dir, f'stage_{stage}_{type_2}/')
    if test_type == 'test':
        typ1_response_path = os.path.join(stg_1_typ1_dir, f't{test[5:]}_stg_{stage}_{type_1}_response.txt')
        typ2_response_path = os.path.join(stg_1_typ2_dir, f't{test[5:]}_stg_{stage}_{type_2}_response.txt')
    elif test_type == 'subtest':
        typ1_response_path = os.path.join(stg_1_typ1_dir, f'{test}_stg_{stage}_{type_1}_response.txt')
        typ2_response_path = os.path.join(stg_1_typ2_dir, f'{test}_stg_{stage}_{type_2}_response.txt')

    typ1_response = f.file_to_string(file_path=typ1_response_path)
    typ2_response = f.file_to_string(file_path=typ2_response_path)

    sys_typ1 = stg_2_typ1_sys + '\n' + typ1_response   # Final system prompts
    sys_typ2 = stg_2_typ2_sys + '\n' + typ2_response

    # Summary data (User prompts)
    version = f.get_summary_version()
    df_typ1 = pd.read_csv(f'test_data/RAsum_{treatment}_{type_1}_v{version}.csv')
    df_typ2 = pd.read_csv(f'test_data/RAsum_{treatment}_{type_2}_v{version}.csv')

    df_typ1['window_number'] = df_typ1['window_number'].astype(int)   # Making sure window number is an integer
    df_typ2['window_number'] = df_typ2['window_number'].astype(int)

    df_typ1 = df_typ1[:max_windows] if max_windows != None else df_typ1  # Adjusting to max windows for Stage 2
    df_typ2 = df_typ2[:max_windows] if max_windows != None else df_typ2

    # Aggregating prompts
    window_prompts = [[type_1, sys_typ1, df_typ1], [type_2, sys_typ2, df_typ2]]

    for i in window_prompts:
        # Requests for instances
        inst_dir = os.path.join(test_dir, f'stage_2_{i[0]}') # Creating ind. instance directory
        os.makedirs(inst_dir, exist_ok=False)

        sys_prmpt = i[1]    # System prompt for ucoop or udef data
        sys_prmpt_path = os.path.join(inst_dir, f't{test[5:]}_stg_2_{i[0]}_sys_prmpt.txt') if test_type == 'test' else os.path.join(inst_dir, f'{test}_stg_2_{i[0]}_sys_prmpt.txt')
        f.write_file(file_path=sys_prmpt_path, file_write=sys_prmpt)

        # Prompt & Response paths
        prompt_path = os.path.join(inst_dir, 'prompts')
        response_path = os.path.join(inst_dir, 'responses')
        os.makedirs(prompt_path, exist_ok=True)
        os.makedirs(response_path, exist_ok=True)

        # Requesting chat completion for each row
        df = i[2]   # Test data for ucoop or udef data
        for k in range(len(df)):
            row = df.iloc[k].to_dict()  # Creating a dictionary for each indv. row
            
            # GPT request output
            model.set_max_tokens(600)
            output = model.GPT_response(sys_prmpt, str(row))
            
            # Creating paths for prompts & GPT responses using window_numbers
            window_number = row['window_number']
            
            if test_type == 'test':
                user_prmpt_path = os.path.join(prompt_path, f't{test[5:]}_{window_number}_user_prmpt.txt')
                output_path = os.path.join(response_path, f't{test[5:]}_{window_number}_response.txt')
            elif test_type == 'subtest':
                user_prmpt_path = os.path.join(prompt_path, f'{test}_{window_number}_user_prmpt.txt')
                output_path = os.path.join(response_path, f'{test}_{window_number}_response.txt')
            
            # Writing .txt files for prompts & GPT response
            f.write_file(user_prmpt_path, str(row))
            f.write_file(output_path, str(output))

        # Prelimaries to final output
        if i[0] == type_1:
            if summary_type == 'FAR':
                df['cooperation'] = 1
                typ1_df = f.response_df(response_dir=response_path, test_df=df)  # Coding GPT classifications for ucoop instances
                typ1_df = f.coop_def_rename(typ1_df, type_1) # Adding ucoop prefix to categories
            else:
                df['unilateral_cooperation'] = 1
                typ1_df = f.response_df(response_dir=response_path, test_df=df)  # Coding GPT classifications for ucoop instances
                typ1_df = f.ucoop_udef_rename(typ1_df, type_1) # Adding ucoop prefix to categories
        else:
            if summary_type == 'FAR':
                df['cooperation'] = 0
                typ2_df = f.response_df(response_dir=response_path, test_df=df)   # Coding GPT classifications for udef instances
                typ2_df = f.coop_def_rename(typ2_df, type_2)    # Adding udef prefix to categories
            else:
                df['unilateral_cooperation'] = 0
                typ2_df = f.response_df(response_dir=response_path, test_df=df)   # Coding GPT classifications for udef instances
                typ2_df = f.ucoop_udef_rename(typ2_df, type_2)    # Adding udef prefix to categories

    # Final output dataframe
    GPT_df = pd.concat([typ1_df, typ2_df], ignore_index=True, sort=False)
    GPT_df = GPT_df.fillna(0)
    og_df = pd.read_csv(f'raw_data/RAsum_{treatment}_v{version}.csv')
    final_df = f.final_merge_df_FAR(GPT_df, og_df) if summary_type == 'FAR' else f.final_merge_df(GPT_df, og_df)
    final_out_path = os.path.join(test_dir, f"t{test[5:]}_final_output.csv" if test_type == 'test' else f"{test}_final_output.csv")
    final_df.to_csv(final_out_path, index=False)
    return print("Stage 2 Complete")


def stage_2_FAR_output(treatment, max_windows=None, test_type='test', refinement=True):
    '''
    Stage 2 function for FAR coding
    '''
    # Getting test directory
    if test_type == 'test':
        test = f.get_test_name(previous=True)
        test_dir = os.path.join('output/', test)
    elif test_type == 'subtest':
        test = f.get_test_name(test_type='subtest', previous=True)
        test_dir = os.path.join('output/_subtests/', test)

    # System Prompts
    stg_2_sys = f.create_system_prompt(approach='approach_1', treatment=treatment, stage='stage_2', window_type='FAR') # Getting system prompts for stage 2

    ## Getting Stage 1 response to merge with Stage 2 system prompt
    stage = '1' if refinement == False else '1r'
    stg_1_dir = os.path.join(test_dir, f'stage_{stage}/')
    if test_type == 'test':
        response_path = os.path.join(stg_1_dir, f't{test[5:]}_stg_{stage}_response.txt')
    elif test_type == 'subtest':
        response_path = os.path.join(stg_1_dir, f'{test}_stg_{stage}_response.txt')

    response = f.file_to_string(file_path=response_path)

    sys = stg_2_sys + '\n' + response   # Final system prompts

    # Summary data (User prompts)
    version = f.get_summary_version()
    df = pd.read_csv(f'test_data/RAsum_{treatment}_v{version}.csv')

    df['window_number'] = df['window_number'].astype(int)   # Making sure window number is an integer

    df = df[:max_windows] if max_windows != None else df  # Adjusting to max windows for Stage 2
    
    # Requests for both ucoop and udef instances
    inst_dir = os.path.join(test_dir, 'stage_2') # Creating directory
    os.makedirs(inst_dir, exist_ok=False)

    sys_prmpt_path = os.path.join(inst_dir, f't{test[5:]}_stg_2_sys_prmpt.txt') if test_type == 'test' else os.path.join(inst_dir, f'{test}_stg_2_sys_prmpt.txt')
    f.write_file(file_path=sys_prmpt_path, file_write=sys)

    # Prompt & Response paths
    prompt_path = os.path.join(inst_dir, 'prompts')
    response_path = os.path.join(inst_dir, 'responses')
    os.makedirs(prompt_path, exist_ok=True)
    os.makedirs(response_path, exist_ok=True)

    # Requesting chat completion for each row
    for k in range(len(df)):
        row = df.iloc[k].to_dict()  # Creating a dictionary for each indv. row
        
        # GPT request output
        model.set_max_tokens(600)
        output = model.GPT_response(sys, str(row))
        
        # Creating paths for prompts & GPT responses using window_numbers
        window_number = row['window_number']
        
        if test_type == 'test':
            user_prmpt_path = os.path.join(prompt_path, f't{test[5:]}_{window_number}_user_prmpt.txt')
            output_path = os.path.join(response_path, f't{test[5:]}_{window_number}_response.txt')
        elif test_type == 'subtest':
            user_prmpt_path = os.path.join(prompt_path, f'{test}_{window_number}_user_prmpt.txt')
            output_path = os.path.join(response_path, f'{test}_{window_number}_response.txt')
        
        # Writing .txt files for prompts & GPT response
        f.write_file(user_prmpt_path, str(row))
        f.write_file(output_path, str(output))
        
        GPT_df = f.response_df(response_dir=response_path, test_df=df)  # Coding GPT classifications for ucoop instances

    # Final output dataframe
    GPT_df = GPT_df.fillna(0)
    GPT_df = GPT_df.drop(['summary'], axis=1)
    og_df = pd.read_csv(f'raw_data/RAsum_{treatment}_v{version}.csv')
    final_df = pd.merge(og_df, GPT_df, on='window_number')
    final_out_path = os.path.join(test_dir, f"t{test[5:]}_final_output.csv" if test_type == 'test' else f"{test}_final_output.csv")
    final_df.to_csv(final_out_path, index=False)
    return print("Stage 2 Complete")


def run_full_test(treatment, summary_type, test_type, max_windows, refinement):
    stage_1_output(treatment=treatment, summary_type=summary_type, test_type=test_type)
    if refinement == True:
        stage_1r_output(treatment=treatment, summary_type=summary_type, test_type=test_type)
    stage_2_output(treatment=treatment, summary_type=summary_type, test_type=test_type, max_windows=max_windows, refinement=refinement)
    return print('Full Test Complete')
    

# stage_1_output(treatment=, summary_type=, test_type=)
# stage_1r_output(treatment=, summary_type=, test_type=)
# stage_2_output(treatment=, summary_type=, test_type=, max_windows=, refinement=)
# run_full_test(treatment=, summary_type=, test_type=, max_windows=, refinement=)