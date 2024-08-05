'''
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

def stage_1_output(treatment, test_type='test', max_windows=None, iterations=1):
    '''
    Stage 1 function
    '''
    # Making test directory
    if test_type == 'test':
        test = f.get_test_dir()
        test_dir = os.path.join('output/', test)
        info_path = os.path.join(test_dir, f't{test[5:]}_test_info.txt')  # Test info path
    elif test_type == 'subtest':
        test = f.get_test_dir(test_type='subtest')
        test_dir = os.path.join('output/_subtests/', test)
        info_path = os.path.join(test_dir, f'{test}__subtest_info.txt')  # Test info path

    os.makedirs(test_dir, exist_ok=False)

    # Test info
    version = f.get_summary_version()
    info = model.test_info(test=f"Test {test[5:]}" if test_type == 'test' else f"Subtest {test}", data_name=f'RAsum_{treatment}_ucoop_{version}.csv & RAsum_{treatment}_udef_{version}.csv')
    f.write_file(file_path=info_path, file_write=info)

    for i in ['ucoop', 'udef']:
        iteration_count = 0

        inst_dir = os.path.join(test_dir, f'stage_1_{i}') # Creating ind. instance directory
        os.makedirs(inst_dir, exist_ok=False)

        # Iterating process
        while iteration_count < iterations:
            ## Step 1: Generate category using set of summaries ##
            # System prompt
            sys_1 = f.file_to_string(file_path=f'prompts/approach_2/stage_1/sys_1.md')

            # Summary data (User prompts)
            if iteration_count < 1:
                df = pd.read_csv(f'test_data/RAsum_{treatment}_{i}_v{version}.csv')
                df = df[:max_windows]
            else:
                previous_cat = f.get_cat_number(stage_dir=inst_dir, previous=True)
                previous_cat_dir = os.path.join(inst_dir, previous_cat)
                previous_cat_out = os.path.join(previous_cat_dir, f't{test[5:]}_stg_1_{previous_cat}_{i}_output.csv' if test_type == 'test' else f'{test}_stg_1_{previous_cat}_{i}_output.csv')
                df = pd.read_csv(previous_cat_out)
                df = df.loc[df['belongs'] == 0]
                df = df.drop(['belongs', 'response'], axis=1)
            
            cat_number = f.get_cat_number(stage_dir=inst_dir, previous=False)
            cat_dir = os.path.join(inst_dir, cat_number)
            os.makedirs(cat_dir, exist_ok=False)
            
            df['window_number'] = df['window_number'].astype(int)   # Making sure window number is an integer
            user_1 = str(df.to_dict('records'))
            
            # GPT request output
            model.set_max_tokens(1300)
            output_1 = model.GPT_response(sys=sys_1, user=user_1)
            
            ## Step 2: Determine immediate classifcations
            # System prompt
            sys_prmpt = f.file_to_string(file_path=f'prompts/approach_2/stage_1/sys_2_{treatment}_{i}.md')
            sys_2 = sys_prmpt + '\n' + str(output_1)
            
            # GPT request output
            output_df = pd.DataFrame(df)
            output_df['belongs'] = 0
            for k in range(len(df)):
                row = df.iloc[k].to_dict()  # Creating a dictionary for each indv. row
                
                # GPT request output
                model.set_max_tokens(10)
                output_2 = model.GPT_response(sys_2, str(row))
                
                output_df.loc[k, 'response'] = output_2
                output_df.loc[k, 'belongs'] = 1 if "yes" in output_2.lower() else 0
            
            
            # Creating paths for prompts & GPT response
            if test_type == 'test':
                sys_1_prmpt_path = os.path.join(cat_dir, f't{test[5:]}_stg_1_{cat_number}_{i}_sys1_prmpt.txt')
                sys_2_prmpt_path = os.path.join(cat_dir, f't{test[5:]}_stg_1_{cat_number}_{i}_sys2_prmpt.txt')
                user_prmpt_path = os.path.join(cat_dir, f't{test[5:]}_stg_1_{cat_number}_{i}_user_prmpt.txt')
                response_path = os.path.join(cat_dir, f't{test[5:]}_stg_1_{cat_number}_{i}_response.txt')
                output_path = os.path.join(cat_dir, f't{test[5:]}_stg_1_{cat_number}_{i}_output.csv')
            elif test_type == 'subtest':
                sys_1_prmpt_path = os.path.join(cat_dir, f'{test}_stg_1_{cat_number}_{i}_sys1_prmpt.txt')
                sys_2_prmpt_path = os.path.join(cat_dir, f'{test}_stg_1_{cat_number}_{i}_sys2_prmpt.txt')
                user_prmpt_path = os.path.join(cat_dir, f'{test}_stg_1_{cat_number}_{i}_user_prmpt.txt')
                response_path = os.path.join(cat_dir, f'{test}_stg_1_{cat_number}_{i}_response.txt')
                output_path = os.path.join(cat_dir, f'{test}_stg_1_{cat_number}_{i}_output.csv')
            
            # Writing .txt files for prompts & GPT response
            f.write_file(file_path=sys_1_prmpt_path, file_write=sys_1)
            f.write_file(file_path=sys_2_prmpt_path, file_write=sys_2)
            f.write_file(file_path=user_prmpt_path, file_write=user_1)
            f.write_file(file_path=response_path, file_write=str(output_1))
            output_df.to_csv(output_path, index=False)
            iteration_count += 1
    return print("Stage 1 Complete")


def stage_2_output(treatment, max_windows=None, test_type='test'):
    '''
    Stage 2 function
    '''
    # Getting test directory
    if test_type == 'test':
        test = f.get_test_dir(previous=True)
        test_dir = os.path.join('output/', test)
    elif test_type == 'subtest':
        test = f.get_test_dir(test_type='subtest', previous=True)
        test_dir = os.path.join('output/_subtests/', test)

    # System Prompts
    stg_2_ucoop_sys = f.file_to_string(file_path=f'prompts/ucoop/{treatment}/sys_2_{treatment}_ucoop.md') # Getting system prompts for stage 2
    stg_2_udef_sys = f.file_to_string(file_path=f'prompts/udef/{treatment}/sys_2_{treatment}_udef.md')

    ## Getting Stage 1 response to merge with Stage 2 system prompt
    stg_1_ucoop_dir = os.path.join(test_dir, 'stage_1_ucoop/')
    stg_1_udef_dir = os.path.join(test_dir, 'stage_1_udef/')
    if test_type == 'test':
        ucoop_response_path = os.path.join(stg_1_ucoop_dir, f't{test[5:]}_stg_1_ucoop_response.txt')
        udef_response_path = os.path.join(stg_1_udef_dir, f't{test[5:]}_stg_1_udef_response.txt')
    elif test_type == 'subtest':
        ucoop_response_path = os.path.join(stg_1_ucoop_dir, f'{test}_stg_1_ucoop_response.txt')
        udef_response_path = os.path.join(stg_1_udef_dir, f'{test}_stg_1_udef_response.txt')

    ucoop_response = f.file_to_string(file_path=ucoop_response_path)
    udef_response = f.file_to_string(file_path=udef_response_path)

    sys_ucoop = stg_2_ucoop_sys + '\n' + ucoop_response   # Final system prompts
    sys_udef = stg_2_udef_sys + '\n' + udef_response

    # Summary data (User prompts)
    version = f.get_summary_version()
    df_ucoop = pd.read_csv(f'test_data/RAsum_{treatment}_ucoop_{version}.csv')
    df_udef = pd.read_csv(f'test_data/RAsum_{treatment}_udef_{version}.csv')

    df_ucoop['window_number'] = df_ucoop['window_number'].astype(int)   # Making sure window number is an integer
    df_udef['window_number'] = df_udef['window_number'].astype(int)

    df_ucoop = df_ucoop[:max_windows] if max_windows != None else df_ucoop  # Adjusting to max windows for Stage 2
    df_udef = df_udef[:max_windows] if max_windows != None else df_udef

    # Aggregating prompts
    window_prompts = [['ucoop', sys_ucoop, df_ucoop], ['udef', sys_udef, df_udef]]

    for i in window_prompts:
        # Requests for both ucoop and udef instances
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
        if i[0] == 'ucoop':
            df['unilateral_cooperation'] = 1
            ucoop_df = f.response_df(response_dir=response_path, test_df=df)  # Coding GPT classifications for ucoop instances
            ucoop_df = f.ucoop_udef_rename(ucoop_df, 'ucoop') # Adding ucoop prefix to categories
        else:
            df['unilateral_cooperation'] = 0
            udef_df = f.response_df(response_dir=response_path, test_df=df)   # Coding GPT classifications for udef instances
            udef_df = f.ucoop_udef_rename(udef_df, 'udef')    # Adding udef prefix to categories

        # Final output dataframe
        GPT_df = pd.concat([ucoop_df, udef_df], ignore_index=True, sort=False)
        GPT_df = GPT_df.fillna(0)
        final_out_path = os.path.join(test_dir, f"t{test[5:]}_final_output.csv" if test_type == 'test' else f"{test}_final_output.csv")
        GPT_df.to_csv(final_out_path, index=False)
    return print("Stage 2 Complete")


# stage_1_output(treatment=, test_type=, max_windows=, iterations=)
# stage_2_output(treatment=, test_type=, max_windows=)