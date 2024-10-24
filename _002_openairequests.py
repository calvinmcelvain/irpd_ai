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
import json
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
model.set_model('gpt-4o-2024-08-06')
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
    # Getting prefixes based on summary type
    type_1, type_2 = f.get_window_types(summary_type=summary_type)
    
    # System prompts
    sys_typ1 = f.file_to_string(file_path=f'prompts/{summary_type}/stg_1_{treatment}_{type_1}.md')
    sys_typ2 = f.file_to_string(file_path=f'prompts/{summary_type}/stg_1_{treatment}_{type_2}.md')

    # Summary data (User prompts)
    df_typ1 = pd.read_csv(f'data/test/{summary_type}_{treatment}_{RA}_{type_1}.csv')
    df_typ2 = pd.read_csv(f'data/test/{summary_type}_{treatment}_{RA}_{type_2}.csv')
    data_file = f'{summary_type}_{treatment}_{RA}_{type_1}.csv & {summary_type}_{treatment}_{RA}_{type_2}.csv'

    # Making sure window number is an integer
    df_typ1['window_number'] = df_typ1['window_number'].astype(int)
    df_typ2['window_number'] = df_typ2['window_number'].astype(int)

    # Turning data into a list of dictionaries, then to string
    user_typ1 = str(df_typ1.to_dict('records'))
    user_typ2 = str(df_typ2.to_dict('records'))

    # Aggregating prompts
    window_prompts = [[type_1, sys_typ1, user_typ1], [type_2, sys_typ2, user_typ2]]

    # Making test directory
    test = f.get_test_name(summary_type=summary_type, test_type=test_type)
    test_number = test[5:] if test_type == 'test' else test
    test_dir = f.get_test_directory(summary_type=summary_type, test_type=test_type, test=test)
    
    # Making raw output directory
    raw_dir = os.path.join(test_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=False)

    # Initializing info data & response
    info_data = {type_1: 0, type_2: 0}
    responses = []
    
    # GPT requests
    for i in window_prompts:
        # Creating ind. instance directory
        inst_dir = os.path.join(raw_dir, f'stage_1_{i[0]}')
        os.makedirs(inst_dir, exist_ok=False)

        # Prompts
        sys_prmpt = i[1]
        user_prmpt = i[2]

        # GPT request output
        model.set_max_tokens(2000)
        response, response_data = model.GPT_response(sys=sys_prmpt, user=user_prmpt, output_structure=gpt_module.Stage_1_Structure)
        info_data[i[0]] = response_data
        responses.append(response)
        
        # Creating paths for prompts & GPT response
        sys_prmpt_path = os.path.join(inst_dir, f't{test_number}_stg_1_{i[0]}_sys_prmpt.txt')
        user_prmpt_path = os.path.join(inst_dir, f't{test_number}_stg_1_{i[0]}_user_prmpt.txt')
        response_path = os.path.join(inst_dir, f't{test_number}_stg_1_{i[0]}_response.txt')

        # Writing .txt files for prompts & GPT response
        f.write_file(file_path=sys_prmpt_path, file_write=sys_prmpt)
        f.write_file(file_path=user_prmpt_path, file_write=user_prmpt)
        f.write_file(file_path=response_path, file_write=str(response))
    
    # Writing test information
    f.write_test_info(test_info=info_data, directory=raw_dir, test_number=test_number, model_info=model, stage_number = '1', data_file=data_file)
    
    # Making formatted, easy to read, Stage 1 output file
    cat_types = [type_1, type_2]
    formatted_response_path = os.path.join(test_dir, f't{test_number}_stg1_categories.pdf')
    f.stage_1_response_format(responses=responses, cat_types=cat_types, file_path=formatted_response_path)
    
    return print("Stage 1 Complete")


def stage_1r_output(treatment: str, summary_type: str, test_type='test'):
    '''
    Stage 1 refinement function
    '''
    # Getting prefixes based on summary type
    type_1, type_2 = f.get_window_types(summary_type=summary_type)
    
    # Getting test directory
    test = f.get_test_name(summary_type=summary_type, test_type=test_type, previous=True)
    test_number = test[5:] if test_type == 'test' else test
    test_dir = f.get_test_directory(summary_type=summary_type, test_type=test_type, test=test)
    
    # Getting raw output directory
    raw_dir = os.path.join(test_dir, 'raw')
    
    # System prompts
    sys_typ1 = f.file_to_string(file_path=f'prompts/{summary_type}/stg_1r_{treatment}_{type_1}.md')
    sys_typ2 = f.file_to_string(file_path=f'prompts/{summary_type}/stg_1r_{treatment}_{type_2}.md')
    
    # User prompts
    stg_1_typ1_dir = os.path.join(raw_dir, f'stage_1_{type_1}/')
    stg_1_typ2_dir = os.path.join(raw_dir, f'stage_1_{type_2}/')
    typ1_response_path = os.path.join(stg_1_typ1_dir, f't{test_number}_stg_1_{type_1}_response.txt')
    typ2_response_path = os.path.join(stg_1_typ2_dir, f't{test_number}_stg_1_{type_2}_response.txt')
    raw_typ1 = f.file_to_string(file_path=typ1_response_path)
    raw_typ2 = f.file_to_string(file_path=typ2_response_path)
    user_typ1 = f.stage_1_response_format(responses=[raw_typ1], cat_types=[type_1], file_path='', stage_1r=True)
    user_typ2 = f.stage_1_response_format(responses=[raw_typ2], cat_types=[type_2], file_path='', stage_1r=True)
    
    # Aggregating prompts
    window_prompts = [[type_1, sys_typ1, user_typ1], [type_2, sys_typ2, user_typ2]]
    
    # Initializing info data & response
    info_data = {type_1: 0, type_2: 0}
    responses = []
    
    # GPT requests
    for i in window_prompts:  # Requests for instances
        inst_dir = os.path.join(raw_dir, f'stage_1r_{i[0]}') # Creating ind. instance directory
        os.makedirs(inst_dir, exist_ok=False)

        # Prompts
        sys_prmpt = i[1]
        user_prmpt = i[2]

        # GPT request output
        model.set_max_tokens(2000)
        response, response_data = model.GPT_response(sys=sys_prmpt, user=user_prmpt, output_structure=gpt_module.Stage_1r_Structure)
        info_data[i[0]] = response_data
        responses.append(response)

        # Creating paths for prompts & GPT response
        sys_prmpt_path = os.path.join(inst_dir, f't{test_number}_stg_1r_{i[0]}_sys_prmpt.txt')
        user_prmpt_path = os.path.join(inst_dir, f't{test_number}_stg_1r_{i[0]}_user_prmpt.txt')
        response_path = os.path.join(inst_dir, f't{test_number}_stg_1r_{i[0]}_response.txt')

        # Writing .txt files for prompts & GPT response
        f.write_file(file_path=sys_prmpt_path, file_write=sys_prmpt)
        f.write_file(file_path=user_prmpt_path, file_write=user_prmpt)
        f.write_file(file_path=response_path, file_write=str(response))
    
    # Writing test information
    f.write_test_info(test_info=info_data, directory=raw_dir, test_number=test_number, model_info=model, stage_number = '1r')
    
    # Making formatted, easy to read, Stage 1 output file
    cat_types = [type_1, type_2]
    formatted_response_path = os.path.join(test_dir, f't{test_number}_stg1r_categories.pdf')
    f.stage_1r_response_format(stage_1_responses=[raw_typ1, raw_typ2], responses=responses, cat_types=cat_types, file_path=formatted_response_path)
    
    return print("Stage 1r Complete")


def stage_2_output(treatment: str, summary_type: str, RA: str, max_windows=None, test_type='test', refinement=True, stage_3 = True):
    '''
    Stage 2 function and Stage 3 function
    '''
    # Getting prefixes based on summary type
    type_1, type_2 = f.get_window_types(summary_type=summary_type)
    
    # Getting test directory
    test = f.get_test_name(summary_type=summary_type, test_type=test_type, previous=True)
    test_number = test[5:] if test_type == 'test' else test
    test_dir = f.get_test_directory(summary_type=summary_type, test_type=test_type, test=test)
    
    # Getting raw output directory
    raw_dir = os.path.join(test_dir, 'raw')

    # System Prompts
    stg_2_typ1_sys = f.file_to_string(file_path=f'prompts/{summary_type}/stg_2_{treatment}_{type_1}.md')
    stg_2_typ2_sys = f.file_to_string(file_path=f'prompts/{summary_type}/stg_2_{treatment}_{type_2}.md')
    stg_3_typ1_sys = f.file_to_string(file_path=f'prompts/{summary_type}/stg_3_{treatment}_{type_1}.md')
    stg_3_typ2_sys = f.file_to_string(file_path=f'prompts/{summary_type}/stg_3_{treatment}_{type_2}.md')
    
    # Getting Stage 1 response
    stg_1_typ1_dir = os.path.join(raw_dir, f'stage_1_{type_1}/')
    stg_1_typ2_dir = os.path.join(raw_dir, f'stage_1_{type_2}/')
    typ1_stg_1_response_path = os.path.join(stg_1_typ1_dir, f't{test_number}_stg_1_{type_1}_response.txt')
    typ2_stg_1_response_path = os.path.join(stg_1_typ2_dir, f't{test_number}_stg_1_{type_2}_response.txt')
    raw_stg_1_typ1 = f.file_to_string(file_path=typ1_stg_1_response_path)
    raw_stg_1_typ2 = f.file_to_string(file_path=typ2_stg_1_response_path)
    
    if refinement == True:
        # Getting Stage 1r responses
        stg_1r_typ1_dir = os.path.join(raw_dir, f'stage_1r_{type_1}/')
        stg_1r_typ2_dir = os.path.join(raw_dir, f'stage_1r_{type_2}/')
        typ1_stg_1r_response_path = os.path.join(stg_1r_typ1_dir, f't{test_number}_stg_1r_{type_1}_response.txt')
        typ2_stg_1r_response_path = os.path.join(stg_1r_typ2_dir, f't{test_number}_stg_1r_{type_2}_response.txt')
        raw_stg_1r_typ1 = f.file_to_string(file_path=typ1_stg_1r_response_path)
        raw_stg_1r_typ2 = f.file_to_string(file_path=typ2_stg_1r_response_path)
        
        typ1_response = f.stage_1r_response_format(stage_1_responses=[raw_stg_1_typ1], responses=[raw_stg_1r_typ1], file_path='', stage_2=True, cat_types=[type_1])
        typ2_response = f.stage_1r_response_format(stage_1_responses=[raw_stg_1_typ2], responses=[raw_stg_1r_typ2], file_path='', stage_2=True, cat_types=[type_2])
    else:
        typ1_response = f.stage_1_response_format(responses=[raw_stg_1_typ1], cat_types=[type_1], file_path='', stage_1r=True)
        typ2_response = f.stage_1_response_format(responses=[raw_stg_1_typ2], cat_types=[type_2], file_path='', stage_1r=True)

    # Final system prompts
    sys_typ1 = stg_2_typ1_sys + '\n' + typ1_response
    sys_typ2 = stg_2_typ2_sys + '\n' + typ2_response
    sys_typ1_stg3 = stg_3_typ1_sys + '\n' + typ1_response
    sys_typ2_stg3 = stg_3_typ2_sys + '\n' + typ2_response

    # Summary data (User prompts)
    df_typ1 = pd.read_csv(f'data/test/{summary_type}_{treatment}_{RA}_{type_1}.csv')
    df_typ2 = pd.read_csv(f'data/test/{summary_type}_{treatment}_{RA}_{type_2}.csv')
    
    # For test info
    data_file = f'{summary_type}_{treatment}_{RA}_{type_1}.csv & {summary_type}_{treatment}_{RA}_{type_2}.csv'

    # Making sure window number is an integer
    df_typ1['window_number'] = df_typ1['window_number'].astype(int)
    df_typ2['window_number'] = df_typ2['window_number'].astype(int)

    # Adjusting to max windows for Stage 2
    df_typ1 = df_typ1[:max_windows] if max_windows != None else df_typ1
    df_typ2 = df_typ2[:max_windows] if max_windows != None else df_typ2

    # Aggregating prompts
    window_prompts = [[type_1, sys_typ1, df_typ1, sys_typ1_stg3], [type_2, sys_typ2, df_typ2, sys_typ2_stg3]]
    
    # Initializing info data
    info_data = {type_1: 0, type_2: 0}
    info_data_stg3 = {type_1: 0, type_2: 0}
    for i in window_prompts:
        # Creating ind. instance directory
        inst_dir = os.path.join(raw_dir, f'stage_2_{i[0]}')
        os.makedirs(inst_dir, exist_ok=False)

        # System prompt for ucoop or udef data
        sys_prmpt = i[1]
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

        # Test data for ucoop or udef data
        df = i[2]
        
        # Stage 3 info and system prompt write
        if stage_3 == True:
            # Making directory
            inst_dir_stg3 = os.path.join(raw_dir, f'stage_3_{i[0]}')
            os.makedirs(inst_dir_stg3, exist_ok=False)
            
            # Writing system prompt
            sys_prmpt_stg3 = i[3]
            sys_prmpt_path_stg3 = os.path.join(inst_dir_stg3, f't{test_number}_stg_3_{i[0]}_sys_prmpt.txt') 
            f.write_file(file_path=sys_prmpt_path_stg3, file_write=sys_prmpt_stg3)
            
            # Prompt & Response paths
            prompt_path_stg3 = os.path.join(inst_dir_stg3, 'prompts')
            response_path_stg3 = os.path.join(inst_dir_stg3, 'responses')
            os.makedirs(prompt_path_stg3, exist_ok=True)
            os.makedirs(response_path_stg3, exist_ok=True)
            
            # Initializing tokens to later average
            completion_tok_stg3 = []
            prompt_tok_stg3 = []
            total_tok_stg3 = []
        
        # Requesting chat completion for each row
        for k in range(len(df)):
            # Creating a dictionary for each indv. row
            row = df.iloc[k].to_dict()
            
            # GPT request output
            model.set_max_tokens(600)
            response, response_data = model.GPT_response(sys=sys_prmpt, user=str(row), output_structure=gpt_module.Stage_2_Structure)
            
            # Appending response data
            if k == 1:
                info_data[i[0]] = response_data
            
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
            
            # For stage 3 prompts
            if stage_3 == True:
                user_prompt = f.stage_3_user_prompts(stage_2_response=response, summary=row)
                
                # GPT request output
                model.set_max_tokens(600)
                response_stg3, response_data_stg3 = model.GPT_response(sys=sys_prmpt_stg3, user=user_prompt, output_structure=gpt_module.Stage_3_Structure)
                
                # Appending response data
                if k == 1:
                    info_data_stg3[i[0]] = response_data_stg3
                
                # Appending all values of token usage
                completion_tok_stg3.append(response_data_stg3.usage.completion_tokens)
                prompt_tok_stg3.append(response_data_stg3.usage.prompt_tokens)
                total_tok_stg3.append(response_data_stg3.usage.total_tokens)
                
                # Creating paths for prompts & GPT responses using window_numbers
                user_prmpt_path_stg3 = os.path.join(prompt_path_stg3, f't{test_number}_{window_number}_user_prmpt.txt')
                output_path_stg3 = os.path.join(response_path_stg3, f't{test_number}_{window_number}_response.txt')
                
                # Writing .txt files for prompts & GPT response
                f.write_file(user_prmpt_path_stg3, str(user_prompt))
                f.write_file(output_path_stg3, str(response_stg3))

        # Gettting average (mean) of usage tokens and overwriting current value
        mcompletion_tok = sum(completion_tok)
        mprompt_tok = sum(prompt_tok)
        mtotal_tok = sum(total_tok)
        info_data[i[0]].usage.completion_tokens = mcompletion_tok
        info_data[i[0]].usage.prompt_tokens = mprompt_tok
        info_data[i[0]].usage.total_tokens = mtotal_tok
        if stage_3 == True:
            mcompletion_tok_stg3 = sum(completion_tok_stg3)
            mprompt_tok_stg3 = sum(prompt_tok_stg3)
            mtotal_tok_stg3 = sum(total_tok_stg3)
            info_data_stg3[i[0]].usage.completion_tokens = mcompletion_tok_stg3
            info_data_stg3[i[0]].usage.prompt_tokens = mprompt_tok_stg3
            info_data_stg3[i[0]].usage.total_tokens = mtotal_tok_stg3
        
        # Prelimaries to final output
        if i[0] == type_1:
            if summary_type == 'first' or summary_type == 'switch':
                df['cooperation'] = 1
            else:
                df['unilateral_cooperation'] = 1
            
            # Adding ucoop prefix to categories
            typ1_df = f.response_df(response_dir=response_path, test_df=df, stage_3=False)
            typ1_df = f.category_prefix(df=typ1_df, summary_type=summary_type, prefix=type_1)
            if stage_3 == True:
                typ1_df_stg3 = f.response_df(response_dir=response_path_stg3, test_df=df, stage_3=True)
                typ1_df_stg3 = f.category_prefix(df=typ1_df_stg3, summary_type=summary_type, prefix=type_1)
        else:
            if summary_type == 'first' or summary_type == 'switch':
                df['cooperation'] = 0
            else:
                df['unilateral_cooperation'] = 0
            
            typ2_df = f.response_df(response_dir=response_path, test_df=df, stage_3=False)
            typ2_df = f.category_prefix(df=typ2_df, summary_type=summary_type, prefix=type_2)
            if stage_3 == True:
                typ2_df_stg3 = f.response_df(response_dir=response_path_stg3, test_df=df, stage_3=True)
                typ2_df_stg3 = f.category_prefix(df=typ2_df_stg3, summary_type=summary_type, prefix=type_1)

    # Final output dataframe
    GPT_df = pd.concat([typ1_df, typ2_df], ignore_index=True, sort=False)
    GPT_df = GPT_df.fillna(0)
    og_df = pd.read_csv(f'data/raw/{summary_type}_{treatment}_{RA}.csv')
    final_df = f.final_merge_df(final_df=GPT_df, og_df=og_df, summary_type=summary_type)
    final_out_path = os.path.join(test_dir, f"t{test_number}_stg2_final_output.csv")
    final_df.to_csv(final_out_path, index=False)
    if stage_3 == True:
        # Final output dataframe
        GPT_df_stg3 = pd.concat([typ1_df_stg3, typ2_df_stg3], ignore_index=True, sort=False)
        GPT_df_stg3 = GPT_df_stg3.fillna(0)
        final_df_stg3 = f.final_merge_df(final_df=GPT_df_stg3, og_df=og_df, summary_type=summary_type)
        final_out_path_stg3 = os.path.join(test_dir, f"t{test_number}_stg3_final_output.csv")
        final_df_stg3.to_csv(final_out_path_stg3, index=False)
    
    # Writing test information
    f.write_test_info(test_info=info_data, directory=raw_dir, test_number=test_number, model_info=model, stage_number='2', data_file=data_file)
    if stage_3 == True:
        f.write_test_info(test_info=info_data_stg3, directory=raw_dir, test_number=test_number, model_info=model, stage_number='3', data_file="")
    
    return print("Stage 2 Complete") if stage_3 == False else print("Stage 2 and 3 Complete")


def run_full_test(treatment: str, summary_type: str, RA: str, test_type: str, max_windows: any, refinement: bool, stage_3: bool):
    stage_1_output(treatment=treatment, summary_type=summary_type, RA=RA, test_type=test_type)
    if refinement == True:
        stage_1r_output(treatment=treatment, summary_type=summary_type, test_type=test_type)
    stage_2_output(treatment=treatment, summary_type=summary_type, RA=RA, test_type=test_type, max_windows=max_windows, refinement=refinement, stage_3=stage_3)
    return print('Full Test Complete')


def run_full_set(summary_type: str, RA: str, test_type: str, max_windows: any, refinement: bool, stage_3: bool):
    run_full_test(treatment='noise', summary_type=summary_type, RA=RA, test_type=test_type, max_windows=max_windows, refinement=refinement, stage_3=stage_3)
    run_full_test(treatment='no_noise', summary_type=summary_type, RA=RA, test_type=test_type, max_windows=max_windows, refinement=refinement, stage_3=stage_3)
    run_full_test(treatment='merged', summary_type=summary_type, RA=RA, test_type=test_type, max_windows=max_windows, refinement=refinement, stage_3=stage_3)
    return print("Full Test Set Complete")
    

# stage_1_output(treatment=, summary_type=, RA=, test_type=)
# stage_1r_output(treatment=, summary_type=, test_type=)
# stage_2_output(treatment=, summary_type=, RA=, test_type=, max_windows=, refinement=, stage_3=)
# run_full_test(treatment=, summary_type=, RA=, test_type=, max_windows=, refinement=, stage_3=)
# run_full_set(summary_type=, RA=, test_type=, max_windows=, refinement=, stage_3=)