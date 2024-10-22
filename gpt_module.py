'''
Module for GPT model and requests
'''
import os

# Modules & Packages
from openai import OpenAI
from pydantic import BaseModel


class GPT:
    '''
    Class for GPT requests
    '''
    def __init__(self, api_key=None, organization=None, project=None, model='gpt-4o-2024-08-06', temperature=0, max_tokens=1300, top_p=1, seed=None, frequency_penalty=0, presence_penalty=0):
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            organization=organization,
            project=project
        )
        # Model Variables
        self.MODEL = model
        self.TEMPERATURE = temperature
        self.MAX_TOKENS = max_tokens
        self.SEED = seed
        self.TOP_P = top_p
        self.FREQUENCY_PENALTY = frequency_penalty
        self.PRESENCE_PENALTY = presence_penalty


    # Setter methods
    def set_model(self, model):
        '''
        Sets GPT model
        '''
        self.MODEL = model


    def set_temperature(self, temperature):
        '''
        Sets temperature of GPT model
        '''
        self.TEMPERATURE = temperature


    def set_max_tokens(self, max_tokens):
        '''
        Sets max tokens for GPT model
        '''
        self.MAX_TOKENS = max_tokens


    def set_top_p(self, top_p):
        '''
        Sets top p of GPT model
        '''
        self.TOP_P = top_p


    def set_frequency_penalty(self, frequency_penalty):
        '''
        Sets frequency penalty for GPT model
        '''
        self.FREQUENCY_PENALTY = frequency_penalty


    def set_presence_penalty(self, presence_penalty):
        '''
        Sets presence penalty for GPT model
        '''
        self.PRESENCE_PENALTY = presence_penalty
        
    
    def set_seed(self, seed):
        '''
        Sets seed parameter
        '''
        self.SEED = seed


    def GPT_response(self, sys, user, output_structure):
        '''
        GPT request function
        '''
        response = self.client.beta.chat.completions.parse(
            model=self.MODEL,
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            top_p=self.TOP_P,
            seed=self.SEED,
            frequency_penalty=self.FREQUENCY_PENALTY,
            presence_penalty=self.PRESENCE_PENALTY,
            messages=[
                {"role": "system", "content": str(sys)},
                {"role": "user", "content": str(user)}
            ],
            response_format=output_structure,
        )
        gpt_response = response.choices[0].message.content  # GPT response var
        response_data = response
        return gpt_response, response_data


# Structured outputs
class Examples(BaseModel):
    window_number: int
    reasoning: str


class Category(BaseModel):
    category_name: str
    definition: str
    examples: list[Examples]


class Stage_1_Structure(BaseModel):
    categories: list[Category]
    

class Refinement(BaseModel):
    category_name: str
    keep_decision: bool
    reasoning: str


class Stage_1r_Structure(BaseModel):
    final_categories: list[Refinement]


class Stage_2_Structure(BaseModel):
    window_number: str
    assigned_categories: list[str]
    reasoning: str