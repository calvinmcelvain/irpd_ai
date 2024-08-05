'''
Module for GPT model and requests
'''
import os

# Modules & Packages
from openai import OpenAI
from datetime import date


class GPT:
    '''
    Class for GPT requests
    '''
    def __init__(self, api_key=None, organization=None, project=None, model='gpt-4o-2024-05-13', temperature=0, max_tokens=1300, top_p=1, frequency_penalty=0, presence_penalty=0):
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


    def GPT_response(self, sys, user):
        '''
        GPT request function
        '''
        response = self.client.chat.completions.create(
            model=self.MODEL,
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            top_p=self.TOP_P,
            frequency_penalty=self.FREQUENCY_PENALTY,
            presence_penalty=self.PRESENCE_PENALTY,
            messages=[
                {"role": "system", "content": str(sys)},
                {"role": "user", "content": str(user)}
            ]
        )
        output = response.choices[0].message.content  # GPT response var
        return output


    def test_info(self, test, data_name):
        '''
        Returns test information
        '''
        info = str(
            'ChatGPT Model Information:' + '\n' +
            'format: OpenAI API' + '\n' +
            'model: ' + str(self.MODEL) + '\n' +
            'temperature: ' + str(self.TEMPERATURE) + '\n' +
            'max tokens: ' + str(self.MAX_TOKENS) + '\n' +
            'top p: ' + str(self.TOP_P) + '\n' +
            'frequency penalty: ' + str(self.FREQUENCY_PENALTY) + '\n' +
            'presence penalty: ' + str(self.PRESENCE_PENALTY) + '\n\n' +
            'Test Information:' + '\n' +
            'test: ' + test + '\n' +
            'data: ' + data_name + '\n' +
            'date: ' + str(date.today()) + '\n'
        )
        return info
