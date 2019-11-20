#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:57:31 2019

@author: daniel
"""
import re, collections
def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
    
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token]

    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token.replace('.', '[.]'))

        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]

        substring_start_position = 0
        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position:substring_end_position]
            string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
        break
    return string_tokens
def load_model():
    import pickle
    F = open('multi_bpe_model.pkl', 'rb')
    E = pickle.load(F)
    F.close()
    return E
    
def multi_bpe(str, E):    
    sorted_tokens= [token for (token, freq) in E]
    word_given = str
    result = tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='</u>')
    return result