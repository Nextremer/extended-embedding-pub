# coding: utf-8

"""
Written_by: Sugimoto Shun
Created_at: 2018-08-31
Abstract: 標準入力された文字列の部首を返却するプログラム
"""

import pandas as pd
import numpy as np


RADICAL_DICT_PATH = './datasets/radical_dic.csv'


def get_radical_dict(dict_path=RADICAL_DICT_PATH):
    radical_dic = pd.read_csv(dict_path, index_col=0)
    return radical_dic


def get_radical(char):
    try:
        char_ids = np.where(characters == char)[0]
        radical = radicals[char_ids[0]].split(' ')
    except IndexError:
        return char
    return radical


if __name__ == "__main__":
    radical_df = get_radical_dict()

    radicals = np.array(radical_df['RADICAL'])
    characters = np.array(radical_df['CHARACTER'])

    while True:
        sentence = input('\nplease enter character or sentence（0→exit）')
        if sentence == '0':
            break
        for char in sentence:
            print(list(get_radical(char)))
