# coding: utf-8


import pandas as pd
import numpy as np
import re


RADICAL_DICT_PATH = 'datasets/radical_dic.csv'


def get_radical_dict(dict_path=RADICAL_DICT_PATH):
    radical_dic = pd.read_csv(dict_path, index_col=0)
    return radical_dic


# 最小単位の部首まで再帰的に分解（ex. 語 → 言,吾 → 言,五,口）
def char2radicals_perfect(char, rad_list, radical_dict):
    pattern = r"&.*?;"
    radicals = np.array(radical_dict['RADICAL'])
    characters = np.array(radical_dict['CHARACTER'])
    try:
        char_ids = np.where(characters == char)[0]
        radical = radicals[char_ids[0]]
    except IndexError:
        rad_list.append(char)
        return 0
    if '&' in radical:
        match_object = re.findall(pattern, radical)
        rad_list.extend(list(map(lambda x: x[1:-1], match_object)))
        radical = re.sub(pattern, "", radical)
    if char == radical:
        rad_list.append(radical)
    else:
        for r in list(radical):
            char2radicals_perfect(r, rad_list, radical_dict)


# 部首に変換したい文字(リスト)を部首と漢字構造(ex. ⿱)に分解
def get_radicals(charcters, radical_dict):
    rad_list = []  # 漢字の部首情報を格納
    for char in charcters:
        char2radicals_perfect(char, rad_list, radical_dict)

    structures = []  # 漢字の構造情報を格納
    structure = ['⿱', '⿳', '⿰', '⿲', '⿹', '⿸', '⿺', '⿵', '⿶', '⿷', '⿴' ,'⿻']
    for r in np.array(rad_list):
        if r in structure:
            rad_list.remove(r)
            structures.append(r)
    return rad_list, structures


if __name__ == "__main__":
    radical_dict = get_radical_dict()
    
    while True:
        char = input('\nplease enter character or sentence（0→exit）')
        if char == '0':
            break
        for c in char:
            print(get_radicals(c, radical_dict)[0])
