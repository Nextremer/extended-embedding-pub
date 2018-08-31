# coding:utf-8

"""
Written_by: Sugimoto Shun
Created_at: 2018-08-31
Abstract: データセットに登場するボキャブラリ辞書を作成するコード
"""

import os
import MeCab
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = current_dir + '/datasets/training_data.xlsx' # エクセル形式のデータ
test_path  = current_dir + '/datasets/test_data.xlsx'


def make_dictionary(train_df, test_df, df):
    m = MeCab.Tagger('-Owakati')

    sentences = list(train_df[u'表層'])
    sentences.extend(list(test_df[u'表層']))
    sentences = np.array(sentences)

    vocab_dic = []
    for sentence in sentences:
        words = m.parse(sentence).split(' ')
        for word in words:
            for character in word[:2]:  # 各単語は3文字まで見る
                radicals = df[df['CHARACTER'] == character]['RADICAL'].values
                if radicals:
                    radicals = radicals[0].split(' ')[:2]  # 部首も3文字まで
                    for radical in radicals:
                        if radical not in vocab_dic:
                            vocab_dic.append(radical)

                else:  # 漢字以外の時
                    if character not in vocab_dic:
                        vocab_dic.append(character)
    vocab_df = pd.DataFrame(vocab_dic, columns=['RADICAL'], )
    return vocab_df


def main():
    train_df = pd.read_excel(train_path, skiprows=0, header=0)
    test_df = pd.read_excel(test_path, skiprows=0, header=0)
    radical_dic = pd.read_csv(current_dir + '/datasets/radical_dic.csv',
                              index_col=0)

    vocab_df = make_dictionary(train_df, test_df, radical_dic)
    vocab_df.to_csv(current_dir + '/datasets/vocab_dictionary.csv')


if __name__ == '__main__':
    main()
