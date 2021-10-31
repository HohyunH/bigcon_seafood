# 내장 라이브러리
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# 외장 라이브러리
from typing import collections, List, Tuple, Dict, DefaultDict, NewType
from collections import Counter
from datetime import datetime, timedelta
from timeit import default_timer as timer

# sklearn
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class data_loading():
    def __init__(self, data):
        self.data = data
    # 품목별 dataframe을 생성하고, 일자별 거래의 갯수를 리스트 형태로 저장
    def datasetting(self, p_name):

        just_p = pd.DataFrame(columns=self.data.columns.tolist())
        len_list = []
        for i, day_time in enumerate(list(Counter(self.data['REG_DATE']).keys())):
            day_df = self.data[self.data['REG_DATE']==day_time]
            just_p = pd.concat([just_p, day_df[day_df['P_NAME']==p_name]])
            len_list.append(len(day_df[day_df['P_NAME']==p_name]))

        return just_p, len_list

    # import type을 multi-hot encoding 하기 위한 방법

    def data_frame(self, trains, ctry_list):

        def data_encoding(df, type_):
            tmp = []
            for impo in df[type_]:
                types = impo.split(',')
                for i in types:
                    tmp.append(i)
            return list(set(tmp))

        p_type = "P_IMPORT_TYPE"
        import_type = data_encoding(self.data, p_type)

        con = ctry_list
        con.sort()
        col_con = [f"country_{co}" for co in con]
        col_imp = [f"import_{im}" for im in import_type]
        cols = col_con + col_imp + ['weight', 'temp', 'price']
        # print(cols)
        x_train = pd.DataFrame(columns=cols)
        tmp = []
        for i in range(len(cols)-1):
            tmp.append([])
        len_a = len(col_con)
        len_b = len(col_imp)
        for c1, c2, p in zip(trains['CTRY_1'], trains['CTRY_2'], trains['P_IMPORT_TYPE']):
            i = con.index(c1)
            j = con.index(c2)
            country = np.eye(len_a)[i] + np.eye(len_a)[j] ## 제조국 + 수출국 정보 결합

            p_import = np.zeros(len_b)
            for ty in p.split(','):
                j = import_type.index(ty)
                p_import += np.eye(len(p_import))[j] ## import type 정보 결합

            row = np.concatenate([country, p_import])

            for i, r in enumerate(row):
                tmp[i].append(r)

        for i, c in enumerate(cols[:-3]): ## 중량, 수온, 가격 이외의 것들만 고려
            x_train[c] = tmp[i]
        ## 중량과 수온 데이터 스케일링
        w = trains['WEIGHT(KG)'].tolist()
        w = np.array(w).reshape(-1,1)
        t = trains['temp'].tolist()
        t = np.array(t).reshape(-1,1)

        scaler = MinMaxScaler()
        scaler.fit(w)
        t_data = scaler.transform(w)

        wscaler = MinMaxScaler()
        wscaler.fit(t)
        w_data = wscaler.transform(t)

        x_train['weight'] =  t_data
        x_train['temp'] = w_data
        x_train['price'] = trains['P_PRICE'].tolist()

        return x_train

if __name__=="__main__":
    ## 중량 수온 데이터를 합친 dataframe, 품목 별로 불러오기
    os.chdir('/content/drive/MyDrive/BIGCONTEST/data/real_use')

    name = input('수산물 어종을 고르세요 (오징어/흰다리새우/연어)')

    if name == "오징어":
        data = pd.read_csv("./squid_temp1.csv")
        val_df = pd.read_csv("./squid_temp2.csv")
    elif name == "흰다리새우":
        data = pd.read_csv("./shrimp_temp1.csv")
        val_df = pd.read_csv("./shrimp_temp2.csv")
    elif name == "연어":
        data = pd.read_csv("./salmon_temp1.csv")
        val_df = pd.read_csv("./salmon_temp2.csv")
    else:
        None
    
    D = data_loading()
    train, t_len = D.datasetting(data, name)
    validation, v_len = D.datasetting(val_df, name)

    ## 제조국과 수출국에 대하여 multi-hot encoding을 진행하기위해서 모든 나라를 리스트형태로 저장
    ctry_1 = set(list(set(train['CTRY_1']))+list(set(train['CTRY_2'])))
    ctry_2 = set(list(set(validation['CTRY_1']))+list(set(validation['CTRY_2'])))
    con = list(ctry_1) + list(ctry_2)
    ctry_list = list(set(con))

    x_train = D.data_frame(train, ctry_list)
    x_val = D.data_frame(validation, ctry_list)

    