# Attention 모델
import keras
from tensorflow_addons.layers import MultiHeadAttention

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

from dataloading import data_loading


## Attention을 계산하는 모듈 생성

class AttentionBlock(keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0.2, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(1)
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1) 

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)
        
        return x

## Attention을 여러번 계산하고 마치막 출력으로 Fully_connected layer를 통해 차원 축소

class ModelTrunk(keras.Model):
    def __init__(self, name='ModelTrunk', num_heads=2, head_size=128, ff_dim=None, num_layers=10, dropout=0, **kwargs):
      super().__init__(name=name, **kwargs)
      if ff_dim is None:
          ff_dim = head_size
      self.dropout = dropout
      self.attention_layers = [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]
      self.dense2 = keras.layers.Dense(1, kernel_initializer='he_normal')

    def call(self, inputs):
      x = inputs
      for attention_layer in self.attention_layers:
          x = attention_layer(x)
      x = self.dense2(x)
      return x

# 소프트맥스 함수 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Attention - FC layer 를 통한 가중치 리스트와 가격 리스트 생성
def make_weight(x, y,length):
    # 날짜별로 attention score 담을 list
    weight_score = []
    # 날짜별로 p_price 담을 list
    p_price = []
    final_weight = []

    before = 0
    for leng in length:
      if leng==0:
        weight_score.append([[0]])
        p_price.append([[0]])
      else:
        a = np.array(x[before:before+leng])
        b = np.array(y[before:before+leng]).tolist()
        arr = attention_layers(a)
        weight_score.append(np.array(softmax(arr)).tolist()) ## attention score를 softmax 함수를 이용해서 정규화
        p_price.append(b)
        before += leng

    for w in weight_score:
        aa = sum(w, [])
        final_weight.append(aa)

    return final_weight,p_price

# 가중치를 이용한 새로운 가격 변수 생성
def make_new_price(weight_list, price_list):
    new_price = []
    new_price_m = 0
    new_price_s = 0

    for i in range(len(weight_list)):
        if weight_list[i]==[0]:
            new_price.append(0)
        else:
            for j in range(len(weight_list[i])):
                new_price_m += (weight_list[i][j] * price_list[i][j])
                new_price_s += weight_list[i][j]
            new_price_sum = new_price_m / new_price_s
            new_price.append(new_price_sum)
    return new_price

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    parser.add_argument('--name', type=str, choices=['squid', 'shrimp', 'salmon'], help='Choose products name')
    parser.add_argument('--head', type=int, default=10, help='Input the number of head of Multihead Attention')
    parser.add_argument('--head_size', type=int, default=32, help='Input the dimenstion of attention head')
    args = parser.parse_args()

    ## 중량 수온 데이터를 합친 dataframe, 품목 별로 불러오기
    # os.chdir('/content/drive/MyDrive/BIGCONTEST/data/real_use')

    # name = input('수산물 어종을 고르세요 (오징어/흰다리새우/연어)')

    name = args.name

    if name == "squid":
        data = pd.read_csv("./squid_temp1.csv")
        val_df = pd.read_csv("./squid_temp2.csv")
        name = "오징어"
    elif name == "shrimp":
        data = pd.read_csv("./shrimp_temp1.csv")
        val_df = pd.read_csv("./shrimp_temp2.csv")
        name = "흰다리새우"
    elif name == "salmon":
        data = pd.read_csv("./salmon_temp1.csv")
        val_df = pd.read_csv("./salmon_temp2.csv")
        name = "연어"
    else:
        None

    D_train = data_loading(data)
    train, t_len = D_train.datasetting(name)

    D_val = data_loading(val_df)
    validation, v_len = D_val.datasetting(name)
    ## 제조국과 수출국에 대하여 multi-hot encoding을 진행하기위해서 모든 나라를 리스트형태로 저장
    ctry_1 = set(list(set(train['CTRY_1']))+list(set(train['CTRY_2'])))
    ctry_2 = set(list(set(validation['CTRY_1']))+list(set(validation['CTRY_2'])))
    con = list(ctry_1) + list(ctry_2)
    ctry_list = list(set(con))

    x_train = D_train.data_frame(train, ctry_list)
    x_val = D_val.data_frame(validation, ctry_list)

    X_train = np.array(x_train.iloc[:,:-1])
    Y_train = np.array(x_train.iloc[:,-1])
    Y_train = Y_train.reshape(-1)
    
    X_val = np.array(x_val.iloc[:,:-1])
    Y_val = np.array(x_val.iloc[:,-1])
    Y_val = Y_val.reshape(-1)

    print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

    num_heads=args.head; head_size=args.head_size; ff_dim=None; dropout=0

    multi = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)

    # Attention 계산 및 FC layer 차원축소 
    attention_layers = ModelTrunk()

    train_weight, train_price = make_weight(X_train, Y_train, t_len)

    val_weight, val_price = make_weight(X_val, Y_val,v_len)

    # attention score의 가중치를 적용한 가중평균 
    train_price = make_new_price(train_weight, train_price)
    val_price = make_new_price(val_weight, val_price)

    train['REG_DATE'] = pd.to_datetime(train['REG_DATE'])
    validation['REG_DATE'] = pd.to_datetime(validation['REG_DATE'])

    # 가중평균한 최종 가격 배열로 만들기
    ts_train = np.array(train_price)

    ## 모델 피팅하기 위해서 dataframe 형태 수정
    df = pd.DataFrame(columns=['ds', 'y'])
    df['ds'] = train['REG_DATE'].unique()
    df['y'] = ts_train
    print(df.shape)

    df.to_csv(f"{args.name}_avg_price.csv")
    print("File saved complete.")