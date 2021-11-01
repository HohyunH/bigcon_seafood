# bigcon_seafood

본 저장소는 2021 빅콘테스트 / 챔피언스 리그 / 수산 Biz 부분 실험 코드입니다.
- 현재 공모전은 2차 피티 심사 준비중에 있으며, 대회 종료후 공개하도록 하겠습니다.

https://www.bigcontest.or.kr/points/content.php#ct03

** Contributor

김보상 : data.liszt@gmail.com

정영상 : videorighter@ds.seoultech.ac.kr

황호현 : hhhwang94@ds.seoultech.ac.kr

한주희 : fgtr153@ds.seoultech.ac.kr

## 문제 정의 :  수산물 수입가격 예측을 통한 최적의 가격 예측 모형 도출 
![image](https://user-images.githubusercontent.com/46701548/139639501-98aa5640-a504-414f-8bb1-c885044fca5f.png)

## 데이터 수집 및 전처리
![image](https://user-images.githubusercontent.com/46701548/139639558-c1a6d9f4-47ac-47e2-9c81-91d5ecd186b6.png)

+ 외부 요인(제품에 대한 중량 / 수온 데이터) 추가
+ 외부 요인이 있는 데이터만 필터링 하여 사용


## 하루치 가격 데이터 가중 평균
![image](https://user-images.githubusercontent.com/46701548/139639799-96be404a-5b47-456c-aa58-ae0163c86334.png)

### Attention 가중치를 이용한 평균 금액 산출
![image](https://user-images.githubusercontent.com/46701548/139639903-5552f6d0-d5e3-4807-89be-5178e0740665.png)

```python
from tensorflow_addons.layers import MultiHeadAttention

num_heads=args.head; head_size=args.head_size; ff_dim=None; dropout=0

multi = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
```

## 가중 평균한 가격 기반, 예측 모델 적용
![image](https://user-images.githubusercontent.com/46701548/139640215-f2bc314d-bc64-487b-add4-7ad43a72c066.png)

#### Prophet 모델 사용
- Prophet Reference : https://facebook.github.io/prophet/
```python
# Neural Prophet - Advanced Facebook Prophet
from neuralprophet import NeuralProphet

df = pd.DataFrame(columns=['ds', 'y'])
df['ds'] = train['REG_DATE'].unique()
df['y'] = ts_train

# hyperparameter 
freq = 'W-MON' # 시작일이 월요일이면서, 일주일 간격 
epochs = 1000


# 모델 학습
m = NeuralProphet()
metrics = m.fit(df, freq =freq, epochs=epochs)
```


- dataloading.py : 데이터 전처리
- attention_weight_avg.py : attention 기반 가중 평균 가격 산출
- attention_neuralprophet.ipynb : 전체 작업 및 Prophet 모델 사용 가격 예측
