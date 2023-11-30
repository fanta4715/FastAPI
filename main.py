from fastapi import FastAPI, HTTPException
# import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta

# 현재 날짜 가져오기


app = FastAPI()
# today = datetime.today()
today = datetime(2023, 6, 30) #임시로 넣은 값임


df = pd.read_csv("updated.csv").iloc[:,:2]
df['ds'] = pd.to_datetime(df['ds'], errors = 'coerce')
df['y'] = df['y'].str.replace(',', '').astype('int64')
df.loc[df['y'] == 0, 'y'] = df['y'].mean()
df['y'] = df['y'].round(0)
model = Prophet()

model.fit(df)

@app.get("/predict")
async def get_prediction(date: str):
    # 시작 날짜(오늘)와 종료 날짜(7일 후) 생성
    start_date = datetime.strptime(date, "%Y-%m-%d")
    start_date += timedelta(days=1)
    # start_date = today.strftime('%Y%m%d')
    end_date = (start_date + timedelta(days=6)).strftime('%Y%m%d')

    future_7days=pd.date_range(start=start_date, end=end_date, freq='D')
    future_7days = pd.DataFrame(future_7days, columns = ['ds'])
    future_7days['ds']= pd.to_datetime(future_7days['ds'])

    forecast = model.predict(future_7days) # 향후 11월 1일부터 11월 8일까지의 예측, dataFrame에 저장
    result = {
        "mean": forecast['yhat'].tolist()
    }
    return result

