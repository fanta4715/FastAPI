from fastapi import FastAPI, HTTPException
import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta

# 현재 날짜 가져오기


app = FastAPI()
today = datetime.today()

model = Prophet()

time = ['2023-09-01', '2023-09-02','2023-09-03','2023-09-04','2023-09-05','2023-09-06','2023-09-07','2023-09-08','2023-09-09','2023-09-10','2023-09-11','2023-09-12','2023-09-13','2023-09-14','2023-09-15','2023-09-16','2023-09-17','2023-09-18','2023-09-19','2023-09-20','2023-09-21','2023-09-22','2023-09-23','2023-09-24','2023-09-25','2023-09-26','2023-09-27','2023-09-28','2023-09-29','2023-09-30','2023-10-01','2023-10-02','2023-10-03','2023-10-04','2023-10-05','2023-10-06','2023-10-07','2023-10-08','2023-10-09','2023-10-10','2023-10-11','2023-10-12','2023-10-13','2023-10-14','2023-10-15','2023-10-16','2023-10-17','2023-10-18','2023-10-19','2023-10-20','2023-10-21','2023-10-22','2023-10-23','2023-10-24','2023-10-25','2023-10-26','2023-10-27','2023-10-28','2023-10-29','2023-10-30','2023-10-31']
sales = [935167, 781900, 599600,731100 ,770600 ,843300 ,934100 ,979800 ,1133900 ,437296 ,821500 ,997100 ,639200 ,982700 ,855700 ,922500 ,546200 ,873400 ,869760 ,691900 ,1110200 ,1027300 ,993400 ,592400 ,889700 ,933800,998100 ,486000 ,0 ,293400 ,0 ,581400 ,473900 ,600600 ,995600 ,1148900 ,939700 ,457200 ,566200 ,1018300 ,956800 ,769900 ,942800 ,823100 ,458100 ,789500 ,862900 ,955200 ,1003075 ,990500 ,911900 ,478800 ,968100 ,1093300 ,1036500 ,967400 ,901400 ,982800 ,340600 ,847700 ,879500]

df = pd.DataFrame({'ds' : time, 'y' : sales})
df['ds'] = pd.to_datetime(df['ds'], errors = 'coerce')

model.fit(df)

@app.get("/predict/")
async def get_prediction():
    # 시작 날짜(오늘)와 종료 날짜(7일 후) 생성
    start_date = today.strftime('%Y%m%d')
    end_date = (today + timedelta(days=7)).strftime('%Y%m%d')

    future_7days=pd.date_range(start=start_date, end=end_date, freq='D')
    future_7days = pd.DataFrame(future_7days, columns = ['ds'])
    future_7days['ds']= pd.to_datetime(future_7days['ds'])

    forecast = model.predict(future_7days) # 향후 11월 1일부터 11월 8일까지의 예측, dataFrame에 저장
    result = {
        "dates": forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
        "lower_bound": forecast['yhat_lower'].tolist(),
        "upper_bound": forecast['yhat_upper'].tolist(),
        "mean": forecast['yhat'].tolist()
    }
    return result
