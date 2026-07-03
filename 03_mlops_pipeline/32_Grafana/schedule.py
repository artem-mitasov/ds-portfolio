import dill
import pandas as pd

from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
from datetime import datetime

sched = BlockingScheduler(timezone=tzlocal.get_localzone_name())

df = pd.read_csv('model/data/homework.csv')
with open('model/cars_pipe.pkl', 'rb') as f:
    model_data = dill.load(f)

model = model_data['model']


@sched.scheduled_job('interval', seconds=5)
def run_model():
    data = df.sample(n=5).copy()

    preds = model.predict(data)
    result = data[['id', 'price']].copy()
    result['predicted_price_cat'] = preds

    print('\n', result)


if __name__ == '__main__':
    sched.start()