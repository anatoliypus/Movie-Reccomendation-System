import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from tensorflow import keras
import sys

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_PATH = os.path.join(ROOT_DIR, 'benchmark/data')
CHECKPOINTS_FOLDER = os.path.join(ROOT_DIR, 'models')
CHECKPOINTS_PATH = os.path.join(CHECKPOINTS_FOLDER, 'v2_keras_model.h5')
K_FACTORS = 100
sys.path.append(CHECKPOINTS_FOLDER)
import model as create_model

data = pd.read_csv(os.path.join(DATA_PATH, 'data_processed.csv'))
max_userid = data['user_id'].drop_duplicates().max()
max_movieid = data['item_id'].drop_duplicates().max()

shuffled_ratings = data.sample(frac=1., random_state=42)
Users = shuffled_ratings['user_emb_id'].values
Movies = shuffled_ratings['item_emb_id'].values
Ratings = shuffled_ratings['rating'].values

model = create_model.get_model(max_userid, max_movieid, K_FACTORS)
model.load_weights(CHECKPOINTS_PATH)

user_ratings = data[['user_id', 'item_id', 'rating']]
user_ids = data['user_id']
item_ids = data['item_id']
user_ratings['prediction'] = model.rate(user_ids, item_ids)
user_ratings['prediction_rounded'] = user_ratings['prediction'].round().astype(int)
accuracy = (user_ratings['prediction_rounded'] == user_ratings['rating']).sum() / user_ratings.shape[0]

print('Accuracy:', accuracy)
