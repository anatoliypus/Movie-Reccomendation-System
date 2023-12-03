from tensorflow import keras
from keras.layers import Embedding, Reshape, Dot, Input
from keras.models import Model as KerasModel
import numpy as np

class Model(KerasModel):
    def __init__(self, n_users=943, m_items=1682, k_factors=100, **kwargs):
        # Input layers for user and item
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # Embedding layers for user and item
        P = Embedding(n_users, k_factors, input_length=2)(user_input)
        Q = Embedding(m_items, k_factors, input_length=1)(item_input)
        
        # Reshape layers
        P = Reshape((k_factors,))(P)
        Q = Reshape((k_factors,))(Q)

        # Dot product layer
        rating = Dot(axes=1)([P, Q])

        super(Model, self).__init__(inputs=[user_input, item_input], outputs=rating, **kwargs)

    def rate(self, users_id, items_id):
        return self.predict([np.array(users_id - 1), np.array(items_id - 1)], verbose=0)
    
def get_model(max_userid, max_movieid, k_factors):
    model = Model(max_userid, max_movieid, k_factors)
    model.compile(loss='mse', optimizer='adamax')
    return model