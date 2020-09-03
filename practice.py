from lightfm import LightFM
import numpy as np
from lightfm.datasets import fetch_movielens

data = fetch_movielens(min_rating=5.0)
print(repr(data['train']))
print(repr(data['test']))


model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

# precision_at_k는 학습된 모델이 도출한 scores의 상위 k개 데이터에 known_positives가 얼마나 들어 있는지를 0~1의 실수로 나타낸다.
from lightfm.evaluation import precision_at_k
print("Train precision: %.2f" % precision_at_k(model, data['train'], k=5).mean())
print("Test precision: %.2f" % precision_at_k(model, data['test'], k=5).mean())

def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices] # userid에 해당하는 사람이 이미 본 영화
        scores = model.predict(user_id, np.arange(n_items)) # model.predict(25, [0, 1, 2, 3, 4... 1682]) #모든 영화에 점수를 매김
        top_items = data['item_labels'][np.argsort(-scores)] # argsort returns ascending indices

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)

sample_recommendation(model, data, [3, 25, 450])


'''
필요한 것
user*item matrix
model = lightFM()
model.fit(matrix, epoch=30)
model.predict(user_id, list(items))
'''