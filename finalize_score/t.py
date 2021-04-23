import pickle

with open('./scores/annotated_score_dict.pickle', 'rb') as fr:
    d = pickle.load(fr)

print(d)
