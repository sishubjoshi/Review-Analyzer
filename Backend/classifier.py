import pickle
import os
from vectorizer import vect, dataprocessing

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                 'pickles',
                 'classifier.pickle'), 'rb'))


def classify(text):
    label = {0: 'Negative', 1: 'Positive'}
    print('-'*10,text,sep='\n')
    clr = dataprocessing(text)
    X=vect.transform(clr)
    y = clf.predict(X)[0]
    return label[y]