from flask import Flask, render_template, request
import pickle
import os
import numpy as np
from vectorizer import vect, dataprocessing

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                 'pickles',
                 'classifier.pickle'), 'rb'))

def classify(text):
    label = {0: 'negative', 1: 'positive'}
    print('-'*10,text,sep='\n')
    clr = dataprocessing(text)
    X=vect.transform(clr)
    y = clf.predict(X)[0]
    return label[y]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/success',methods=['POST'])
def success():
    if request.method == 'POST':
        
        review = request.form['user_review']
        Y = classify(review)
        return render_template('success.html',val=Y)


if __name__ == '__main__':
    app.run(debug=True)