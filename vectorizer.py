from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize 
import os
import pickle
CUR_DIR = os.path.dirname(__file__)
stop_words = pickle.load(open(os.path.join(CUR_DIR,'pickles','stopwords.pickle'),'rb'))
ss=SnowballStemmer('english')
def dataprocessing(text):
    sent = text.lower()
    words = word_tokenize(sent)
    clrwds = [ss.stem(word) for word in words if not word in stop_words]
    rt = " ".join(clrwds)
    return(rt)

vect = TfidfVectorizer(ngram_range= (1,2), tokenizer=dataprocessing)

print(vect.transform(['this was a good movie']))