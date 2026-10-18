import string
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pickle

stemmer = None
stopwords_set = None
clf = None
vectorizer = None

def get_spam_tools():
    global stemmer, stopwords_set, clf, vectorizer
    
    if clf is None:
        factory_stemmer = StemmerFactory()
        stemmer = factory_stemmer.create_stemmer()
        
        factory_stopwords = StopWordRemoverFactory()
        stopwords_list = factory_stopwords.get_stop_words()
        important_words = {'anda', 'kamu', 'di', 'ke', 'dari', 'ini', 'itu', 'adalah'}
        stopwords_set = set(stopwords_list) - important_words
        
        with open('model_saved/rf_model.pkl', 'rb') as f:
            clf = pickle.load(f)

        with open('model_saved/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

def predict(raw_text: str):
    get_spam_tools()
    
    text_lower = raw_text.lower()
    
    if re.search(r'cutt\.ly|bit\.ly|s\.id|rekening diblokir|klik tautan|undian|hadiah rp', text_lower):
        return 'spam', 0.95
    
    email_words = text_lower.translate(str.maketrans('', '', string.punctuation)).split()
    email_words_filtered = [word for word in email_words if word not in stopwords_set]
    email_text_processed = stemmer.stem(' '.join(email_words_filtered))
    
    X_email = vectorizer.transform([email_text_processed])
    prediction = clf.predict(X_email)[0]
    proba = clf.predict_proba(X_email)[0]
    
    label = 'spam' if prediction == 1 else 'not_spam'
    return label, float(proba[prediction])