import string
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopwords = StopWordRemoverFactory()
stopwords_list = factory_stopwords.get_stop_words()
important_words = {'anda', 'kamu', 'di', 'ke', 'dari', 'ini', 'itu', 'adalah'}
stopwords_set = set(stopwords_list) - important_words

cache_file = 'model_saved/stem_cache.pkl'
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        stem_cache = pickle.load(f)
    print(f"Total kata di dalam stem_cache saat ini: {len(stem_cache)}")
else:
    print("Membuat cache baru")
    stem_cache = {}

df = pd.read_csv('assets/email_spam.csv')
df = df.dropna(subset=['Pesan', 'Kategori'])

# df = df.sample(n=1500, random_state=42)
print(f"Jumlah baris setelah dibersihkan & dibatasi: {len(df)}")

df['text'] = df['Pesan'].astype(str)
df['label_num'] = (df['Kategori'] == 'spam').astype(int)
df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

print("Text stemming")
corpus = []

line_total = len(df)

for i in range(len(df)):
    print(f"line processing: {i + 1} / {line_total}", end="\r")
    
    text = df['text'].iloc[i].lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    
    filtered_words = []
    for word in text:
        if word not in stopwords_set:
            if word in stem_cache:
                filtered_words.append(stem_cache[word])
            else:
                stemmed_word = stemmer.stem(word)
                stem_cache[word] = stemmed_word
                filtered_words.append(stemmed_word)
                
    corpus.append(' '.join(filtered_words))

print()

os.makedirs('model_saved', exist_ok=True)
with open(cache_file, 'wb') as f:
    pickle.dump(stem_cache, f)

print(f"Total akhir kata unik di stem_cache: {len(stem_cache)}")
    
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus).toarray()
y = df.label_num

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)
print("Model selesai di-training")

with open('model_saved/rf_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('model_saved/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
    
print("Proses selesai")