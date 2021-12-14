import os
import sys
import nltk
import string
import nltk.corpus
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

classes = []
model = None
X_train = None
Y_train = None
X_test = None
Y_test = None
vectorizer = None
tfidf_transformer = None
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

df=pd.read_csv(FILE_PATH + "/../data/drugsComTrain_raw.csv")
all_conditions = df['condition']
all_reviews = df['review']
print(all_reviews.head(10))
print(all_conditions.head(10))
print(all_reviews.shape)

for category in all_conditions:
    if category not in classes:
        classes.append(category)

print('num categories..',len(classes))

def clean_corpus(raw_corp):
    clean_docs = []
    for sentence in raw_corp:
        doc = [word for word in sentence.split() if word not in stop_words]
        doc = [lemmatizer.lemmatize(word) for word in doc]
        clean_docs.append(' '.join(doc))
    return clean_docs

def vectorize():
    global X_train
    global Y_train
    global X_test
    global Y_test
    global vectorizer
    global tfidf_transformer
    y_all = [ classes.index(x) for x in all_conditions ]
    X_train, X_test, Y_train, Y_test = train_test_split(clean_corpus(all_reviews), y_all, test_size=0.15, random_state=42)
    vectorizer = HashingVectorizer(ngram_range=(1,2), strip_accents='ascii', n_features=2**18)
    X_train_hashed = vectorizer.transform(X_train)
    tfidf_transformer=TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train_hashed)
    X_test_hashed = vectorizer.transform(X_test)
    X_test = tfidf_transformer.transform(X_test_hashed)
    print('xtrain shape ...',X_train.shape)

def build_model():
    global model
    epoch = 5
    batchsize = 1000
    model = SGDClassifier(loss="hinge", penalty="l2")
    batches = int(X_train.shape[0]/batchsize) + 1
    samples = X_train.shape[0]
    for i in range(epoch):
        for j in range(batches):
            #print('in j...', j, j*batchsize, '----2is:',samples, (j+1)*batchsize )
            model.partial_fit(X_train[j*batchsize:min(samples,(j+1)*batchsize)], Y_train[j*batchsize:min(samples,(j+1)*batchsize)], classes=range(len(classes)))
    print ("Accuracy on testing data :", epoch, model.score(X_test, Y_test))

def process_input(input_statement):
    test_stmt = []
    test_stmt.append(input_statement)
    X_testing_counts = vectorizer.transform(clean_corpus(test_stmt))
    X_testing = tfidf_transformer.transform(X_testing_counts)
    predicted = model.predict(X_testing)
    for doc, category in zip(test_stmt, predicted):
        return classes[category]

def cmd_chat():
    while True:
        inp = input("Type your symptom here:")
        if inp.lower() == "quit":
            break
        resp = process_input(inp)
        if resp is None:
            resp = 'Could not categorize your condition, please rephrase'
        print('Medical Assist: ', resp)

def train():
    vectorize()
    build_model()

train()
cmd_chat()
