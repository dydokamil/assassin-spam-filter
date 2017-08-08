"""Dataset taken from http://spamassassin.apache.org/old/publiccorpus/"""

import email
import os
import re
from email.message import Message

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score, train_test_split

DATA_PATH = '/media/hdd/training_data/spamham'
dirs = os.listdir(DATA_PATH)

spam_messages = []
ham_messages = []


def load_email(path):
    with open(filepath, encoding='latin-1') as f:
        message = email.message_from_string(f.read()).get_payload()

    while isinstance(message, Message) or isinstance(message, list):
        if isinstance(message, list):
            message = message[0].get_payload()
        if isinstance(message, Message):
            message = message.get_payload()

    assert isinstance(message, str)
    return message


def cleanup(messages):
    # remove html tags
    html_tags = '<[^>]*>'
    messages = [re.sub(html_tags, "", content) for content in messages]

    # remove new line
    messages = [re.sub('\\n', " ", content) for content in messages]

    # remove tabs
    messages = [re.sub('\\t', " ", content) for content in messages]

    # remove numbers, commas, dots, etc.
    messages = [re.sub("[^a-zA-Z]", " ", content) for content in messages]

    # remove emessagecess whitespaces
    messages = [re.sub("\s+", " ", content) for content in messages]

    # remove nbsp
    messages = [re.sub('nbsp', '', content) for content in messages]

    # remove empty strings
    messages = [message.split(' ') for message in messages]
    messages = [[word for word in words if word] for words in messages]

    # remove empty messages
    messages = [message for message in messages if message]

    # turn messages into strings again
    messages = [' '.join(message) for message in messages]

    return messages


def stem(messages):
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer('english')
    messages = [[stemmer.stem(word) for word in words.split(' ') if word is not ''] for words in messages]
    return messages


def remove_stopwords(messages):
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))
    messages = [[word for word in words if word not in stopwords] for words in messages]
    # turn messages into strings again
    messages = [' '.join(message) for message in messages]
    return messages


for dir in dirs:
    new_files = os.listdir(os.path.join(DATA_PATH, dir))
    for new_file in new_files:
        filepath = os.path.join(DATA_PATH, dir, new_file)
        message = load_email(filepath)
        if 'ham' in dir:
            ham_messages.append(message)
        elif 'spam' in dir:
            spam_messages.append(message)

X_spam = cleanup(spam_messages)
X_ham = cleanup(ham_messages)

X_spam = stem(X_spam)
X_ham = stem(X_ham)

X_spam = remove_stopwords(X_spam)
X_ham = remove_stopwords(X_ham)

y = np.concatenate([np.zeros(len(X_spam)), np.ones(len(X_ham))])
# X = np.concatenate([X_spam, X_ham])
X = X_spam + X_ham

assert len(X) == len(y)

# split into train/test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# CountVectorizer
cv = CountVectorizer(min_df=0.005, analyzer='word', ngram_range=(1, 3))
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# StratifiedKFold()

tfidf_transformer = TfidfTransformer(norm='l1', use_idf=True)
X_train = tfidf_transformer.fit_transform(X_train)
X_test = tfidf_transformer.transform(X_test)

clf = RandomForestClassifier(30)
clf.fit(X_train, y_train)

print(cross_val_score(clf, X_test, y_test, scoring='accuracy'))
print(clf.score(X_test, y_test))
