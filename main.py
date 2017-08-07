import email
import os
import re
from email.message import Message

import numpy as np

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


for dir in dirs:
    new_files = os.listdir(os.path.join(DATA_PATH, dir))
    for new_file in new_files:
        filepath = os.path.join(DATA_PATH, dir, new_file)
        message = load_email(filepath)
        if 'ham' in dir:
            ham_messages.append(message)
        elif 'spam' in dir:
            spam_messages.append(message)

y = np.concatenate([np.zeros(len(ham_messages)), np.ones(len(spam_messages))])
X = spam_messages + ham_messages

# remove html tags
html_tags = '<[^>]*>'
X = [re.sub(html_tags, "", content) for content in X]

# remove new line
X = [re.sub('\\n', " ", content) for content in X]

# remove tabs
X = [re.sub('\\t', " ", content) for content in X]

# remove numbers, commas, dots, etc.
X = [re.sub("[^a-zA-Z]", " ", content) for content in X]
