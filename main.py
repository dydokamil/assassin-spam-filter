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

y = np.concatenate([np.zeros(len(ham_messages)), np.ones(len(spam_messages))])
X = spam_messages + ham_messages
