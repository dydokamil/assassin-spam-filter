import os
import numpy as np

DATA_PATH = '/media/hdd/training_data/spamham'
dirs = os.listdir(DATA_PATH)

spam_files = []
ham_files = []

for dir in dirs:
    new_files = os.listdir(os.path.join(DATA_PATH, dir))
    for new_file in new_files:
        with open(os.path.join(DATA_PATH, dir, new_file), encoding='latin-1') as f:
            if 'ham' in dir:
                try:
                    ham_files.append(f.read())
                except Exception as e:
                    print(e)
            elif 'spam' in dir:
                try:
                    spam_files.append(f.read())
                except Exception as e:
                    print(e)

y = np.concatenate([np.zeros(len(ham_files)), np.ones(len(spam_files))])
X = np.concatenate([spam_files, ham_files])
