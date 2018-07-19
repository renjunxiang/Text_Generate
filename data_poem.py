import pandas as pd
import os
import pickle
from collections import Counter
import re

DIR = os.path.dirname(os.path.abspath(__file__))
path = DIR + '/data/Tang_Poetry'
files = os.listdir(path)

# start = 'B'
end = 'E'

texts = []
texts_str = ''
for file in files:
    with open(path + '/' + file, mode='r') as f:
        text = f.readlines()
        if text:
            text = text[0]
            text = re.sub(pattern='[_（）《》 ]', repl='', string=text)
            texts.append(text + end)
            texts_str += text
        else:
            continue

# with open(DIR + '/data/Tang_Poetry.txt', mode='w') as f:
#     f.write(texts_str)

with open(DIR + '/data/Tang_Poetry.pkl', mode='wb') as f:
    pickle.dump(texts, f)

texts_len = Counter([len(i) for i in texts])
texts_len = pd.DataFrame({'length': list(texts_len.keys()), 'count': list(texts_len.values())},
                         columns=['length', 'count'])
texts_len = texts_len.sort_values(by='count', ascending=False).iloc[:10, ]
'''
     length  count
44       50  12190
60       66   7451
28       34   7250
20       26   2290
68       74   2260
92       98   1663
116     122    844
140     146    565
56       62    499
36       42    432

'''
