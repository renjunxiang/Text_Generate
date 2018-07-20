import pandas as pd
import os
import pickle
from collections import Counter
import re

DIR = os.path.dirname(os.path.abspath(__file__))
path = DIR + '/data/'
# path = './data/'

# start = 'B'
# end = 'E'

texts = []
with open(path + 'story.txt', mode='r') as f:
    line=1
    while line:
        line=f.readline()
        text = re.sub(pattern='[_（）《》\s]', repl='', string=line)
        if text:
            texts.append(text)
        else:
            continue

with open(path + 'story.pkl', mode='wb') as f:
    pickle.dump(texts, f)

texts_len = Counter([len(i) for i in texts])
texts_len = pd.DataFrame({'length': list(texts_len.keys()), 'count': list(texts_len.values())},
                         columns=['length', 'count'])
texts_len = texts_len.sort_values(by='count', ascending=False).iloc[:20, ]
'''
    length  count
21      24   1278
25      28   1223
27      30   1208
28      31   1208
24      27   1206
23      26   1200
17      20   1188
20      23   1185
26      29   1184
22      25   1171
19      22   1166
18      21   1159
15      18   1147

'''
