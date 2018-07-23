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
print(texts_len)
'''
    length  count
141     145     65
101     105     65
106     110     63
103     107     63
145     149     62
105     109     62
95       99     60
102     106     60
94       98     60
143     147     58
66       70     57
107     111     56
68       72     55
98      102     55
104     108     55
134     138     54
142     146     54
67       71     54
96      100     53
178     182     53

'''
