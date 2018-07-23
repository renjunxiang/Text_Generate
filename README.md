# generate_text
[![](https://img.shields.io/badge/Python-3.5,3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/pandas-0.23.0-brightgreen.svg)](https://pypi.python.org/pypi/pandas/0.23.0)
[![](https://img.shields.io/badge/numpy-1.14.3-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.14.3)
[![](https://img.shields.io/badge/keras-2.1.6-brightgreen.svg)](https://pypi.python.org/pypi/keras/2.1.6)
[![](https://img.shields.io/badge/tensorflow-1.4,1.6-brightgreen.svg)](https://pypi.python.org/pypi/tensorflow/1.6.0)<br>

文本生成

## **项目介绍**
最近生成型的网络越来越火，就试着模型写写诗歌。网上这方面的资料和现成的代码还是挺多的。不过大多都是tensorflow，少量torch，我平时keras用的多一些，就想着用keras来写一下，效果比tensorflow还是差不少<br>
tensorflow的代码参考了github一个比较火的项目<https://github.com/jinfagang/tensorflow_poems>，在此表示感谢！

## **模块简介**
### 模块结构
结构很简单，包括：<br>
* **数据**：全唐诗，来源<https://github.com/todototry/AncientChinesePoemsDB>，在此表示感谢！<br>
* **预处理**：data_poem.py是个脚本，用于合并、清洗每个txt文档；Data_process.py是个方法，用于分词、编码、填充<br>
* **网络**：在文件夹rnn中，model_keras.py、model_tensorflow.py分别是keras、tensorflow的2层lstm<br>
* **训练**：train_keras.py、train_tensorflow.py，分别用keras、tensorflow训练网络<br>
* **生成**：generate_keras.py、generate_tensorflow.py，分别用keras、tensorflow生成诗歌<br>

### 写诗遇到的问题
1.keras貌似不能对标签数据在网络内部做one-hot，所以标签会非常占内存，我用服务器96G内存都吃不消30000首诗5000长度字典的生成。<br>
<br>
2.同样的网络，keras训练效果不是特别好、loss在5以上，tensorflow的loss能降到1左右，不知道是不是因为没有对x和y做reshape拼接的原因。<br>
<br>
3.固定的网络输出是固定的。<br>
<br>
4.断句很难控制在5或者7。<br>

### 解决方案
1.keras我加入了分批训练时外部执行one-hot，但会大幅增加训练时间。<br>
<br>
2.在预测的时候不取唯一解，而是取最大概率的n个答案，或者对生成的结果做不同概率的抽样。<br>

### 其他说明
1.网上很多代码的做法是：每次输入一个字，输出lstm的hidden用于预测下一个字、state用于保存cell状态。下一个循环输入上一轮预测的字和state作为新一轮lsrm的cell初始状态。<br>
我觉得太麻烦了，直接保留整个序列，[1] > [2]  |  [1,2] > [3]  |  [1,2,3] > [4]。因为tensorflow里面有一个output = tf.reshape(outputs, [-1,num_units])，输出的就是这句话后移一个单位预测值。<br>
<br>
2.由于随机抽样很容易在标点生成的时候跳过标点，我一开始加入了修正：小于3个字符出现标点重新抽样直到字，多余7个字符不出现标点重新抽样直到标点。这里抽样也必须是最大概率的n个，而不是作弊的方式人为断句，这样的结果还是挺不错的，而且会出现4-7个字，比较有韵味，这段代码在generate_tensorflow_correct.py。<br>
但是后面发现，在抽样的时候不用均匀，而是给每个字附上计算得到的概率，RNN完全可以学到标点的位置，结果更贴近唐诗5、7字，也同样丢失了4、6字的风格。<br>
<br>
3.训练的时候batchsize是大于1的，生成的时候batchsize=1，cell_mul.zero_state这里要注意。所以要保存训练的参数，生成的时候模型结构有改变，再导入训练参数。<br>
<br>
4.闲来无事想试着生成小说，一开始用的网络小说生成的很差，后来找了射雕英雄传有那么点感觉。

## 成果展示
**train_keras.py、train_tensorflow.py进行训练，generate_keras.py、generate_tensorflow.py即可创作，参考demo_xxx.py**<br>
<br>
**Tensorflow 诗歌 随机抽样+修正**<br>
![](https://github.com/renjunxiang/generate_text/blob/master/picture/tensorflow_correct.jpg)<br><br>
**Tensorflow 诗歌 概率抽样**<br>
![](https://github.com/renjunxiang/generate_text/blob/master/picture/tensorflow.jpg)<br><br>
**Keras 诗歌 概率抽样**<br>
![](https://github.com/renjunxiang/generate_text/blob/master/picture/keras.jpg)<br><br>
**Tensorflow 小说 概率抽样**<br>
![](https://github.com/renjunxiang/generate_text/blob/master/picture/story.jpg)<br><br>

**有几首不错的诗大家看着乐呵乐呵**<br>
<br>
**神**<br>
神龙既已落云泉，犹恐风流四百声。<br>
几株红叶几重关，不得人间不敢归。<br>
<br>
**仙**<br>
仙驾千般出水滨，夜凉寒色向山眠。<br>
万里江边过远梦，一日同吟不见人？<br>
<br>
**鬼**<br>
鬼疟朝来入夜风，不似春窗入禁垣。<br>
云飞树下春山暮，水满江山雪未飞。<br>