

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)## with_info=True 参数返回一个元组，包含数据集examples和元数据metadata，as_supervised=True参数以二元元组结构形式加载数据集
train_examples, val_examples = examples['train'], examples['validation'] ## 打标签



## 从训练数据集创建自定义子词分词器（subwords tokenizer），将单词分解为子词
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)
## build_from_corpus从文本语料库中构建SubwordTextEncoder，该方法接受两个参数：1.符串的可迭代对象(en.numpy() for pt, en in train_example) and (pt.numpy() for pt, en in train_examples). 2.目标词汇大小(2**13)
## tfds.features.text用于处理文本数据，SubwordTextEncoder将文本编码为整数序列
print(tokenizer_en) ## 输入
print(tokenizer_pt) ## 预测


sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string) ## .encode编码获得编译的字符串
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string) ## .decode解码获得原字符串
print ('The original string: {}'.format(original_string))

assert original_string == sample_string ## 若两个字符串不相等将引发AssertionEorr异常，检查句



## 如果单词不在词典中，则分词器（tokenizer）通过将单词分解为子词来对字符串进行编码
for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))
    ## 7915 ----> T
    ## 1248 ----> ran
    ## 7946 ----> s
    ## 7194 ----> former 
    ## 13 ----> is 
    ## 2799 ----> awesome
    ## 7877 ----> .

BUFFER_SIZE = 20000
BATCH_SIZE = 64

##将开始和结束标记（token）添加到输入和目标
def encode(lang1, lang2):
  ## TensorFlow张量是一种多维数组，可以包含数字、字符串及其他类型，张量可具有任意数量的维度，每个维度可以有任意长度
  ## tokenizer_en.vocab_size为一个整数，表示SubwordTextEncoder对象中词汇表的大小，词汇表包含所有可能的单词和子词
  lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size+1]  ##lang1.numpy()是一个tensorFlow张量，包含文本数据但是张量不能支持所有numpy操作，.numpy()方法用于将张量转换为numpy数组
  lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size+1]
  return lang1, lang2



## 删除长度超过40的样本
MAX_LENGTH = 40
def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)
## 函数返回一个布尔值，如果x和y都小于等于max_length返回True
## tf.logical_and()比较两个张量对应位置元素并返回新张量，张量的每个元素都是对应位置元素的逻辑与运算结果



train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# 将数据集缓存到内存中以加快读取速度。
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)