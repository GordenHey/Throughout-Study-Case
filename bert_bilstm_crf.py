import numpy as np
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Masking, Embedding, Bidirectional, LSTM, Dense, Input, TimeDistributed, Activation
from keras.preprocessing import sequence
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras import backend as K
from keras.utils import np_utils
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model
from keras_bert import load_trained_model_from_checkpoint
from keras_contrib.layers import CRF

char_vocab_path = "./data/char_vocabs.txt"  # 字典文件
train_data_path = "./train_data.txt"  # 训练测试数据

special_words = ['<PAD>', '<UNK>']  # 特殊词表示

# CCL2021数据标签：
label2idx = {"O": 0, "B-Symptom": 1, "I-Symptom": 2, "B-Drug_Category": 3, "I-Drug_Category": 4, "B-Drug": 5,
             "I-Drug": 6, "B-Medical_Examination": 7, "I-Medical_Examination": 8, "B-Operation": 9, "I-Operation": 10}

# 索引和BIO标签对应
idx2label = {idx: label for label, idx in label2idx.items()}

# 读取字符词典文件
with open(char_vocab_path, "r", encoding="utf8") as fo:
    char_vocabs = [line.strip() for line in fo]
char_vocabs = special_words + char_vocabs

# 字符和索引编号对应
idx2vocab = {idx: char for idx, char in enumerate(char_vocabs)}
vocab2idx = {char: idx for idx, char in idx2vocab.items()}


# 读取训练语料
def read_corpus(corpus_path, vocab2idx, label2idx, flag):
    datas, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    if flag == "train":
        lines = lines[:int(len(lines)*0.7)]
    else:
        lines = lines[int(len(lines)*0.7):]
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            sent_ids = [vocab2idx[char] if char in vocab2idx else vocab2idx['<UNK>'] for char in sent_]
            tag_ids = [label2idx[label] if label in label2idx else 0 for label in tag_]
            datas.append(sent_ids)
            labels.append(tag_ids)
            sent_, tag_ = [], []
    return datas, labels


# 按照7：3划分训练集与测试集
train_datas, train_labels = read_corpus(train_data_path, vocab2idx, label2idx, flag="train")
test_datas, test_labels = read_corpus(train_data_path, vocab2idx, label2idx, flag="test")


# BERT模型【案例进度5】
model_path = "./chinese_L-12_H-768_A-12/"
bert = load_trained_model_from_checkpoint(
    model_path + "bert_config.json",
    model_path + "bert_model.ckpt",
    seq_len=128
)
for layer in bert.layers:
    layer.trainable = True

EPOCHS = 2
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_SIZE = 64
MAX_LEN = 100
VOCAB_SIZE = len(vocab2idx)
CLASS_NUMS = len(label2idx)
print(VOCAB_SIZE, CLASS_NUMS)

print('padding sequences')
train_datas = sequence.pad_sequences(train_datas, maxlen=MAX_LEN)
train_labels = sequence.pad_sequences(train_labels, maxlen=MAX_LEN)
test_datas = sequence.pad_sequences(test_datas, maxlen=MAX_LEN)
test_labels = sequence.pad_sequences(test_labels, maxlen=MAX_LEN)
print('x_train shape:', train_datas.shape)
print('x_train_label shape:', train_labels.shape)
print('x_test shape:', test_datas.shape)

# one-hot编码方式，稀疏矩阵，计算更快
train_labels = keras.utils.np_utils.to_categorical(train_labels, CLASS_NUMS)
test_labels = keras.utils.np_utils.to_categorical(test_labels, CLASS_NUMS)
print('trainlabels shape:', train_labels.shape)
print('testlabels shape:', test_labels.shape)

# BiLSTM【案例进度3】+CRF模型【案例进度1】构建
x1 = Input(shape=(MAX_LEN,), dtype='int32')
x2 = Input(shape=(MAX_LEN,), dtype='int32')
x = bert([x1, x2])

x = Masking(mask_value=0)(x)  # 屏蔽层，数据预处理屏蔽掉0
x = Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(x)  # 降维
x = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True))(x)  # BiLSTM
x = TimeDistributed(Dense(CLASS_NUMS))(x)  # 每一步结束后直接进行计算
outputs = CRF(CLASS_NUMS)(x)  # 添加约束，判断是否合法
model = Model(inputs=x, outputs=outputs)
model.summary()  # 输出各层参数，Param是神经元权重个数

model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_viterbi_accuracy])  # loss损失函数，metrics评价函数，optimizer优化器
model.fit(train_datas, train_labels, epochs=EPOCHS, verbose=1, validation_split=0.1)  # epochs轮数，verbose=1输出进度条，validation_split=0.1将0.1比例作为验证集

score = model.evaluate(test_datas, test_labels, batch_size=BATCH_SIZE)
print(model.metrics_names)
print(score)

# save model
model.save("./model/ch_ner_model.h5")