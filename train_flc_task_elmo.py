from data_flc_task import load_data, get_label2i #, load_bpemb_model

#from bpemb import BPEmb
import numpy as np
import numpy.random as npr
import random
from collections import Counter
from itertools import combinations
from bert_serving.client import BertClient

from imblearn.over_sampling import SMOTE

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, TimeDistributed, Activation, Dense, Masking, Input, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.initializers import Constant

import gc
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='3'

ELMO_DIR = "../mehdi/elmo1024/"

seed_val = 3
npr.seed(seed_val)
random.seed(seed_val)

label2i = get_label2i()
#bpemb_model = load_bpemb_model()

def format_labels(labels):
    return [
        np.array([to_categorical(label2i[l], num_classes=len(label2i)) for l in x]).sum(0) if len(x) != 0 else to_categorical(0, num_classes=len(label2i))
        for x in labels
        ]

def pad_batch(batch, max_len):
    for i in range(len(batch)):
        if max_len - len(batch[i]) > 0:
            batch[i] = np.concatenate([
                np.array(batch[i]), 
                np.array([to_categorical(0, num_classes=len(label2i)) for _ in range(max_len - len(batch[i]))])
                ], 0)
    return np.array(batch, dtype=np.float32)

def sample_dataset(X, Y):
    sm = SMOTE() #{0: 1000, 1: 1000})
    X, Y = sm.fit_resample(X, Y)
    return X, Y

def prepare_sentences(dataset):
    all_text, all_vec_ids, all_lbls = dataset
    all_sents = [
        (vid, lbls)
        for vid, lbls in zip(all_vec_ids, all_lbls)
    ]
    npr.shuffle(all_sents)
    return all_sents

def data_generator_sents(all_sents, bs):
    while True:
        batch_x, batch_y, max_len = [], [], 129
        for xi, yi in all_sents:
            x = np.load(f'{ELMO_DIR}/train_X_sent_{xi}.npy')
            if x.sum() == 0:
                continue
            batch_x.append(x)
            batch_y.append(format_labels(yi))
            if len(batch_x) == bs:
                batch_y = pad_batch(batch_y, max_len)
                batch_y = batch_y[:, :, 1:]
                yield np.array(batch_x, dtype=np.float32), batch_y 
                batch_x, batch_y = [], []

def data_generator_sents_slc(all_sents, bs):
    while True:
        batch_x, batch_y, max_len = [], [], 129
        for xi, yi in all_sents:
            x = np.load(f'{ELMO_DIR}/train_X_sent_{xi}.npy')
            if x.sum() == 0:
                continue
            batch_x.append(x)
            batch_y.append(format_labels(yi))
            if len(batch_x) == bs:
                batch_y = pad_batch(batch_y, max_len)
                batch_y = batch_y[:,:,1:]
                batch_y = batch_y.max(1)
                yield np.array(batch_x, dtype=np.float32), batch_y 
                batch_x, batch_y = [], []

                
def resample_generator(X, Y, bs, data_lim):
    counter = 0
    while True:
        #if counter != 0:
        #    print('>>> samples per epoch:', counter)
        counter = 0
        
        #for i in range(1,len(label2i)-1):
        #    _mask1 = np.where(Y[:,0] == 1)[0]
        class_combs = list(combinations(range(0, len(label2i)-1), 2))
        npr.shuffle(class_combs)
        for i, k in class_combs:
            _mask1 = np.where(Y[:,k] == 1)[0]
            _mask2 = np.where(Y[:,i] == 1)[0]
            if len(_mask1) == 0 or len(_mask2) == 0:
                continue
            npr.shuffle(_mask1)
            npr.shuffle(_mask2)
            mask = np.concatenate([_mask1[:data_lim], _mask2[:data_lim//2]], 0)
            del _mask1
            del _mask2
            npr.shuffle(mask)
            Y_masked = Y[mask][:,i]
            X_masked = X[mask]
            # minority class must have at least 6 examples and there must be at least two classes:
            if min(Counter(Y_masked).values()) < 6 or len(Counter(Y_masked).values()) < 2:
                continue
            new_x, new_yn = sample_dataset(X_masked, Y_masked)
            counter += new_x.shape[0]
            # create multi-hots for all classes beging zeros
            new_y = np.zeros([new_yn.shape[0], len(label2i)-1])
            new_y[:, i] = new_yn
            new_y[:, k] = 1-new_yn
            # fake sentence:
            new_x = np.expand_dims(new_x, 1)
            new_y = np.expand_dims(new_y, 1)
            
            #yield new_x[:bs], new_y[:bs]
            indices = np.arange(new_x.shape[0])
            npr.shuffle(indices)
            
            indices = indices[:data_lim]
            new_x = new_x[indices]
            new_y = new_y[indices]
            
            for j in range(0, data_lim, bs):
                bx = new_x[j:j+bs]
                by = new_y[j:j+bs]
                if bx.shape[0] < bs:
                    bx = np.concatenate([bx, np.zeros([bs-bx.shape[0], bx.shape[1], bx.shape[2]])])
                    by = np.concatenate([by, np.zeros([bs-by.shape[0], by.shape[1], by.shape[2]])])
                    
                yield bx, by
                gc.collect()
            


def create_model(n_labels: int, batch_size: int = 1) -> object:
    dropout_rate=0.2
    nn_size = 512
    
    #inp = Input(batch_shape=(batch_size, 129, 1024))
    
    model1 = Sequential([
        Bidirectional(LSTM(nn_size,
                              return_sequences=False,
                              stateful=False,
                              dropout=dropout_rate),
                      batch_input_shape=[batch_size, 129, 1024]),
        Dense(nn_size, activation='relu'),
        Dense(n_labels-1, activation='softmax')
    ])
    #model1.compile(loss='binary_crossentropy',#'mean_squared_error',
    model1.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
    
    model2 = Sequential([
        TimeDistributed(Dense(nn_size, activation='relu'), 
                        batch_input_shape=[batch_size, None, 1024]),
        TimeDistributed(Dense(n_labels-1, activation='softmax'))
    ])
    model2.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

    model1.summary()
    model2.summary()
    
    return model1, model2

if __name__ == '__main__':
    dataset = load_data()
    bs = 512
    valid_batches = 9
    
    dataset = prepare_sentences(dataset)
    #dataset = dataset[:int(len(dataset) * 0.8)]
    print(label2i)
    
    valid_data = set(npr.choice(list(range(len(dataset))), size=bs*valid_batches, replace=False))
    train_data = set(list(range(len(dataset)))).difference(valid_data)
    
    valid_data = [dataset[x] for x in valid_data]
    train_data = [dataset[x] for x in train_data]
    del dataset
    
    train_gen = data_generator_sents_slc(train_data, bs)
    train_gen2 = data_generator_sents(train_data, bs)
    valid_gen = data_generator_sents(valid_data, bs)

    model1, model2 = create_model(len(label2i.values()), bs)
    
    gc.collect()
    #model1.fit_generator(generator=train_gen, 
    #                     steps_per_epoch=len(train_data)//bs, 
    #                     epochs=2)
    
    x, y = next(data_generator_sents(train_data, len(train_data)))
    print('t1', x.shape, y.shape)
    y = y.reshape([y.shape[0] * 129, len(label2i)-1])
    x = x.reshape([x.shape[0] * 129, 1024])
    print('t2', x.shape, y.shape)
    not_pad_token = np.where(y.max(-1) == 1)
    x = x[not_pad_token]
    y = y[not_pad_token]
    print('t3', x.shape, y.shape)
    
    data_lim = 2 ** 20
                
    model2.fit_generator(generator=resample_generator(x, y, bs, data_lim), 
                         steps_per_epoch=(data_lim // bs) * ((len(label2i) - 1) * (len(label2i) - 2)) // 2,#143270,
                         epochs=10)
    
    model2.fit_generator(generator=train_gen2, 
                         steps_per_epoch=len(train_data)//bs, 
                         epochs=5)
    del y
    del x
    gc.collect()
    
    i2label = {v:k for k, v in label2i.items()}
    rs = {i: {'tp': 0, 'tp+fn': 0, 'tp+fp': 0} for l, i in label2i.items()}
    
    for bi, (x, y) in enumerate(valid_gen):
        _pred_y = model2.predict_on_batch(x)
        pred_y = np.array(1/(len(label2i)-1) <= _pred_y, dtype=int)

        for l, i in label2i.items():
            if i > 0:
                rs[i]['tp'] += (y[:,:,i-1] * (pred_y[:,:,i-1] == y[:,:,i-1])).sum()
                rs[i]['tp+fp'] += pred_y[:,:,i-1].sum()
                rs[i]['tp+fn'] += y[:,:,i-1].sum()
                
        if bi == valid_batches:
            break
    
    print(f'precision, recall, total_true, total_pred,\tlabel')        
    for k, v in rs.items():
        pr = v['tp'] / v['tp+fp'] if v['tp+fp'] != 0 else -1
        re = v['tp'] / v['tp+fn'] if v['tp+fn'] != 0 else -1
        total_true  = v['tp+fn']
        total_pred = v['tp+fp']
        
        print(f'{pr:9.3f}, {re:6.3f}, {total_true:10.0f}, {total_pred:10.0f},\t{i2label[k]}')

    # save the models (as model1.h5 and model2.h5)
    model1.save('model1.h5')
    model2.save('model2.h5')
