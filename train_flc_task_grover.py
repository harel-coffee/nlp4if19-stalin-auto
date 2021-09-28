from data_flc_task import load_data, get_label2i 

import numpy as np
import numpy.random as npr
import random
from collections import Counter
from itertools import combinations
from time import time

from imblearn.over_sampling import SMOTE

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, TimeDistributed, Activation, Dense, Masking, Input, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.initializers import Constant

import gc
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='3'

GROVER_DIR = "/scratch/mehdi/nlp4if/grover-mega"

sorted_dev_dataset = np.load('../mehdi/grover-test/sorted_dev_dataset.npy', allow_pickle=True)


seed_val = 3
npr.seed(seed_val)
random.seed(seed_val)

label2i = get_label2i()
i2label = {v:k for k, v in label2i.items()}

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
    sm = SMOTE() 
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
            x = np.load(f'{GROVER_DIR}/train_{xi}.npy')
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
            x = np.load(f'{GROVER_DIR}/train_{xi}.npy')
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
    global n_epoch
    n_epoch = 0
    counter = 0
    while True:
        n_epoch += 1
        counter = 0
        
        new_x, new_y = sample_dataset(X, Y)
        gc.collect()
        
        # shuffle
        indices = np.arange(new_x.shape[0])
        npr.shuffle(indices)
        new_x, new_y = new_x[indices], new_y[indices]
        gc.collect()
        
        # fix the shape
        new_x = np.expand_dims(new_x, 1)
        new_y = np.expand_dims(new_y, 1)

        for j in range(0, new_x.shape[0], bs):
            bx = new_x[j:j+bs]
            by = new_y[j:j+bs]
            if bx.shape[0] < bs:
                bx = np.concatenate([bx, np.zeros([bs-bx.shape[0], bx.shape[1], bx.shape[2]])], 0)
                by = np.concatenate([by, np.zeros([bs-by.shape[0], by.shape[1], by.shape[2]])], 0)
            yield bx, by
            gc.collect()

                
def resample_generator2(X, Y, bs, data_lim):
    global n_epoch
    global class_state
    n_epoch = 0
    counter = 0
    
    while True:
        n_epoch += 1
        #if counter != 0:
        #    print('>>> samples per epoch:', counter)
        counter = 0
        class_combs = list(combinations(range(0, len(label2i)-1), 2))
        npr.shuffle(class_combs)
        for c1, c2 in class_combs:
            class_state = (c1, c2)
            _mask1 = np.where(Y[:,c1] == 1)[0]
            _mask2 = np.where(Y[:,c2] == 1)[0]
            _mask1set = set(_mask1) - set(_mask2)
            _mask2set = set(_mask2) - set(_mask1)
            _mask1 = np.array(list(_mask1set))
            _mask2 = np.array(list(_mask2set))
            if len(_mask1) == 0 or len(_mask2) == 0:
                continue
            npr.shuffle(_mask1)
            npr.shuffle(_mask2)
            mask = np.concatenate([_mask1[:data_lim], _mask2[:data_lim//2]], 0)
            del _mask1
            del _mask2
            npr.shuffle(mask)
            Y_masked = Y[mask][:,c1]
            X_masked = X[mask]
            # minority class must have at least 6 examples and there must be at least two classes:
            if min(Counter(Y_masked).values()) < 6 or len(Counter(Y_masked).values()) < 2:
                continue
            new_x, new_yn = sample_dataset(X_masked, Y_masked)
            counter += new_x.shape[0]
            # create multi-hots for all classes beging zeros
            new_y = np.zeros([new_yn.shape[0], len(label2i)-1])
            new_y[:, c1] = new_yn
            new_y[:, c2] = 1-new_yn
            # fake sentence:
            new_x = np.expand_dims(new_x, 1)
            new_y = np.expand_dims(new_y, 1)
            
            #yield new_x[:bs], new_y[:bs]
            indices = np.arange(new_x.shape[0])
            npr.shuffle(indices)
            
            new_x = new_x[indices]
            new_y = new_y[indices]

            for j in range(0, new_x.shape[0], bs):
                bx = new_x[j:j+bs]
                by = new_y[j:j+bs]
                if bx.shape[0] < bs:
                    bx = np.concatenate([bx, np.zeros([bs-bx.shape[0], bx.shape[1], bx.shape[2]])], 0)
                    by = np.concatenate([by, np.zeros([bs-by.shape[0], by.shape[1], by.shape[2]])], 0)
                yield bx, by
                gc.collect()

def create_model(n_labels: int, batch_size: int = 1) -> object:
    dropout_rate=0.5
    nn_size = 1536
    #inp = Input(batch_shape=(batch_size, 129, 1024))
    """
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
    """
    model2 = Sequential([
        TimeDistributed(Dense(nn_size, activation='relu'), 
                        batch_input_shape=[batch_size, None, 1024]),
        Dropout(dropout_rate),
        TimeDistributed(Dense(n_labels-1, activation='softmax'))
    ])
    model2.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

    #model1.summary()
    model2.summary()
    return model2 #model1, model2

if __name__ == '__main__':
    dataset = load_data()
    bs = 1024
    valid_batches = 3
    dataset = prepare_sentences(dataset)
    
    valid_data = set(npr.choice(list(range(len(dataset))), size=bs*valid_batches, replace=False))
    train_data = set(list(range(len(dataset)))).difference(valid_data)
    
    valid_data = [dataset[x] for x in valid_data]
    train_data = [dataset[x] for x in train_data]
    del dataset
    
    train_gen = data_generator_sents_slc(train_data, bs)
    train_gen2 = data_generator_sents(train_data, bs)
    valid_gen = data_generator_sents(valid_data, bs)

    model2 = create_model(len(label2i.values()), bs)
    gc.collect()
    
    # all training data at once:
    x, y = next(data_generator_sents(train_data, len(train_data)))
    print('t1', x.shape, y.shape)
    y = y.reshape([y.shape[0] * 129, len(label2i)-1])
    x = x.reshape([x.shape[0] * 129, 1024])
    print('t2', x.shape, y.shape)
    not_pad_token = np.where(y.max(-1) == 1)
    x = x[not_pad_token]
    y = y[not_pad_token]
    print('t3', x.shape, y.shape)
    
    # fix multilabels:
    multi_label = np.where(y.sum(-1) > 1)[0]
    xi_multilabel, y_multilabel = np.where(y[multi_label] == 1)
    print('t4', xi_multilabel.shape)
    
    not_multi_label = np.where(y.sum(-1) <= 1)[0]
    y = np.concatenate([y[not_multi_label], to_categorical(y_multilabel, num_classes=len(label2i)-1)], 0)
    x = np.concatenate([x[not_multi_label], x[xi_multilabel]], 0)
    print('t5', x.shape, y.shape)

    data_lim = 2 ** 32 # no limit?
    n_epoch = 0
    history1 = []
    s = time()
    class_state = (0,0)
    for i, (_x, _y) in enumerate(resample_generator(x, y, bs, data_lim)):
        if n_epoch == 11:
            break
        h = model2.train_on_batch(_x, _y)
        history1.append([i, n_epoch] + list(class_state) + list(h))
        # report
        if i % 10 == 0:
            e = time()
            print("e {0}, step {1}, loss {2[0]:5.3f}, acc {2[1]:5.3f}, time {3:5.3f}s, {4[0]}-{4[1]}".format(
                n_epoch,
                i,
                np.mean(history1[-10:],0)[-2:],
                e-s,
                class_state
            ))
            s = time()

    model2.save('grover_before_second_train.h5')
    np.save('grover_history1.npy', history1)

    del y
    del x
    gc.collect()

    # report 1
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
        tt = v['tp+fn']
        tp = v['tp+fp']
        print(f'{pr:9.3f}, {re:6.3f}, {tt:10.0f}, {tp:10.0f},\t{i2label[k]}')

    # finetune
    history2 = model2.fit_generator(generator=train_gen2,
                         steps_per_epoch=len(train_data)//bs,
                         epochs=5)
    model2.save('grover_after_second_train.h5')
    np.save('grover_history2.npy', history2.history)

    # report 2
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
        tt = v['tp+fn']
        tp = v['tp+fp']
        print(f'{pr:9.3f}, {re:6.3f}, {tt:10.0f}, {tp:10.0f},\t{i2label[k]}')
