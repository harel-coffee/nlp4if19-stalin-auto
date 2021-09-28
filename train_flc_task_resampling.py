from data_flc_task import load_data, get_label2i, load_bpemb_model

from bpemb import BPEmb
import numpy as np
import numpy.random as npr
import random
from collections import Counter

from imblearn.over_sampling import SMOTE

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, TimeDistributed, Activation, Dense, Masking, Input, Bidirectional
from tensorflow.keras.initializers import Constant

import gc
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

seed_val = 3
npr.seed(seed_val)
random.seed(seed_val)

label2i = get_label2i()
bpemb_model = load_bpemb_model()

def format_labels(labels):
    return [to_categorical(label2i[x], num_classes=len(label2i)) for x in labels]

#def embed_vector(v, bpemb_model):
#    return np.array([bpemb_model.vectors[x] for x in v])

def pad_batch(batch, max_len, if_y=False):
    for i in range(len(batch)):
        batch[i] = batch[i] + [to_categorical(0, num_classes=len(label2i)) if if_y else 0 for _ in range(max_len - len(batch[i]))]
    return np.array(batch, dtype=np.float32)

def sample_dataset(X, Y):
    sm = SMOTE()
    #Y = np.max(Y[:,2:], -1)
    #print('s1', X.shape, Y.shape)
    #print('s2', Counter(Y))
    sm = SMOTE({0: 1000, 1: 1000})
    X, Y = sm.fit_resample(X, Y)
    return X, Y

def prepare_sentences(articles):
    all_sents = []
    for v in articles.values():
        for i, x in enumerate(v['vectors']):
            all_sents.append((x, v['labels'][i]))
    npr.shuffle(all_sents)
    return all_sents

def data_generator_sents(all_sents, bs):
    while True:
        batch_x, batch_y, max_len = [], [], 0    
        for xi, yi in all_sents:
            batch_x.append(xi)
            batch_y.append(format_labels(yi))
            if len(xi) > max_len:
                max_len = len(xi)
                
            if len(batch_x) == bs:
                batch_x = pad_batch(batch_x, max_len, False)
                batch_y = pad_batch(batch_y, max_len, True)
                batch_y = batch_y[:,:,1:]
                yield batch_x, batch_y 
                batch_x, batch_y, max_len = [], [], 0

def data_generator(articles, bs):
    while True:
        batch_x, batch_y, max_len = [], [], 0
        for k, v in articles.items():
            if len(v['vectors']) > max_len:
                max_len = len(v['vectors'])
            batch_x.append(v['vectors'])
            batch_y.append(format_labels(v['labels']))
            
            if len(batch_y) == bs:
                batch_x = pad_batch(batch_x, max_len, False)
                batch_y = pad_batch(batch_y, max_len, True)
                batch_y = batch_y[:,:,1:]
                yield batch_x, batch_y 
                batch_x, batch_y, max_len = [], [], 0
                
def create_model(n_labels: int, batch_size: int) -> object:
    nn_size = 100
    embed_dim = bpemb_model.vectors.shape[-1]
    
    wm = np.zeros((1000,embed_dim))
    for i in range(wm.shape[0]):
        # set pad to 0
        if i == 999:
            wm[0] = bpemb_model.vectors[-1]
        else:
            wm[i+1] = bpemb_model.vectors[i]

    inp = Input(batch_shape=(batch_size, None))
    emb = Embedding(1001,
                    embed_dim,
                    embeddings_initializer=Constant(wm),
                    batch_input_shape=(batch_size, None))(inp)
    
    emb_masked = Masking(mask_value=0, input_shape=(batch_size, None, embed_dim))(emb)
    # + dropout, + epochs in model_top
    out1 = Bidirectional(LSTM(nn_size,
                              return_sequences=True,
                              stateful=False,
                              dropout=0.5))(emb_masked, training=True)

    model_base = Model(inp, out1)

    
    model_top = Sequential([
        TimeDistributed(Dense(nn_size, activation='relu'), 
                        batch_input_shape=[batch_size, None, 2*nn_size]),
        TimeDistributed(Dense(n_labels-1, activation='softmax'))
    ])
    
    model_top0 = Sequential([
        Bidirectional(LSTM(nn_size,
                           return_sequences=True,
                           stateful=False,
                           dropout=0.3),
                      batch_input_shape=[batch_size, None, 2*nn_size],
                    ),
        TimeDistributed(Dense(nn_size, activation='relu')),
        TimeDistributed(Dense(n_labels-1, activation='softmax'))
    ])

    model = Model(model_base.input, model_top(model_base.output))

    #model.compile(loss='binary_crossentropy',#'mean_squared_error',
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model_base.summary()
    model_top.summary()
    model.summary()
    
    return model, model_base, model_top

if __name__ == '__main__':
    dataset = load_data()
    bs = 256
    
    dataset = prepare_sentences(dataset)
    dataset = dataset[:int(len(dataset))]
    
    valid_data = set(npr.choice(list(range(len(dataset))), size=bs*1, replace=False))
    train_data = set(list(range(len(dataset)))).difference(valid_data)
    
    valid_data = [dataset[x] for x in valid_data]
    train_data = [dataset[x] for x in train_data]
    del dataset
    #train_data = sample_dataset(train_data)
    
    train_gen = data_generator_sents(train_data, bs)
    valid_gen = data_generator_sents(valid_data, bs)
    model, model_base, model_top = create_model(len(label2i.values()), bs)
    
    model.fit_generator(generator=train_gen, 
                        steps_per_epoch=len(train_data)//bs, 
                        epochs=1)
    gc.collect()
    model_top.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    
    x, y = next(data_generator_sents(train_data, len(train_data)))

    print('t1', x.shape, y.shape)
    x_vec = np.zeros([0, model_base.output_shape[-1]], dtype=np.float32)
    for i in range(0, len(x), bs):
        bx = x[i:i+bs]
        if bx.shape[0] < bs:
            y = np.concatenate([y, np.zeros((bs-bx.shape[0], y.shape[1], y.shape[2]), dtype=np.float32)], 0)
            bx = np.concatenate([bx,np.zeros((bs-bx.shape[0],bx.shape[1]), dtype=np.float32)], 0)
            
        vx = model_base.predict(bx)
        vx = vx.reshape([vx.shape[0]*vx.shape[1], vx.shape[2]])
        x_vec = np.concatenate([x_vec, vx], 0)
        del bx
        del vx
        gc.collect()

    del x
    print('t2', x_vec.shape, y.shape)
    y = y.reshape([x_vec.shape[0], len(label2i)-1])
    print('t3', x_vec.shape, y.shape)
    
    
    def resample_generator(X, Y, bs):
        #indices = np.arange(X.shape[0])
        counter = 0
        while True:
            #np.random.shuffle(indices)
            #X = X[indices]
            #gc.collect()
            #Y = Y[indices]
            #gc.collect()
            if counter != 0:
                print('>>> samples per epoch:', counter)
            counter = 0
            for i in range(2,len(label2i)-1):
                _mask = (Y[:,[1,i]].max(-1) == 1)
                npr.shuffle(_mask)
                minority = (_mask != 0)
                mask = np.concatenate([_mask[_mask == 0][:int(1.5*minority.shape[0])], _mask[minority]], 0) 
                del _mask 
                npr.shuffle(mask)
                Y_masked = Y[mask][:,i]
                X_masked = X[mask]
                if min(Counter(Y_masked).values()) < 6:
                    continue
                #print(Y_masked.shape)
                
                new_x, new_yn = sample_dataset(X_masked, Y_masked)
                counter += new_x.shape[0]
                new_y = np.zeros([new_yn.shape[0], len(label2i)-1])
                new_y[:, i] = new_yn
                new_y[:, 1] = 1-new_yn
                # fake sentence:
                new_x = np.expand_dims(new_x, 1)
                new_y = np.expand_dims(new_y, 1)
                for j in range(0, len(new_x), bs):
                    bx = new_x[j:j+bs]
                    by = new_y[j:j+bs]
                    if bx.shape[0] < bs:
                        bx = np.concatenate([bx, np.zeros([bs-bx.shape[0], bx.shape[1], bx.shape[2]])])
                        by = np.concatenate([by, np.zeros([bs-by.shape[0], by.shape[1], by.shape[2]])])
                        
                    yield bx, by
                    gc.collect()

    model_top.fit_generator(generator=resample_generator(x_vec, y, bs), 
                            steps_per_epoch=41602,
                            epochs=3)
        
    i2label = {v:k for k, v in label2i.items()}
    results = {i: None for l, i in label2i.items()}
    for x, y in valid_gen:
        new_y = model.predict_on_batch(x)
        new_y = np.array(1/(len(label2i)-1) <= new_y, dtype=int)

        for l, i in label2i.items():
            results[i] =  [(y[:,:,i-1] * (new_y[:,:,i-1] == y[:,:,i-1])).sum() / new_y[:,:,i-1].sum(),
                           (y[:,:,i-1] * (new_y[:,:,i-1] == y[:,:,i-1])).sum() / y[:,:,i-1].sum(),
                           y[:,:,i-1].sum()]

            
        for k, v in results.items():
            print(f'{v[0]:.3f}, {v[1]:.3f}, {v[2]}\t{i2label[k]}')
        break
