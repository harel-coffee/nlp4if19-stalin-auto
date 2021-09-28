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
from keras.initializers import Constant

seed_val = 3
npr.seed(seed_val)
random.seed(seed_val)

label2i = get_label2i()
bpemb_model = load_bpemb_model()

def format_labels(labels):
    return [to_categorical(label2i[x], num_classes=len(label2i)) for x in labels]

def embed_vector(v, bpemb_model):
    return np.array([bpemb_model.vectors[x] for x in v])

def pad_batch(batch, max_len, if_y=False):
    for i in range(len(batch)):
        batch[i] = batch[i] + [to_categorical(0, num_classes=len(label2i)) if if_y else 0 for _ in range(max_len - len(batch[i]))]
    return np.array(batch)

def emb_pad_batch(batch, max_len, if_y=False):
    for i in range(len(batch)):
        batch[i] = batch[i] + [to_categorical(0, num_classes=len(label2i)) if if_y else np.zeros(300) for _ in range(max_len - len(batch[i]))]
    return np.array(batch)

def sample_dataset(dataset):
    X, Y = zip(*dataset)
    Y = np.array(list(map(lambda x: np.max(format_labels(x), 0)[2:].max(), Y)))
    X = np.array([bpemb_model.vectors[x].sum(0) for x in X])
    print(X.shape, Y.shape)
    #print(X.shape)
    sm = SMOTE()
    X, Y = sm.fit_resample(X, Y)
    print(X.shape, Y.shape)
    return 

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
    nn_size = 300
    embed_dim = 300
    
    #print(bpemb_model.vectors.shape)
    wm = np.zeros((1000,300))
    for i in range(wm.shape[0]):
        # set pad to 0
        if i == 999:
            wm[0] = bpemb_model.vectors[-1]
        else:
            wm[i+1] = bpemb_model.vectors[i]

    model = Sequential()
    model.add(Embedding(1000,
                        embed_dim,
                        embeddings_initializer=Constant(wm),
                        batch_input_shape=(batch_size, None)))
    
    model.add(Masking(mask_value=0, input_shape=(batch_size, None)))
    #model.add(Input(shape=(bs, None, embed_dim)))
    
    model.add(Bidirectional(LSTM(nn_size,
                                 return_sequences=True,
                                 stateful=False,
                                 dropout=0.3)))
    
    model.add(Bidirectional(LSTM(nn_size,
                                 return_sequences=True,
                                 stateful=False,
                                 dropout=0.3)))
    
    
    #model.add(Dense(n_labels-1, activation='softmax'))
    model.add(Dense(n_labels-1, activation='sigmoid'))
    
    
    
    #model.compile(loss='binary_crossentropy',#'mean_squared_error',
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    
    
    print(model.summary())
    
    return model

if __name__ == '__main__':
    dataset = load_data()
    bs = 128
    
    dataset = prepare_sentences(dataset)
    
    valid_data = set(npr.choice(list(range(len(dataset))), size=bs*1, replace=False))
    train_data = set(list(range(len(dataset)))).difference(valid_data)
    
    valid_data = [dataset[x] for x in valid_data]
    train_data = [dataset[x] for x in train_data]
    
    train_data = sample_dataset(train_data)
    """
    train_gen = data_generator_sents(train_data, bs)
    valid_gen = data_generator_sents(valid_data, bs)
    model = create_model(len(label2i.values()), bs)
    
    model.fit_generator(generator=train_gen, 
                        steps_per_epoch=len(train_data)//bs, 
                        epochs=9)
    
    i2label = {v:k for k, v in label2i.items()}
    results = {i: None for l, i in label2i.items()}
    for x, y in valid_gen:
        new_y = model.predict_on_batch(x)
        new_y = np.round(new_y)

        for l, i in label2i.items():
            #results[i] =  [(new_y[(y == i)] == i).sum() / ((new_y == i).sum()),
            #               (new_y[(y == i)] == i).sum() / ((y == i).sum())]
            results[i] =  [(y[:,:,i-1] * (new_y[:,:,i-1] == y[:,:,i-1])).sum() / new_y[:,:,i-1].sum(),
                           (y[:,:,i-1] * (new_y[:,:,i-1] == y[:,:,i-1])).sum() / y[:,:,i-1].sum()]

            
        for k, v in results.items():
            print(f'{v[0]:.3f}, {v[1]:.3f},\t{i2label[k]}')
        break
    """