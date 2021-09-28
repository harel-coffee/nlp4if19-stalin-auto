from os import listdir
import pickle
import numpy as np
import random
import string
from collections import Counter, defaultdict
from typing import Tuple, List, Any, Iterator, Dict
from itertools import chain
from bpemb import BPEmb

# from imblearn.over_sampling import SMOTE

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, TimeDistributed, Activation, Dense, Masking, Input, Bidirectional
# import tensorflow.keras.backend as K

random.seed(3)


def get_dataset(data_path: str) -> list:
    dataset = []
    for dfile in listdir(data_path+'/data/'):
        with open(data_path+'/data/'+dfile, 'rb') as f:
            data = pickle.load(f)
        with open(data_path+'/gold/'+dfile[:-7]+'-gold.pickle', 'rb') as f:
            annotations = pickle.load(f)
            dataset.append((data, annotations))
    return dataset

def data2words(dataset: Tuple[List[int], List[int]], bpemb: object, char2i: Dict[bytes, int]) -> Tuple:
    # bemp
    
    print('NEW ARTICLE')
    
    i2char = {v:k for k, v in char2i.items()}
    x, y = dataset
    
    for xi, yi in zip(x, y):
        print(i2char[xi], yi)
        
    x = ''.join(list(map(lambda z: i2char[z].decode(), x)))
    
    yn = []
    xn = bpemb.encode(x)
    labels = [len(s) for s in xn]
    prev = 0
    for i, ll in enumerate(labels):
        yn.append(y[prev:prev+ll][0])
        prev += ll
        print(xn[i], yn[-1], y[prev:prev+ll])
        
    
    x = bpemb.encode_ids(x)
    
    
    return x, yn

def convert_dataset(dataset: list, char2i: dict, labels2i: dict) -> List[Any]:
    bpemb_en = BPEmb(lang='en', vs=200000, dim=300)
    
    d = {'O':1, 'B':2, 'I':3}
    for i, (a, l) in enumerate(dataset):
        a, l = data2words((a, l), bpemb_en, char2i)
        dataset[i] = (np.array(a), np.array([d[x] for x in l]))
        return
    return dataset
        

def load_x2is() -> Tuple[Dict[bytes, int], Dict[str, int]]:
    with open('/home/adamek/git/nlp4if-teamstalin/data/x2i/char2i.pickle', 'rb') as f:
        char2i = pickle.load(f)
    with open('/home/adamek/git/nlp4if-teamstalin/data/x2i/label2i.pickle', 'rb') as f:
        label2i = pickle.load(f)
    return char2i, label2i


def pad_batch(batch: list, max_len: int) -> List[Any]:
    #print('pad', batch.shape)
    for i, x in enumerate(batch):
        # obtain number of missing pad symbols
        missing = max_len - len(x)
        # pad symbol is 0 for both labels and chars
        batch[i] = np.append(batch[i], np.array([0 for _ in range(missing)]))
    return np.array(batch)


def batch_to_categorical(batch: List[Any], n_labels: int) -> List[Any]:
    b = np.zeros((batch.shape[0], batch.shape[1], n_labels))
    for i in range(batch.shape[0]):
        b[i] = to_categorical(batch[i], num_classes=n_labels)
    return b

def get_article_samples(chars: List[int], labels: List[int]) -> List[Any]:
    """
    For an article, find all spans with non-O labels:
    span size = (start_index - n, end_index + m) and return it.
    E.g. use all instaces of labels as training data: aka stupid sampling
    """
    article_samples = []
    b = 'i'
    for i, (x, y) in enumerate(zip(chars, labels)):
        if y == 2:
            b = i
        if y == 1 and isinstance(b, int):
            
            extended_context = b - i + random.randint(100, 300)
            
            if i+extended_context > len(chars):
                i = len(chars)
            else:
                i += extended_context
                
            if b-extended_context <= 0:
                b = 0
            else:
                b -= extended_context
            #print(b, i)
            #print(len([chars[k] for k in range(b,i)]), extended_context)
            #print([labels[k] for k in range(b, i)])
            
            if len([chars[k] for k in range(b,i)]) == 0:
                b = 'i'
                continue
            
            article_samples.append(([chars[k] for k in range(b,i)], [labels[k] for k in range(b, i)]))
            b = 'i'
            
    #print(article_samples[0])
    return article_samples


def train_valid_sample_gens(dataset: List[Any], batch_size: int = 8, valid_batches: int = 1) -> Tuple[Iterator[Tuple[List[Any], List[Any]]], Iterator[Tuple[List[Any], List[Any]]]]:
    valid_data = set(random.choices(list(range(len(dataset))), k=batch_size*valid_batches))
    train_data = set(list(range(len(dataset)))).difference(valid_data)
    
    valid_data = [dataset[x] for x in valid_data]
    train_data = [dataset[x] for x in train_data]
    
    train_data = list(chain.from_iterable([get_article_samples(x[0], x[1]) for x in train_data]))
    
    return batchify(train_data, batch_size), batchify(valid_data, batch_size)

def train_and_valid_gens(dataset: List[Any], batch_size: int = 8, valid_batches: int = 1) -> Tuple[Iterator[Tuple[List[Any], List[Any]]], Iterator[Tuple[List[Any], List[Any]]]]:
    valid_data = set(random.choices(list(range(len(dataset))), k=batch_size*valid_batches))
    train_data = set(list(range(len(dataset)))).difference(valid_data)

    valid_data = [dataset[x] for x in valid_data]
    train_data = [dataset[x] for x in train_data]
    
    return batchify(train_data, batch_size), batchify(valid_data, batch_size)


def batchify(dataset: List[Tuple[Any, Any]], batch_size: int, sort_articles: bool = True) -> Iterator[Tuple[List[Any], List[Any]]]:
    batch_x, batch_y = [], []
    
    if sort_articles:
        dataset = sorted(dataset, key=lambda x: len(x[0]))
    
    for article_data, article_gold in dataset:
        batch_x.append(article_data)
        batch_y.append(article_gold)
        if len(batch_x) == batch_size:
            max_len = max([len(x) for x in batch_x])
            yield pad_batch(batch_x, max_len), np.expand_dims(pad_batch(batch_y, max_len), -1)
            batch_x, batch_y = [], []
    #else:
    #    max_len = max([len(x) for x in batch_x])
    #    yield pad_batch(batch_x, max_len), pad_batch(batch_y, max_len)
    #    batch_x, batch_y = [], []
        
    #return None, None


def create_model(vocab: Dict[bytes, int], labels: list, batch_size: int) -> object:
    nn_size = 100
    embed_dim = 300

    model = Sequential()

    model.add(Embedding(10000,
                        embed_dim,
                        mask_zero=True,
                        batch_input_shape=(batch_size, None)))
    
    model.add(Masking(mask_value=0., input_shape=(batch_size, None)))
    model.add(Bidirectional(LSTM(nn_size,
                                 return_sequences=True,
                                 stateful=False,
                                 dropout=0.3)))
    
    #model.add(Bidirectional(LSTM(nn_size,
    #                             return_sequences=True,
    #                             stateful=False,
    #                             dropout=0.3)))
    
    #model.add(Bidirectional(LSTM(nn_size,
    #                             return_sequences=True,
    #                             stateful=False,
    #                             dropout=0.3)))
    
    #model.add(LSTM(nn_size, return_sequences=True))
    
    model.add(TimeDistributed(Dense(len(labels), activation='softmax')))
    
    model.compile(loss='sparse_categorical_crossentropy',#'mean_squared_error',
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy'])
    
    print(model.summary())
    
    return model


def run_model(dataset: list, batch_size: int, n_epochs: int) -> List[int]:
    char2i, label2i = load_x2is()
    labels = [0, 1, 2, 3]
    
    model = create_model(char2i, labels, batch_size)
    
    # validation losses
    vlosses = []
    
    # show predictions on validation data
    inspect = False
    
    for epoch in range(n_epochs):
        print('running epoch', epoch)
        tlosses = []
        train_generator, valid_generator = train_and_valid_gens(dataset,
                                                                batch_size=batch_size,
                                                                valid_batches=1)
        #train_generator, valid_generator = train_valid_sample_gens(dataset, 
        #                                                           batch_size=batch_size,
        #                                                           valid_batches=1)

        for i, (x, y) in enumerate(train_generator):
            #y = batch_to_categorical(y, len(labels))
            tloss = model.train_on_batch(x, y)
            tlosses.append(tloss[0])
            print('train batch loss:', tloss, x.shape)

        for x, y in valid_generator:
            #y = batch_to_categorical(y, len(labels))
            vloss = model.test_on_batch(x, y)
            print('--- --- ---')
            print('- train loss', np.mean(tlosses))
            print('-- valid batch loss', vloss, x.shape)
            vlosses.append(vloss[1])
            if inspect:
                f = defaultdict(int)
                pred = model.predict_on_batch(x)
                preds = np.argmax(pred, axis=-1)
                for d in list(range(pred.shape[0])):
                    gc = Counter(np.squeeze(y[d], -1))
                    pc = Counter(preds[d])
                    f['g0'] += gc[0]
                    f['p0'] += pc[0]
                    f['g1'] += gc[1]
                    f['p1'] += pc[1]
                    f['g2'] += gc[2]
                    f['p2'] += pc[2]
                    f['g3'] += gc[3]
                    f['p3'] += pc[3]
                for k, v in f.items():
                    print(k, v)

    return vlosses

def main():
    dataset_path = '/home/adamek/git/nlp4if-teamstalin/data/train/'
    # smote somewhere here  
    char2i, label2i = load_x2is()
    dataset = convert_dataset(get_dataset(dataset_path), char2i, label2i)

    print('dataset size:', len(dataset))
    filter_dataset = True
    if filter_dataset:
        dataset = list(filter(lambda x: len(x[0]) < 10000, dataset))
        print('dataset size after filter:', len(dataset))
    
    return
    batch_size = 8
    n_epochs = 9
    run_model(dataset, batch_size, n_epochs)
    
if __name__ == '__main__':
    main()
