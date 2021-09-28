from os import listdir
from collections import defaultdict
import pickle
import numpy as np
from itertools import chain, zip_longest

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'

current_usr = os.environ["USER"]
if current_usr == "mehdi":
    BASE_DIR = ".."
else:    
    BASE_DIR = f"/home/{current_usr}/git/nlp4if-teamstalin"

GROVER_DIR = "/scratch/mehdi/nlp4if/grover-mega"

import sys
sys.path.append(f'{BASE_DIR}/mehdi/grover-test/grover/')
from sample.encoder import get_encoder

encoder = get_encoder()

def get_grover_indices(data):
    # load datasets
    sorted_train_dataset = np.load(f'{BASE_DIR}/mehdi/grover-test/sorted_train_dataset.npy', allow_pickle=True)
    sorted_valid_dataset = np.load(f'{BASE_DIR}/mehdi/grover-test/sorted_valid_dataset.npy', allow_pickle=True)

    keys_train, tokens_train_label = list(zip(*sorted_train_dataset))
    keys_valid, tokens_valid_label = list(zip(*sorted_valid_dataset))
    tokens_train, _ = list(zip(*tokens_train_label))
    tokens_valid, _ = list(zip(*tokens_valid_label))
    keys = list(keys_train) + list(keys_valid)
    tokens = list(tokens_train) + list(tokens_valid)
    # pointers to the feature vectors
    out = {
        k: list(map(lambda x: x[1], sorted([
            (lid, i)
            for i, ((aid, lid), line) in enumerate(zip(keys, tokens))
            if aid == k
        ])))
        for k in data
    }
    return out

def read_data(train_f: str):
    articles = {}
    for file in listdir(train_f):
        with open(train_f+file) as f:
            articles[file[7:-4]] = ''.join([x for x in f.readlines()])
    return articles

def read_gold_file(gold_f: str):
    articles = {}
    for file in listdir(gold_f):
        with open(gold_f+'/'+file) as f:
            span_labels = defaultdict(list)
            for line in f.readlines():
                _, label, begin, end = line.rstrip().split('\t')
                span_labels[label].append(list(range(int(begin)+1, int(end))))
            articles[file[7:-16]] = span_labels
    return articles

def transform_repetition(articles_text, articles_labels):
    for article in articles_text.keys():
        #print(articles_text[article])
        if 'Repetition' in articles_labels[article].keys():
            rep_eq = defaultdict(list)
            for rep_span in articles_labels[article]['Repetition']:
                b, e = rep_span[0], rep_span[-1]
                rep = articles_text[article][b:e]
                # repetition found, find what is repeated
                for k, other_spans in articles_labels[article].items():
                    if k == 'Repetition':
                        continue
                    else:
                        for span in other_spans:
                            a, b = span[0], span[-1]
                            if articles_text[article][a:b] == rep:
                                rep_eq[k].append(span)
            del articles_labels[article]['Repetition']
            for k, new_spans in rep_eq.items():
                for v in new_spans:
                    articles_labels[article][k].append(v)
    return articles_text, articles_labels

def assign_labels(text_a, labels_a):
    articles = {}
    for k, v in text_a.items():
        labels = [{'O'} for _ in range(len(list(v)))]
        for i, char in enumerate(v):
            for lk, lv in labels_a[k].items():
                for slv in lv:
                    if i in slv:

                        labels[i].add(lk)
        articles[k] = (v, labels)
    return articles

def tokenize_and_label(data, vecs):
    articles = defaultdict(dict)
    for k, (text, labels) in data.items():
        #ntext = text.replace(' ', '_ ').replace('\n', '| ').split()
        ntext = [encoder.decode([ix]) for ix in encoder.encode(text)]
        #ntext = [
        #    encoder.decode([ix])+('|' if j == len(enc_line)-1 else '')
        #    for lines in [text.split('\n')]
        #    for i, line in enumerate(lines)
        #    for enc_line in [encoder.encode(line.strip())]
        #    for j, ix in enumerate(enc_line)
        #]
        nvecs = vecs[k]
        ntext, nlabels = assign_token_labels(ntext, labels)
        articles[k] = {'text':ntext, 'vectors':nvecs, 'labels':nlabels, 'tlen':len(ntext)}
    return articles

def assign_token_labels(text, labels):
    nlabels = [[set(['O'])]*len(x) for x in text]
    prev_i = 0
    adjust_n = 0
    for i, tok in enumerate(text):
        tok_len = len(tok)
        #if len(tok) == 1 and tok in {'�'}:
        if tok.find('�') > -1:
            adjust_n += 1
            #print(tok, tok_len)
            tok_len -= 0.5
        # label cluster on this token:
        curr_labels = labels[int(prev_i):int(prev_i+tok_len)]
        if len(curr_labels) == 0:
            curr_labels = [labels[int(prev_i)]]
            
        # unify the cluster into one set:
        nlabels[i] = set([]).union(*curr_labels)
        # remove extra 'O':
        nlabels[i] = nlabels[i].difference(set(['O']))
        # fix empty sets with 'O'
        if nlabels[i] == set([]):
            nlabels[i] = set(['O'])
        prev_i += tok_len
        
        if prev_i >= len(labels) and i < len(text)-1:
            # if we are here, we may be in trouble but it's ok
            print(prev_i, len(labels))
            print(i, adjust_n, tok, curr_labels, len(text), len(nlabels))
            print(text[i:])
            print(text)
            print("===="*10)
            break
    return text, nlabels

def format_articles(articles):
    with open('all_articles.txt', 'w') as f:
        for k, v in articles.items():
            f.write(f'### new article: {k}\n')
            for i, (sents, sent_lbls) in enumerate(zip(v['text'], v['labels'])):
                f.write(f'# new sentence: {i}\n')
                for word, label in zip(sents, sent_lbls): 
                    f.write('\t'.join([word, label])+'\n')

def generate_data(save_as_txt: bool = False, save_as_pickle: bool = False):
    text_a = read_data(f'{BASE_DIR}/data/datasets/train-articles/')
    gold_a = read_gold_file(f'{BASE_DIR}/data/datasets/train-labels-FLC/')

    vecs = get_grover_indices(text_a)
    # change repetition to the class of which it is repeated 
    text_a, gold_a = transform_repetition(text_a, gold_a)
    articles = assign_labels(text_a, gold_a)
    articles = tokenize_and_label(articles, vecs)

    articles = split_into_grover_sentences(articles)
    if save_as_pickle:
        with open(f'{BASE_DIR}/systems/plain-lbl_articles-sents_grover.pickle', 'wb') as f:
            pickle.dump(articles, f)
    if save_as_txt:
        format_articles(articles)

def get_label2i():
    _, _ , lbls = load_data()
    label2i = {'<PAD>':0}
    for vals in lbls:
        for v in vals:
            for lbl in v:
                if lbl not in label2i:
                    label2i[lbl] = len(label2i)
    return label2i

def load_data():
    with open(f'{BASE_DIR}/systems/plain-lbl_articles-sents_grover.pickle', 'rb') as f:
        return pickle.load(f)

def split_into_grover_sentences(articles):
    all_sents, all_vecs, all_lbls =  [], [], []
    for k, v in articles.items():
        print(k)
        sents = []
        sent = []
        print('len lbl / text', len(v['labels']), len(v['text']))
        for token, lbl in zip(v['text'], v['labels']):
            if token == '\n':
            #if token[-1] == '|':
                #sent.append((token, lbl))
                if len(sent) > 2:
                    sents.append(sent)
                sent = []
            else:
                sent.append((token, lbl))
        if sent != []:
            sents.append(sent + [(token, lbl)])
            sent = []
        
        # test grover vector alignments:
        print('len sent / grover-sent-vector', len(sents), len(v['vectors']))
        #assert len(sents) == len(v['vectors'])
        for i, (sent, xi) in enumerate(zip(sents, v['vectors'])):
            vec = np.load(f'{GROVER_DIR}/train_{xi}.npy')
            # dirty hack:
            if len(sent) > vec.shape[0]:
                sents[i] = sent[:-1]
                print(len(sent), vec.shape, sent[-1])
                
            if len(sents[i]) != vec.shape[0]:
                #xi1 = v['vectors'][i-1]
                #xi2 = v['vectors'][i+1]
                #vec_1 = np.load(f'{GROVER_DIR}/train_{xi1}.npy')
                #vec_2 = np.load(f'{GROVER_DIR}/train_{xi2}.npy')
                #print(-1, len(sents[i-1]), vec_1.shape[0]) #, sents[i-1])
                print('sent', i, len(sents[i]), vec.shape[0], sents[i])
                stext, _ = zip(*sents[i])
                print(''.join(stext))
                print()
                #print(+1, len(sents[i+1]), vec_2.shape[0], sents[i+1])
                
            #assert len(sents[i]) == vec.shape[0]
        new_sents = []
        new_lbls = []
        for sent, vec in zip(sents, v['vectors']):
            sent, lbl = list(zip(*sent))
            new_sents.append(sent)
            new_lbls.append(lbl)

        all_sents += new_sents
        all_vecs += v['vectors']
        all_lbls += new_lbls
        sents = []
    return all_sents, all_vecs, all_lbls #articles_sents

if __name__ == '__main__':
    generate_data(save_as_txt=False, save_as_pickle=True)
