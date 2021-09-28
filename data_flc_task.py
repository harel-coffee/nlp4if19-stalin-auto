from os import listdir
from collections import defaultdict
#from bpemb import BPEmb
import pickle
import numpy as np
from itertools import chain, zip_longest

ELMO_DATASETS = '../mehdi/datasets.npy'

def elmo_test_tokens(data, articles):    
    datasets = np.load(ELMO_DATASETS, allow_pickle=True)[None][0]
    keys, tokens, _ = datasets['train']
    elmo_s = {
        k: list(chain.from_iterable([line for i, ((aid, lid), line) in enumerate(zip(keys, tokens)) if aid == k]))
        for k in data }
    
    for k, v in articles.items():
        for i, (t1, t2) in enumerate(zip_longest(v['text'], elmo_s[k])):
            print(f'{i}\t{t1}\t{t2}')
        break
    
def elmo_test_sents(data, ck, sents):    
    datasets = np.load(ELMO_DATASETS, allow_pickle=True)[None][0]
    keys, tokens, _ = datasets['train']
    elmo_s = {
        k: [line for i, ((aid, lid), line) in enumerate(zip(keys, tokens)) if aid == k]
        for k in data }
    
    ml = 0
    for i, (t1, t2) in enumerate(zip_longest(sents, elmo_s[ck])):
        if t1 is not None:
            print(i, ''.join(t1), len(t1))
        else:
            print(i, None)
        print(i, ' '.join(t2), len(t2))
        if len(t2)>ml:
            ml = len(t2)
        print()
    print(ml)

def get_elmo_indices(data):
    datasets = np.load(ELMO_DATASETS, allow_pickle=True)[None][0]
    keys, tokens, _ = datasets['train']
    return {
        k: [i for i, ((aid, lid), line) in enumerate(zip(keys, tokens)) if aid == k]
        for k in data
    }
    
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
        labels = ['O' for _ in range(len(list(v)))]
        #assert len(labels) == len(list(v))
        for i, char in enumerate(v):
            for lk, lv in labels_a[k].items():
                for slv in lv:
                    if i in slv:
                        labels[i] = lk
        articles[k] = (v, labels)
    return articles

def tokenize_and_label(data, vecs):
    
    articles = defaultdict(dict)
    
    for k, (text, labels) in data.items():
        ntext = text.replace(' ', '_ ').replace('\n', '| ').split()
        
        nvecs = vecs[k]
        #bpe_text = bpe_tok.encode(text)
        #bpe_vecs = [0 if x == 10000 else x +1 for x in bpe_tok.encode_ids(text)]

        ntext, nlabels = assign_token_labels(ntext, labels)
        #bpe_labels = bio_labeling(bpe_labels)

        articles[k] = {'text':ntext, 'vectors':nvecs, 'labels':nlabels, 'tlen':len(ntext)}

    return articles

def assign_token_labels(text, labels):
    nlabels = [[set(['O'])]*len(x) for x in text]
    prev_i = 0
    for i, tok_len in enumerate([len(x) for x in text]):
        nlabels[i] = set(labels[prev_i:prev_i+tok_len]) 
        if nlabels[i] != set(['O']):
            nlabels[i] = nlabels[i].difference(set(['O']))
        prev_i += tok_len
    return text, nlabels

def assign_bpe_labels(bpe_text, labels):
    bpe_labels = [['O']*len(x) for x in bpe_text]
    prev_i = 0
    for i, tok_len in enumerate([len(x) for x in bpe_text]):
        bpe_labels[i] = labels[prev_i:prev_i+tok_len]
        prev_i += tok_len
        if bpe_labels[i] == []: # stupid newlineshit at the end of file which does not receive a label
            bpe_labels[i] = ['O']
            
    for i, (w, l) in enumerate(zip(bpe_text, bpe_labels)):
        if w.startswith('_'):
            if len(l) > 1:
                bpe_labels[i] = l[1]
            else:
                bpe_labels[i] = l[0]
        else:
            bpe_labels[i] = l[0] if l else 'O'
            
    return bpe_text, bpe_labels
                
def bio_labeling(bpe_labels):
    for i, lbl in enumerate(bpe_labels):
            if lbl != 'O': 
                if bpe_labels[i-1] == 'O' or i == 0:
                    bpe_labels[i] = 'B-' + bpe_labels[i]
                else:
                    bpe_labels[i] = 'I-' + bpe_labels[i]
    return bpe_labels

def format_articles(articles):
    with open('all_articles.txt', 'w') as f:
        for k, v in articles.items():
            f.write(f'### new article: {k}\n')
            for i, (sents, sent_lbls) in enumerate(zip(v['text'], v['labels'])):
                f.write(f'# new sentence: {i}\n')
                for word, label in zip(sents, sent_lbls): 
                    f.write('\t'.join([word, label])+'\n')

def generate_data(save_as_txt: bool = False, save_as_pickle: bool = False):
        text_a = read_data('/home/adamek/git/nlp4if-teamstalin/data/datasets/train-articles/')
        gold_a = read_gold_file('/home/adamek/git/nlp4if-teamstalin/data/datasets/train-labels-FLC/')
        
        vecs = get_elmo_indices(text_a)
        # change repetition to the class of which it is repeated 
        #text_a, gold_a = transform_repetition(text_a, gold_a)
        articles = assign_labels(text_a, gold_a)
        articles = tokenize_and_label(articles, vecs)
        #elmo_test(text_a, articles)
        
        articles = split_into_elmo_sentences(articles) #split_into_sentences(articles)
        
        if save_as_pickle:
            with open('/home/adamek/git/nlp4if-teamstalin/systems/lbl_articles-sents_elmo.pickle', 'wb') as f:
            #with open('plain-lbl_articles-sents_bpemb-vs1000-e100.pickle', 'wb') as f:
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
    with open('../systems/lbl_articles-sents_elmo.pickle', 'rb') as f:
        return pickle.load(f)


def split_into_elmo_sentences(articles):
    all_sents, all_vecs, all_lbls =  [], [], []
    for k, v in articles.items():
        sents = []
        sent = []
        for token in v['text']:
            #if token == '\n':
            if token.endswith('|'):
                sent.append(token)
                if sent:
                    sents.append(sent)
                sent = []
            else:
                sent.append(token)
        if sent != []:
            sents.append(sent + [token])
            sent = []
        
        new_lbls = []
        lbls = iter(v['labels'])
        #assert len(sents) == len(v['vectors'])
        for x in sents:
            new_lbls.append([[y if y != set([]) else set(['O']) for y in next(lbls)] for _ in range(len(x))])
        #elmo_test_sents(articles, k, sents)
        
        all_sents += sents
        all_vecs += v['vectors']
        all_lbls += new_lbls
        sents = []

    return all_sents, all_vecs, all_lbls #articles_sents
    

def split_into_sentences(articles):
    articles_sents = {}
    for k, v in articles.items():
        sents = []
        sent = []
        for token in v['text']:
            if token == '\n':
                if sent:
                    sents.append(sent)
                sent = []
            else:
                sent.append(token)
        new_vecs, new_lbls = [], []
        vecs = iter(v['vectors'])
        lbls = iter(v['labels'])
        for x in sents:
            new_vecs.append([next(vecs) for _ in range(len(x))])
            new_lbls.append([next(lbls) for _ in range(len(x))])
        articles_sents[k] = {'text':sents, 'vectors':new_vecs, 'labels':new_lbls}
        sents = []
    return articles_sents
    
#def load_bpemb_model():
#    return BPEmb(lang='en', vs=1000, dim=100, add_pad_emb=True)
        
if __name__ == '__main__':
    generate_data(save_as_txt=False, save_as_pickle=True)
    #get_label2i()
    
