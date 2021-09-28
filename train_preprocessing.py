from os import listdir
from collections import defaultdict
import pickle
from typing import Dict, List

def read_article(apath):
    with open(apath, 'rb') as f:
        return pickle.load(f)

def read_gold_file(gold_file):
    span_labels = defaultdict(list)
    spans = []
    with open(gold_file) as f:
        for line in f.readlines():
            _, label, begin, end = line.rstrip().split('\t')
            span_labels[label].append((int(begin), int(end)))
            spans += list(range(int(begin),int(end)))
    return span_labels, spans

def get_b(ann: List[str]) -> List[str]:
    prev = ''
    for i, x in enumerate(ann):
        if prev == 'O' and x != 'O':
            ann[i] = 'B-' + x
        elif i == 0 and x != 'O':
            ann[i] = 'B-' + x
        prev = x
    return ann

def span_label_annotations(article: list, spans: Dict[str, List[int]]) -> list:
    anns = ['O' for x in article]
    for i, x in enumerate(article):
        for l, lspans in spans.items():
            for x in lspans:
                if i in lspans:
                    anns[i] = 'I-' + l
    return anns
            
def span_annotations(article, spans):
    return get_b(['I' if i in spans else 'O' for i, _ in enumerate(article)])
            
def create_label2i(gold_dir, save=False):
    label2i = defaultdict(int)
    label2i['<pad>'] = 0
    for gold_file in listdir(gold_dir):
        span_labels, spans = read_gold_file(gold_dir+gold_file)
        for label, _ in span_labels.items():
            if label not in label2i:
                label2i[label] = len(label2i)
    if save:
        with open('label2i.pickle', 'wb') as f:
            pickle.dump(label2i, f)

    return label2i

def annotate_data_sentences(article_dir, gold_dir, label2i, save=False):
    pass

def annotate_data(article_dir, gold_dir, label2i, save=False, inspect_labels=True):
    dataset = []

    label_c = defaultdict(int)
    
    for article_file in listdir(article_dir):
        print('reading', article_file)
        article_path = article_dir+article_file
        gold_path = gold_dir+article_file[:-7] + '.task-FLC.labels'
        span_labels, spans = read_gold_file(gold_path)
        article = read_article(article_path)
        annotations = span_annotations(article, spans)

        for c in annotations:
            if c == 'O':
                label_c[0] += 1
            else:
                label_c[1] += 1
        if save:
            with open(f'/home/adamek/git/nlp4if-teamstalin/data/train/gold/{article_file[:-7]}-gold.pickle', 'wb') as f:
                pickle.dump(annotations, f)

        dataset.append((article, annotations))

    if inspect_labels:
        print(label_c)
        lsum = sum(label_c.values())
        for k, v in label_c.items():
            print(k, v, v/lsum)

    return dataset

def save_dataset(dataset):
    with open('annotated_data.pickle', 'wb') as f:
        pickle.dump(dataset, f)
    return None

if __name__ == '__main__':
    gold_dir = '/home/adam/git/nlp4if-teamstalin/data/datasets/train-labels-FLC/'
    label2i = create_label2i(gold_dir)
    data2i_article_paths = '/home/adam/git/nlp4if-teamstalin/data/train/data/'

    dataset = annotate_data(data2i_article_paths, gold_dir, label2i)
    #save_dataset(dataset)
    





    
