from os import listdir
from collections import defaultdict
import pickle
import numpy as np

def read_article_sentences(apath):
    article_sentences = []
    with open(apath) as f:
        for line in f.readlines():
            article_sentences.append(line.lower())
    return article_sentences
    
def read_article(apath):
    article_raw_text = ''
    with open(apath) as f:
        for line in f.readlines():
            article_raw_text += line.lower()
    return article_raw_text

def create_char2i(article_dir):
    char2i = defaultdict(int)
    char2i['<pad>'] = 0
    for article_file in listdir(article_dir):
        text = read_article(article_dir+article_file)
        for ch in text:
            if ch.encode() not in char2i:
                char2i[ch.encode()] = len(char2i)
                
    with open('/home/adamek/git/nlp4if-teamstalin/data/x2i/char2i.pickle', 'wb') as f:
        pickle.dump(char2i, f)

def data2int(article_dir, char2i):
    for article_file in listdir(article_dir):
        print(f'reading {article_file}')
        article = read_article(article_dir+article_file)
        article = np.array([char2i[x.encode()] for x in article])
        with open(f'/home/adamek/git/nlp4if-teamstalin/data/train/data/{article_file[:-4]}.pickle', 'wb') as f:
            pickle.dump(article, f)
        print('done')
        
        
if __name__ == '__main__':
    article_dir = '/home/adamek/git/nlp4if-teamstalin/data/datasets/train-articles/'
    create_char2i(article_dir)
    with open('/home/adamek/git/nlp4if-teamstalin/data/x2i/char2i.pickle', 'rb') as f:
        char2i = pickle.load(f)
    data2int(article_dir, char2i)
