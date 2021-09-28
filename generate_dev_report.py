from data_flc_task import load_data, get_label2i

import numpy as np
from tensorflow.keras.models import load_model

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'

# load this dictionary later:
rule_based_slogan = {
    '12345': [(12, 20), (300,50)]
}


model_names = [
    'elmo+blstm-smote18.h5'
    # "elmo_multi_sampling_20e.h5",
    # "elmo_multi_sampling_20e_5e.h5",
    # "elmo_multi_sampling_20e+lstm_10e.h5",
    # "elmo_multi_sampling_10e.h5",
    # "elmo_multi_sampling_10e_5e.h5",
    # "elmo_multi_sampling_10e+lstm_10e.h5",
]

def load_dev_vectors():
    print(111)
    vecdir = '/scratch/mehdi/nlp4if/elmo1024/'
    dev_vecs = np.zeros((len(list(filter(lambda x: x.startswith('dev'), os.listdir('/scratch/mehdi/nlp4if/elmo1024/')))), 1024))
    for i, filename in enumerate(filter(lambda x: x.startswith('dev'), os.listdir('/scratch/mehdi/nlp4if/elmo1024/'))):
        dev_vecs[i] = np.load(vecdir+filename)
    print(dev_vecs.shape)

def get_data():
    load_dev_vectors()
    datasets = np.load('../mehdi/datasets.npy', allow_pickle=True)[None][0]
    #dev_vectors = np.load('../mehdi/dev_elmo_vectors.npy', allow_pickle=True)

    label2i = get_label2i()
    i2label = {v:k for k, v in label2i.items()}

    # generate
    keys, values, _ = datasets['dev']
    article_ids = sorted({aid for aid, _ in keys})

def generate_report():

    for model_name in model_names:
        model = load_model(f"saved_models/{model_name}")
        # predictions
        _pred_y = model.predict(dev_vectors)

        # naive threshold:
        base_threshold = 1/(len(label2i)-1)

        # aggresive threshold:
        agg_threshold0 = np.mean([
            ys
            for (aid, line_no), line, onehots in zip(keys, values, _pred_y)
            for w, ys in zip(line, onehots)
        ], 0)

        agg_threshold1 = agg_threshold0 / np.max([
            ys
            for (aid, line_no), line, onehots in zip(keys, values, _pred_y)
            for w, ys in zip(line, onehots)
        ], 0)

        agg_threshold2 = base_threshold * np.max([
            ys
            for (aid, line_no), line, onehots in zip(keys, values, _pred_y)
            for w, ys in zip(line, onehots)
        ], 0)
        
        pred_y_base = np.array(base_threshold <= _pred_y, dtype=int)
        pred_y_agg0 = np.array(agg_threshold0 <= _pred_y, dtype=int)
        pred_y_agg1 = np.array(agg_threshold1 <= _pred_y, dtype=int)
        pred_y_agg2 = np.array(agg_threshold2 <= _pred_y, dtype=int)

        pred_y_agg3 = np.zeros_like(_pred_y)
        i,j = np.where(_pred_y[:,:,0] < 0.987)
        k = _pred_y[i,j,1:].argmax(-1)+1
        pred_y_agg3[i,j,k] = 1
        
        # TODO: search for repetitions and find i, j for repetitions then:
        #pred_y_agg3[i,j,repetition_k] = 1
        i, j = np.load('detected_slogans.npy')
        
        
        for threshold, pred_y in [('agg3', pred_y_agg3), ('agg2', pred_y_agg2), ('agg1', pred_y_agg1), ('agg0', pred_y_agg0), ('base', pred_y_base)]:
            # article: list set of labels per word
            results = {
                _aid: [
                    (line_no, len(w), {i2label[i+1] for i, y in enumerate(ys) if y == 1})
                    for (aid, line_no), line, onehots in zip(keys, values, pred_y)
                    for w, ys in zip(line, onehots)
                    if _aid == aid
                ]
                for _aid in article_ids
            }

            # fill holes:
            for aid in results:
                results[aid] = [
                    (line_no, w_len, curr_labels)
                    for (_, _, w_labels_prev), (line_no, w_len, w_labels), (_, _, w_labels_next) in zip(
                        [(None, None, {None})]+results[aid], 
                        results[aid], 
                        results[aid]+[(None, None, {None})])
                    for curr_labels in [set(list(w_labels) + list(w_labels_prev&w_labels_next))] 
                ]

            # print the results:
            f = open(f'report_dev_flc_{model_name}_{threshold}.txt', 'w')
            raw = []
            for aid, article in results.items():
                for label in label2i:
                    if label in {'O', '<PAD>'}:
                        continue
                    counter = 0
                    for _, word_len, word_labels in article:
                        s = counter
                        counter += word_len + 1
                        e = counter
                        if label in word_labels:
                            raw.append((aid,label,s,e))

            compact = []
            buffer = (None, None, None, None)
            for aid,label,s,e in raw:
                if aid == buffer[0] and label == buffer[1] and s == buffer[3]:
                    buffer = (aid, label, buffer[2], e)
                else:
                    if buffer[0] != None:
                        compact.append(buffer)
                    buffer = (aid, label, s, e)
            compact.append(buffer)

            for aid,label,s,e in compact:
                out = ""
                # insert the rule-based results:
                # if aid in rule_based_slogan:
                #     for s, e in rule_based_slogan[aid]
                #         out += f"{aid}\tSlogans\t{s}\t{e}\n"
                out += f"{aid}\t{label}\t{s}\t{e}\n"
                #print(out)
                f.write(out)
            f.close()

            f = open(f'report_dev_slc_{model_name}_{threshold}.txt', 'w')
            for line in open('../data/datasets/dev.template-output-SLC.out'):
                aid, line_no, _ = line.split('\t')
                line_no = int(line_no)
                labels = set([
                    l
                    for _aid, article in results.items()
                    for _line_no, _, word_labels in article
                    for l in word_labels
                    if aid == _aid and _line_no == line_no
                ]) - {'O', '<PAD>'}
                label = "non-propaganda" if len(labels) == 0 else "propaganda"
                out = f"{aid}\t{line_no}\t{label}"
                f.write(out + "\n")
            f.close()


if __name__ == '__main__':
    get_data()