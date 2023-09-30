import json
import collections
from preprocess.utils import *

TRAIN = '../dataset/RAMS_1.0/data/train.jsonlines'
DEV = '../dataset/RAMS_1.0/data/dev.jsonlines'
TEST = '../dataset/RAMS_1.0/data/test.jsonlines'

def find_position(start, end, lengths):
    offset = 0
    for i, l in enumerate(lengths):
        if offset + l > start:
            return i, start - offset, end - offset
        offset += l
    print('Cannot resolve:')
    print(start)
    print(end)
    print(lengths)
    return 0, start, end


def read_data(files):
    raw_data = []
    for file in files:
        with open(file, 'r') as f:
            raw_data += [json.loads(x) for x in f.readlines()]

    events = []
    for doc in raw_data:
        doc_id = doc['doc_key']
        sentences = doc['sentences']
        lengths = [len(x) for x in sentences]

        for trigger_id, (start, end, label) in enumerate(doc['evt_triggers']):
            sent_id, trigger_span_start, trigger_span_end = find_position(start, end, lengths)
            assert sent_id < len(sentences), 'Doclen: {}/Sent_id: {}'.format(len(sentences), sent_id)
            assert sent_id > -1, 'Sent_id: {}'.format(sent_id)

            tokens = sentences[sent_id]

            arguments = []
            for ent_start, ent_end, arg_label in doc['ent_spans']:
                arg_sent_id, arg_start, arg_end = find_position(ent_start, ent_end, lengths)
                if arg_sent_id == sent_id:
                    arg = [arg_start, arg_end, arg_label[0][0][11:]]
                    arguments.append(arg)

            event = {
                'id': '{}#{}'.format(doc_id, trigger_id),
                'token': tokens,
                'trigger': [trigger_span_start, trigger_span_end],
                'label': label[0][0],
                'argument': arguments
            }
            events.append(event)
    return events


def load_supervised():
    train = read_data([TRAIN])
    dev = read_data([DEV])
    test = read_data([TEST])
    return train, dev, test


def print_statistic(data):
    labels = [x['label3'] for x in data]
    counter = collections.Counter()
    counter.update(labels)

    print('-' * 80)
    print('#class: ', len(set(labels)))
    print('#sample:', len(labels))

    hist(labels)

    doc_sent_counts = [len(x['sentence_tokens']) for x in data]
    doc_lengths = []
    sent_lengths = []
    trigger_sen_lengths = []

    for doc in data:
        sentences = doc['sentence_tokens']
        lengths = [len(x) for x in sentences]
        doc_lengths.append(sum(lengths))
        sent_lengths += lengths
        trigger_sen_lengths.append(lengths[doc['trigger_sentence_index']])

    print('-' * 20)
    print('| Max sentence count: ', max(doc_sent_counts))
    hist(doc_sent_counts)
    print('| Max document length: ', max(doc_lengths))
    hist(doc_lengths)
    print('| Max setence length: ', max(sent_lengths))
    hist(sent_lengths)
    print('| Max trigger sentence length: ', max(trigger_sen_lengths))
    hist(trigger_sen_lengths)


def load_fsl():
    all_data = read_data([TRAIN, DEV, TEST])
    train_label_set = ['artifactexistence', 'conflict', 'contact', 'disaster', 'government', 'inspection',
                       'manufacture', 'movement']
    dev_label_set = ['justice', 'life']
    test_label_set = ['personnel', 'transaction']

    ignore_label3_set = ['conflict.attack.strangling',
                         'movement.transportperson.fall',
                         'conflict.attack.hanging',
                         'contact.negotiate.n/a',
                         'movement.transportperson.bringcarryunload']
    train = []
    dev = []
    test = []

    for item in all_data:
        label_parts = item['label'].split('.')
        if item['label'] in ignore_label3_set:
            continue
        if label_parts[0] in dev_label_set:
            dev.append(item)
        elif label_parts[0] in test_label_set:
            test.append(item)
        else:
            train.append(item)

    return train, dev, test


if __name__ == '__main__':
    # train, dev, test = load_fsl()
    #
    # save_json(train, 'datasets/rams/fsl/train.json')
    # save_json(dev, 'datasets/rams/fsl/dev.json')
    # save_json(test, 'datasets/rams/fsl/test.json')

    train, dev, test = load_supervised()
    save_json(train, '../datasets/rams/supervised/train.json')
    save_json(dev, '../datasets/rams/supervised/dev.json')
    save_json(test, '../datasets/rams/supervised/test.json')
