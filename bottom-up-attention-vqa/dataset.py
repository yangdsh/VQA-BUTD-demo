from __future__ import print_function
import os
import json
import cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset

USE_ALL = True

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer, bert):
    img_id = answer['image_id']
    answer.pop('image_id')
    answer.pop('question_id')
    if bert:
        entry = {
            'question_id' : question['question_id'],
            'image_id'    : img_id,
            'image'       : img,
            'question'    : question['question'],
            'q_token'     : question['embedding'],
            'answer'      : answer}
    else:
        entry = {
            'question_id' : question['question_id'],
            'image_id'    : img_id,
            'image'       : img,
            'question'    : question['question'],
            'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val, bert, demo):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if demo:
        question_path = '../question.json'
    elif bert:
        question_path = '../bert_data/%sOut.json' % name
    else:
        question_path = os.path.join(
            dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    if demo:
        answers = [{'question_id':0, 'image_id':42, 'labels':[], 'scores':[]}]
        print(questions[0]['question'])
    else: 
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    n_loc = 0
    n_other = 0
    n_train = 0
    if USE_ALL:
        location_words = [' ']
    else:
        location_words = [' behind ', ' front ', ' back ', ' below ', ' above ', ' under ',
                          ' top ', ' left ', ' bottom ', ' over ', 'next to', ' right ',
                          ' Where ', ' inside ', ' outside ', ' on ']
    if 'val' in name:
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            img_id = answer['image_id']
            if img_id2val.__contains__(img_id):
                if any(x in str(question['question']) for x in location_words):
                    # print(str(question['question']))
                    n_loc += 1
                    entries.append(_create_entry(img_id2val[img_id], question, answer, bert))
                else:
                    n_other += 1
    else:
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            img_id = answer['image_id']
            if img_id2val.__contains__(img_id):
                entries.append(_create_entry(img_id2val[img_id], question, answer, bert))
                n_train += 1

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', bert=False, demo=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']
        self.bert = bert

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name), 'rb'))
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, bert, demo)

        if not bert:
            self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(2)
        self.s_dim = self.spatials.size(2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            if not self.bert:
                embedding = torch.from_numpy(np.array(entry['q_token']))
            else:
                embedding = torch.from_numpy(np.array(entry['q_token'], dtype=np.float32))
            entry['q_token'] = embedding

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]

        question = entry['q_token']
        answer = entry['answer']
        target = torch.zeros(self.num_ans_candidates)

        return features, spatials, question, target

    def __len__(self):
        return len(self.entries)