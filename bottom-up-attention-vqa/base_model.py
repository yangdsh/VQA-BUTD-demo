import torch
import torch.nn as nn
import numpy as np
from attention import Attention, NewAttention, SelfAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, q_att, v_att, q_net, v_net, classifier, bert):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_att = q_att
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.bert = bert

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        if self.bert:
            q_emb = self.q_emb(q)
        else:
            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

        v = torch.cat((v, b[:, :, 0:2]), 2)
        att = self.v_att(v, q_emb)
        with open("../attention.npy", 'wb') as f:
            np.save(f, att.cpu().detach().numpy())
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    if not dataset.bert:
        q_att = SelfAttention(q_emb.num_hid, num_hid)
        v_att = NewAttention(dataset.v_dim + 2, q_emb.num_hid, num_hid)
        q_net = FCNet([q_emb.num_hid, num_hid])
    else:
        q_att = SelfAttention(768, num_hid)
        q_emb = FCNet([768, 768])
        v_att = NewAttention(dataset.v_dim, 768, num_hid)
        q_net = FCNet([768, num_hid])
    v_net = FCNet([dataset.v_dim + 2, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, q_att, v_att, q_net, v_net, classifier, dataset.bert)
