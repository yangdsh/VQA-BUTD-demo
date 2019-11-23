import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size())
    if torch.cuda.is_available():
        one_hots = one_hots.cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters(), lr=2e-3)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, q, a) in enumerate(train_loader):
            if torch.cuda.is_available():
                v = v.cuda()
                b = b.cuda()
                q = q.cuda()
                a = a.cuda()

            pred = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data.item() * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def vqa(model, train_loader, eval_dset, num_ans=1):
    model.train(False)
    for i, (v, b, q, a) in enumerate(train_loader):
        if torch.cuda.is_available():
            v = v.cuda()
            b = b.cuda()
            q = q.cuda()
            a = a.cuda()

        for j in range(num_ans):
            pred = model(v, b, q, a)
            label = pred.max(1).indices.item()
            value = pred.max(1).values.item()
            # print(label, value)
            print(eval_dset.label2ans[label])
            pred[0][label] = -1000000
    model.train(True)


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    with torch.no_grad():
        for v, b, q, a in iter(dataloader):
            if torch.cuda.is_available():
                v = (v).cuda()
                b = (b).cuda()
                q = (q).cuda()
                a = (a).cuda()
            pred = model(v, b, q, None)
            batch_score = compute_score_with_logits(pred, a).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
