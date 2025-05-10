import json
from collections import Counter

import jieba
import numpy as np
from itertools import chain
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
import re

# 计算生成句子的bleu，distinct，rouge
def get_rouge(ground_truth, response):
    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": []
    }

    for g, r in zip(ground_truth, response):
        target = list(jieba.cut(g))
        pred = list(jieba.cut(r))

        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred), ' '.join(target))
        result = scores[0]
        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))

    return score_dict

def get_bleu_ave(pred, target):
    turns = len(target)
    b1_s, b2_s, b3_s, b4_s = 0, 0, 0, 0
    for index in range(turns):
        pred_utt = pred[index]
        target_utt = target[index]
        min_len = min(len(pred_utt), len(target_utt))
        if min_len >= 4:
            bleu_1 = sentence_bleu([target_utt], pred_utt, weights=[1, 0, 0, 0],
                                   smoothing_function=SmoothingFunction().method7)
            bleu_2 = sentence_bleu([target_utt], pred_utt, weights=[0.5, 0.5, 0, 0],
                                   smoothing_function=SmoothingFunction().method7)
            bleu_3 = sentence_bleu([target_utt], pred_utt, weights=[1 / 3, 1 / 3, 1 / 3, 0],
                                   smoothing_function=SmoothingFunction().method7)
            bleu_4 = sentence_bleu([target_utt], pred_utt, weights=[0.25, 0.25, 0.25, 0.25],
                                   smoothing_function=SmoothingFunction().method7)
        else:
            bleu_1, bleu_2, bleu_3, bleu_4 = 0, 0, 0, 0

        b1_s += bleu_1
        b2_s += bleu_2
        b3_s += bleu_3
        b4_s += bleu_4

    bleu_scores = {
        "bleu-1": b1_s / turns,
        "bleu-2": b2_s / turns,
        "bleu-3": b3_s / turns,
        "bleu-4": b4_s / turns
    }
    return bleu_scores

def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence

def ngrams(sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None):
    sequence = pad_sequence(sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def get_distinct(sentences, n):
    dist_score = 0
    for sentence in sentences:
        if len(sentence) == 0:
            return 0.0  # Prevent a zero division
        distinct_ngrams = set(ngrams(sentence, n))
        _dist = len(distinct_ngrams) / len(sentence)
        dist_score += _dist
    return dist_score / len(sentences)

def cal_gen(ground_truth, response):
    assert len(ground_truth) == len(response)
    bleu_scores = get_bleu_ave(response, ground_truth)
    for k, v in bleu_scores.items():
        print(k + ':{:.2f}'.format(v * 100))

    rouge_scores = get_rouge(ground_truth, response)
    for k, v in rouge_scores.items():
        print(k + ':{:.2f}'.format(v))

    dist_1 = get_distinct(response, 1)
    print('Dist-1:{:.2f}'.format(dist_1 * 100))
    dist_2 = get_distinct(response, 2)
    print('Dist-2:{:.2f}'.format(dist_2 * 100))


# 计算推断实体的P，R，F1

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+�,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _f1_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    p, r, f1 = max(p for p, _, _ in scores), max(r for _, r, _ in scores), max(f1 for _, _, f1 in scores)
    return p, r, f1


def f1_metric(hypothesis, references):
    '''
    calculate f1 metric
    :param hypothesis: list of str
    :param references: list of str
    :return:
    '''
    f1 = []
    p = []
    r = []
    for hyp, ref in zip(hypothesis, references):
        _p, _r, _f1 = _f1_score(hyp, [ref])
        p.append(_p)
        r.append(_r)
        f1.append(_f1)
    return np.mean(p), np.mean(r), np.mean(f1)

def cal_F1(path):
    preds = []
    target = []
    with open(path,'r',encoding='utf-8') as fo:
        entities = fo.readlines()
        for line in entities:
            _pred = line.strip().split('|||')[0]
            _tgt = line.strip().split('|||')[1]
            # _pred = normalize_answer(_pred)
            # _tgt = normalize_answer(_tgt)
            print(_pred, _tgt)
            preds.append(_pred)
            target.append(_tgt)

    p,r,f1 = f1_metric(preds, target)
    print(p,r,f1)

def compute_metrics(preds, labels):
    score_dict = {
        # "rouge-1": [],
        # "rouge-2": [],
        # "rouge-l": [],
        "bleu-1": [],
        "bleu-2": [],
        "bleu-3": [],
        "bleu-4": [],
        "bleu-avg": []
    }
    for pred, label in zip(preds, labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))

        hypothesis = ['None'] if hypothesis == [] else hypothesis

        # rouge = Rouge()
        # scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
        # result = scores[0]
        # for k, v in result.items():
        #     score_dict[k].append(round(v["f"] * 100, 4))

        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method7)
        bleu_1 = sentence_bleu([list(label)], list(pred), weights=[1, 0, 0, 0], smoothing_function=SmoothingFunction().method7)
        bleu_2 = sentence_bleu([list(label)], list(pred), weights=[1/2, 1/2, 0, 0], smoothing_function=SmoothingFunction().method7)
        bleu_3 = sentence_bleu([list(label)], list(pred), weights=[1/3, 1/3, 1/3, 0], smoothing_function=SmoothingFunction().method7)
        bleu_4 = sentence_bleu([list(label)], list(pred), weights=[1/4, 1/4, 1/4, 1/4], smoothing_function=SmoothingFunction().method7)
        score_dict["bleu-1"].append(round(bleu_1 * 100, 4))
        score_dict["bleu-2"].append(round(bleu_2 * 100, 4))
        score_dict["bleu-3"].append(round(bleu_3 * 100, 4))
        score_dict["bleu-4"].append(round(bleu_4 * 100, 4))
        score_dict["bleu-avg"].append(round(bleu_score * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = round(float(np.mean(v)),2)
    print(score_dict),
    return score_dict


if __name__ == "__main__":
    ground_truth = []
    response = []
    with open('multiple_dialogue/xiaoron/meddialog/linear/meddialog_generated_predictions.json', "r", encoding="utf-8") as f:
        outputs = f.readlines()
        for d in outputs:
            d = eval(d)
            ground_truth.append(d['labels'].strip())
            response.append(d['predict'].strip())
    cal_gen(ground_truth, response)
    compute_metrics(response, ground_truth)
    # cal_F1('data/preds.txt')

