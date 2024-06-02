import sys
import collections
import os
import regex as re
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as t_func
import torch.optim as optim
import random
import unicodedata
import numpy as np
import argparse
import tokenizer
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel, AutoTokenizer, BertTokenizer

default_config = argparse.Namespace(
    seed=871253,
    lang='ru',
    flavor=None,
    max_length=256,
    batch_size=16,
    updates=50000,
    period=1000,
    lr=1e-5,
    dab_rate=0.1,
    device='cuda',
    debug=False
)

default_flavors = {
    'ru': 'DeepPavlov/rubert-base-cased'
}


class Config(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in default_config.__dict__.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert self.lang in ['ru']

        if 'lang' in kwargs and ('flavor' not in kwargs or kwargs['flavor'] is None):
            self.flavor = default_flavors[self.lang]


def init_random(seed):
    # make sure everything is deterministic
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# NOTE: it is assumed in the implementation that y[:,0] is the punctuation label, and y[:,1] is the case label!

punctuation = {
    'O': 0,
    'COMMA': 1,
    'PERIOD': 2,
    'QUESTION': 3,
    'EXCLAMATION': 4,
}

punctuation_syms = ['', ',', '.', ' ?', ' !']

case = {
    'LOWER': 0,
    'UPPER': 1,
    'CAPITALIZE': 2,
    'OTHER': 3,
}


class Model(nn.Module):
    def __init__(self, flavor, device):
        super().__init__()
        self.bert = AutoModel.from_pretrained(flavor)
        # need a proper way of determining representation size
        size = self.bert.dim if hasattr(self.bert, 'dim') else self.bert.config.pooler_fc_size if hasattr(
            self.bert.config, 'pooler_fc_size') else self.bert.config.emb_dim if hasattr(self.bert.config,
                                                                                         'emb_dim') else self.bert.config.hidden_size
        self.punc = nn.Linear(size, 5)
        self.case = nn.Linear(size, 4)
        self.dropout = nn.Dropout(0.3)
        self.to(device)

    def forward(self, x):
        output = self.bert(x)
        representations = self.dropout(t_func.gelu(output['last_hidden_state']))
        punc = self.punc(representations)
        case = self.case(representations)
        return punc, case


# randomly create sequences that align to punctuation boundaries
def drop_at_boundaries(rate, x, y, cls_token_id, sep_token_id, pad_token_id):
    for i, dropped in enumerate(torch.rand((len(x),)) < rate):
        if dropped:
            # select all indices that are sentence endings
            indices = (y[i, :, 0] > 1).nonzero(as_tuple=True)[0]
            if len(indices) < 2:
                continue
            start = indices[0] + 1
            end = indices[random.randint(1, len(indices) - 1)] + 1
            length = end - start
            if length + 2 > len(x[i]):
                continue
            x[i, 0] = cls_token_id
            x[i, 1: length + 1] = x[i, start: end].clone()
            x[i, length + 1] = sep_token_id
            x[i, length + 2:] = pad_token_id
            y[i, 0] = 0
            y[i, 1: length + 1] = y[i, start: end].clone()
            y[i, length + 1:] = 0


def compute_performance(config, model, loader):
    device = config.device
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = all_correct1 = all_correct2 = num_loss = num_perf = 0
    num_ref = collections.defaultdict(float)
    num_hyp = collections.defaultdict(float)
    num_correct = collections.defaultdict(float)
    for x, y in loader:
        x = x.long().to(device)
        y = y.long().to(device)
        y1 = y[:, :, 0]
        y2 = y[:, :, 1]
        with torch.no_grad():
            y_scores1, y_scores2 = model(x.to(device))
            loss1 = criterion(y_scores1.view(y1.size(0) * y1.size(1), -1), y1.view(y1.size(0) * y1.size(1)))
            loss2 = criterion(y_scores2.view(y2.size(0) * y2.size(1), -1), y2.view(y2.size(0) * y2.size(1)))
            loss = loss1 + loss2
            y_pred1 = torch.max(y_scores1, 2)[1]
            y_pred2 = torch.max(y_scores2, 2)[1]
            for label in range(1, 5):
                ref = (y1 == label)
                hyp = (y_pred1 == label)
                correct = (ref * hyp == 1)
                num_ref[label] += ref.sum()
                num_hyp[label] += hyp.sum()
                num_correct[label] += correct.sum()
                num_ref[0] += ref.sum()
                num_hyp[0] += hyp.sum()
                num_correct[0] += correct.sum()
            all_correct1 += (y_pred1 == y1).sum()
            all_correct2 += (y_pred2 == y2).sum()
            total_loss += loss.item()
            num_loss += len(y)
            num_perf += len(y) * config.max_length
    recall = {}
    precision = {}
    fscore = {}
    for label in range(0, 5):
        recall[label] = num_correct[label] / num_ref[label] if num_ref[label] > 0 else 0
        precision[label] = num_correct[label] / num_hyp[label] if num_hyp[label] > 0 else 0
        fscore[label] = (2 * recall[label] * precision[label] / (recall[label] + precision[label])).item() if recall[
                                                                                                                  label] + \
                                                                                                              precision[
                                                                                                                  label] > 0 else 0
    return total_loss / num_loss, all_correct2.item() / num_perf, all_correct1.item() / num_perf, fscore


def fit(config, model, checkpoint_path, train_loader, valid_loader, iterations, valid_period=200, lr=1e-5):
    device = config.device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=lr)
    iteration = 0
    while True:
        model.train()
        total_loss = num = 0
        for x, y in tqdm(train_loader):
            x = x.long().to(device)
            y = y.long().to(device)
            drop_at_boundaries(config.dab_rate, x, y, config.cls_token_id, config.sep_token_id, config.pad_token_id)
            y1 = y[:, :, 0]
            y2 = y[:, :, 1]
            optimizer.zero_grad()
            y_scores1, y_scores2 = model(x)
            loss1 = criterion(y_scores1.view(y1.size(0) * y1.size(1), -1), y1.view(y1.size(0) * y1.size(1)))
            loss2 = criterion(y_scores2.view(y2.size(0) * y2.size(1), -1), y2.view(y2.size(0) * y2.size(1)))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num += len(y)
            if iteration % valid_period == valid_period - 1:
                train_loss = total_loss / num
                valid_loss, valid_accuracy_case, valid_accuracy_punc, valid_fscore = compute_performance(config, model,
                                                                                                         valid_loader)
                torch.save({
                    'iteration': iteration + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'valid_accuracy_case': valid_accuracy_case,
                    'valid_accuracy_punc': valid_accuracy_punc,
                    'valid_fscore': valid_fscore,
                    'config': config.__dict__,
                }, '%s.%d' % (checkpoint_path, iteration + 1))
                print(iteration + 1, train_loss, valid_loss, valid_accuracy_case, valid_accuracy_punc, valid_fscore)
                total_loss = num = 0

            iteration += 1
            if iteration > iterations:
                return

            sys.stderr.flush()
            sys.stdout.flush()


def batchify(max_length, x, y):
    print(x.shape)
    print(y.shape)
    x = x[:(len(x) // max_length) * max_length].reshape(-1, max_length)
    y = y[:(len(y) // max_length) * max_length, :].reshape(-1, max_length, 2)
    return x, y


def train(config, train_x_fn, train_y_fn, valid_x_fn, valid_y_fn, checkpoint_path):
    X_train, Y_train = batchify(config.max_length, torch.load(train_x_fn), torch.load(train_y_fn))
    X_valid, Y_valid = batchify(config.max_length, torch.load(valid_x_fn), torch.load(valid_y_fn))

    train_set = TensorDataset(X_train, Y_train)
    valid_set = TensorDataset(X_valid, Y_valid)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size)

    model = Model(config.flavor, config.device)

    fit(config, model, checkpoint_path, train_loader, valid_loader, config.updates, config.period, config.lr)



def recase(token, label):
    if label == case['LOWER']:
        return token.lower()
    elif label == case['CAPITALIZE']:
        return token.lower().capitalize()
    elif label == case['UPPER']:
        return token.upper()
    else:
        return token


class CasePuncPredictor:
    def __init__(self, checkpoint_path, lang=default_config.lang, flavor=default_config.flavor,
                 device=default_config.device):
        loaded = torch.load(checkpoint_path, map_location=device if torch.cuda.is_available() else 'cpu')
        if 'config' in loaded:
            self.config = Config(**loaded['config'])
        else:
            self.config = Config(lang=lang, flavor=flavor, device=device)
        init(self.config)

        self.model = Model(self.config.flavor, self.config.device)
        self.model.load_state_dict(loaded['model_state_dict'])
        self.model.eval()
        self.model.to(self.config.device)

        self.rev_case = {b: a for a, b in case.items()}
        self.rev_punc = {b: a for a, b in punctuation.items()}

    def tokenize(self, text):
        return [self.config.cls_token] + self.config.tokenizer.tokenize(text) + [self.config.sep_token]

    def predict(self, tokens, getter=lambda x: x):
        max_length = self.config.max_length
        device = self.config.device
        if type(tokens) == str:
            tokens = self.tokenize(tokens)
        previous_label = punctuation['PERIOD']
        for start in range(0, len(tokens), max_length):
            instance = tokens[start: start + max_length]
            if type(getter(instance[0])) == str:
                ids = self.config.tokenizer.convert_tokens_to_ids(getter(token) for token in instance)
            else:
                ids = [getter(token) for token in instance]
            if len(ids) < max_length:
                ids += [0] * (max_length - len(ids))
            x = torch.tensor([ids]).long().to(device)
            y_scores1, y_scores2 = self.model(x)
            y_pred1 = torch.max(y_scores1, 2)[1]
            y_pred2 = torch.max(y_scores2, 2)[1]
            for i, id, token, punc_label, case_label in zip(range(len(instance)), ids, instance,
                                                            y_pred1[0].tolist()[:len(instance)],
                                                            y_pred2[0].tolist()[:len(instance)]):
                if id == self.config.cls_token_id or id == self.config.sep_token_id:
                    continue
                if previous_label != None and previous_label > 1:
                    if case_label in [case['LOWER'], case['OTHER']]:  # LOWER, OTHER
                        case_label = case['CAPITALIZE']
                if i + start == len(tokens) - 2 and punc_label == punctuation['O']:
                    punc_label = punctuation['PERIOD']
                yield (token, self.rev_case[case_label], self.rev_punc[punc_label])
                previous_label = punc_label

    def map_case_label(self, token, case_label):
        if token.endswith('</w>'):
            token = token[:-4]
        if token.startswith('##'):
            token = token[2:]
        return recase(token, case[case_label])

    def map_punc_label(self, token, punc_label):
        if token.endswith('</w>'):
            token = token[:-4]
        if token.startswith('##'):
            token = token[2:]
        return token + punctuation_syms[punctuation[punc_label]]




mapped_punctuation = {
    '.': 'PERIOD',
    '...': 'PERIOD',
    ',': 'COMMA',
    ';': 'COMMA',
    ':': 'COMMA',
    '(': 'COMMA',
    ')': 'COMMA',
    '?': 'QUESTION',
    '!': 'EXCLAMATION',
    '，': 'COMMA',
    '！': 'EXCLAMATION',
    '？': 'QUESTION',
    '；': 'COMMA',
    '：': 'COMMA',
    '（': 'COMMA',
    '(': 'COMMA',
    '）': 'COMMA',
    '［': 'COMMA',
    '］': 'COMMA',
    '【': 'COMMA',
    '】': 'COMMA',
    '└': 'COMMA',
    '└ ': 'COMMA',
    '_': 'O',
    '。': 'PERIOD',
    '、': 'COMMA',  # enumeration comma
    '、': 'COMMA',
    '…': 'PERIOD',
    '—': 'COMMA',
    '「': 'COMMA',
    '」': 'COMMA',
    '．': 'PERIOD',
    '《': 'O',
    '》': 'O',
    '，': 'COMMA',
    '“': 'O',
    '”': 'O',
    '"': 'O',
    '-': 'O',
    '-': 'O',
    '〉': 'COMMA',
    '〈': 'COMMA',
    '↑': 'O',
    '〔': 'COMMA',
    '〕': 'COMMA',
}



class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100, keep_case=True):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.keep_case = keep_case

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in text.strip().split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    # optionaly lowercase substring before checking for inclusion in vocab
                    if (self.keep_case and substr.lower() in self.vocab) or (substr in self.vocab):
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


# modification of XLM bpe tokenizer for keeping case information when vocab is lowercase
# forked from https://github.com/huggingface/transformers/blob/cd56f3fe7eae4a53a9880e3f5e8f91877a78271c/src/transformers/models/xlm/tokenization_xlm.py
def bpe(self, token):
    def to_lower(pair):
        # print('  ',pair)
        return (pair[0].lower(), pair[1].lower())

    from transformers.models.xlm.tokenization_xlm import get_pairs

    word = tuple(token[:-1]) + (token[-1] + "</w>",)
    if token in self.cache:
        return self.cache[token]
    pairs = get_pairs(word)

    if not pairs:
        return token + "</w>"

    while True:
        bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(to_lower(pair), float("inf")))
        # print(bigram)
        if to_lower(bigram) not in self.bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                new_word.extend(word[i:])
                break
            else:
                new_word.extend(word[i:j])
                i = j

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    word = " ".join(word)
    if word == "\n  </w>":
        word = "\n</w>"
    self.cache[token] = word
    return word


def init(config):
    init_random(config.seed)

    if config.lang == 'fr':
        config.tokenizer = tokenizer = AutoTokenizer.from_pretrained(config.flavor, do_lower_case=False)

        from transformers.models.xlm.tokenization_xlm import XLMTokenizer
        assert isinstance(tokenizer, XLMTokenizer)

        # monkey patch XLM tokenizer
        import types
        tokenizer.bpe = types.MethodType(bpe, tokenizer)
    else:
        # warning: needs to be BertTokenizer for monkey patching to work
        config.tokenizer = tokenizer = BertTokenizer.from_pretrained(config.flavor, do_lower_case=False)

        # warning: monkey patch tokenizer to keep case information 
        # from recasing_tokenizer import WordpieceTokenizer
        config.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token)

    if config.lang == 'fr':
        config.pad_token_id = tokenizer.pad_token_id
        config.cls_token_id = tokenizer.bos_token_id
        config.cls_token = tokenizer.bos_token
        config.sep_token_id = tokenizer.sep_token_id
        config.sep_token = tokenizer.sep_token
    else:
        config.pad_token_id = tokenizer.pad_token_id
        config.cls_token_id = tokenizer.cls_token_id
        config.cls_token = tokenizer.cls_token
        config.sep_token_id = tokenizer.sep_token_id
        config.sep_token = tokenizer.sep_token

    if not torch.cuda.is_available() and config.device == 'cuda':
        print('WARNING: reverting to cpu as cuda is not available', file=sys.stderr)
    config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

