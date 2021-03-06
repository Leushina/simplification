import math
import copy
import time
import pandas as pd
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, TabularDataset
from torchtext import data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from get_model import preprocess
from sari.SARI import SARIsent


def batch_size_fn(new, count):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    :param new:
    :param count:
    :return:
    """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.Complex))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.Simple) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def create_csv():
    data_txt = preprocess()
    data_txt = data_txt[['Complex', 'Simple']]

    train, val = train_test_split(data_txt, test_size=0.1, shuffle=False, random_state=13)
    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv", index=False)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        if device.type == 'cuda':
            x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        else:
            x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        if device.type == 'cuda':
            self.ff = FeedForward(d_model).cuda()
        else:
            self.ff = FeedForward(d_model)
        # self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)

        if device.type == 'cuda':
            self.ff = FeedForward(d_model).cuda()
        else:
            self.ff = FeedForward(d_model)
        # self.ff = FeedForward(d_model)

    def forward(self, x, e_outputs, src_mask, trg_mask):

        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
                                           src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    # we don't perform softmax on the output as this will be handled
    # automatically by our loss function
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if device.type == 'cuda':
        np_mask = np_mask.cuda()
    return np_mask


def translate(model, src, max_len=80, custom_string=True):
    model.eval()

    if custom_string:
        src = tokenize_en(src)  # .transpose(0,1)
        # sentence = Variable(torch.LongTensor([[input_text.vocab.stoi[tok] for tok in sentence]])) #.cuda()
        if device.type == 'cuda':
            src = Variable(torch.LongTensor([[input_text.vocab.stoi[tok] for tok in src]])).cuda()
        else:
            src = Variable(torch.LongTensor([[input_text.vocab.stoi[tok] for tok in src]]))
        src_mask = (src != input_text.vocab.stoi['<pad>']).unsqueeze(-2)

    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([target_text.vocab.stoi['<start>']])
    for i in range(1, max_len):

        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype("uint8")
        if device.type == 'cuda':
            trg_mask = Variable(torch.from_numpy(trg_mask) == 0).cuda()
        else:
            trg_mask = Variable(torch.from_numpy(trg_mask) == 0)
        out = model.out(model.decoder(outputs[:i].unsqueeze(0),
                                      e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == target_text.vocab.stoi['<end>']:
            break
        temp = len('<start> ')
    return ' '.join([target_text.vocab.itos[ix] for ix in outputs[:i]])[temp:]


def create_masks(src, trg):
    src_mask = (src != input_text.vocab.stoi['<pad>']).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != target_text.vocab.stoi['<pad>']).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size)
        if device.type == 'cuda':
            np_mask = np_mask.cuda()
            trg_mask = trg_mask.cuda()
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None

    return src_mask, trg_mask


def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    spacy.prefer_gpu()
en = spacy.load('en_core_web_sm')


# input_text.build_vocab(train, val)
# target_text.build_vocab(train, val)
#
# torch.save(input_text, 'input_text.p')
# torch.save(target_text, 'target_text.p')

input_text = torch.load('input_text.p')
target_text = torch.load('target_text.p')

global max_src_in_batch, max_tgt_in_batch

input_pad = input_text.vocab.stoi['<pad>']
target_pad = target_text.vocab.stoi['<pad>']

d_model = 512
heads = 4
N = 3
src_vocab = len(input_text.vocab)
trg_vocab = len(target_text.vocab)


def create_transformer():

    def sari_evaluate():
        data_txt = preprocess()
        data_txt = data_txt[['Complex', 'Simple']]

        input_train, input_val, target_train, target_val = train_test_split(list(data_txt['Complex']),
                                                                            list(data_txt['Simple']),
                                                                            test_size=0.1,
                                                                            shuffle=False,
                                                                            random_state=13)
        sentences = pd.DataFrame(list(zip(input_val, target_val)),
                                 columns=['Complex', 'References'])

        sentences = sentences.groupby(['Complex']).agg(lambda x: tuple(x)).applymap(list).reset_index()
        sents, refs = list(sentences['Complex']), list(sentences['References'])
        saris = []
        n = len(sents)
        for i, pair in enumerate(zip(sents, refs)):
            sent, ref = pair
            predicted_sent = translate(model, sent)
            saris.append(SARIsent(sent, predicted_sent, ref))
        print(max(saris))
        print(sum(saris) / n)
        return max(saris), sum(saris) / n

    def train_model(epochs, print_every=500):

        batch_size = 1300

        data_fields = [('Complex', input_text), ('Simple', target_text)]
        train, val = data.TabularDataset.splits(path='data_csv/',
                                                    train='train.csv',
                                                    validation='val.csv',
                                                    format='csv', fields=data_fields)

        train_iter = MyIterator(train, batch_size=batch_size,  # device=device,
                               repeat=False, sort_key=lambda x: (len(x.Complex), len(x.Simple)),
                               batch_size_fn=batch_size_fn, train=True,
                               shuffle=True)

        val_iter = MyIterator(val, batch_size=batch_size,  # device=device,
                              repeat=False, sort_key=lambda x: (len(x.Complex), len(x.Simple)),
                              batch_size_fn=batch_size_fn, train=True,
                              shuffle=True)

        model.train()

        start = time.time()
        temp = start

        for epoch in range(epochs):

            total_loss = 0
            epoch_loss = 0

            for i, batch in enumerate(train_iter):
                src = batch.Complex.transpose(0, 1)
                trg = batch.Simple.transpose(0, 1)
                if device.type == 'cuda':
                    src, trg = src.to('cuda'), trg.to('cuda')
                # the Target sentence we input has all words except
                # the last, as it is using each word to predict the next

                trg_input = trg[:, :-1]

                # the words we are trying to predict

                targets = trg[:, 1:].contiguous().view(-1)

                # create function to make masks using mask code above

                src_mask, trg_mask = create_masks(src, trg_input)

                preds = model(src, trg_input, src_mask, trg_mask)

                optim.zero_grad()

                loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                                       targets, ignore_index=target_pad)
                loss.backward()
                optim.step()

                total_loss += loss.data
                epoch_loss += loss.data
                if (i + 1) % print_every == 0:
                    loss_avg = total_loss / print_every
                    print(
                        "time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60,
                                                                                            epoch + 1, i + 1, loss_avg,
                                                                                            time.time() - temp,
                                                                                            print_every))
                    total_loss = 0
                    temp = time.time()

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                epoch_loss / i))
            test(epoch)
            torch.save(model.state_dict(), f'{dst}')
            print()
            temp = time.time()

            def test(epoch):

                model.eval()

                total_loss = 0

                for i, batch in enumerate(val_iter):
                    src = batch.Complex.transpose(0, 1)
                    trg = batch.Simple.transpose(0, 1)
                    if device.type == 'cuda':
                        src, trg = src.to('cuda'), trg.to('cuda')

                    # the Target sentence we input has all words except
                    # the last, as it is using each word to predict the next

                    trg_input = trg[:, :-1]

                    # the words we are trying to predict

                    targets = trg[:, 1:].contiguous().view(-1)

                    # create function to make masks using mask code above

                    src_mask, trg_mask = create_masks(src, trg_input)

                    preds = model(src, trg_input, src_mask, trg_mask)

                    loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                                           targets, ignore_index=target_pad)

                    total_loss += loss.data

                print('Test on epoch {} Loss {:.4f}'.format(epoch + 1,
                                                            total_loss / i))

    model = Transformer(src_vocab, trg_vocab, d_model, N, heads)
    if device.type == 'cuda':
        model = model.cuda()
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optim = torch.optim.Adam(model.parameters(), lr=0.000001, betas=(0.9, 0.98), eps=1e-9)

    dst = "transformer/Transformer_1"  # loss 0.3215  512 units
    if device.type == 'cuda':
        model.load_state_dict(torch.load(dst))
    else:
        model.load_state_dict(torch.load(dst, map_location=torch.device('cpu')))
    model.eval()
    # train_model(5)
    return model


if __name__ == "__main__":
    create_transformer()
