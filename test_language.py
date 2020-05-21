import os
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torch.nn as nn
import torch
import yaml
from argparse import Namespace
import dataset
from torch.utils.data import DataLoader


def parse_list(root='text'):
    inputs = []
    for name in os.listdir(root):
        if os.path.isdir(os.path.join(root, name)):
            for file in os.listdir(os.path.join(root, name)):
                path = os.path.join(root, name, file.split(sep='.')[0]) + '.txt'
                inputs.append(path)
    return inputs

if __name__ == '__main__':
    rand_tensor = torch.zeros(size=(4, 512, 8, 8))
    c_bias = torch.ones(size=(512,))
    eval = rand_tensor + c_bias.view(1, -1, 1, 1)
    print(eval)
    #enc = tokenizer.encode("this bird's coloration is varying shades of gray and has dark primaries, a dark crown and a long, slender bill.")
    #v_size = tokenizer.get_vocab_size()

    #wte = nn.Embedding(tokenizer.get_vocab_size(), embedding_dim=256, padding_idx=1).cuda()
    #model = nn.LSTM(256, 1024, 4, batch_first=True, bidirectional=True).cuda()
    #print(model)

    #ids = torch.tensor(enc.ids).unsqueeze(0).cuda()
    #emb = wte(ids).cuda()
    #emb = emb.repeat(2, 1, 1)
    #state = torch.zeros(8, 2, 1024).cuda()
    #cell = torch.zeros(8, 2, 1024).cuda()
    #print(emb.size())
    #out, (_, hid) = model(emb, hx=(cell, state))
    #hid = hid.view(4, 2, hid.size(1), hid.size(2))
    #hid = hid[-1]
    #hid = torch.cat([hid[0], hid[1]], dim=1)
    #print(hid.size(), out.size())

