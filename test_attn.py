#from basic_blocks.conv_layers import MultiHeadConvolutionalAttention, MultiHeadAttention, Attention
import torch.nn as nn
import torch.nn.functional as F
import torch

if __name__ == '__main__':
    inp = torch.randn(size=(4, 256, 64, 64))
    #q_dim = inp.size()[1:]
    #print(inp.size()[0:])
    #mha = MultiHeadAttention(inp.size()[1:], inp.size()[1:], inp.size()[1:], 8, 8, inp.size()[1:])
    #cmha = MultiHeadConvolutionalAttention(256, 8, 8, 8, 64)
    #score = mha(inp, inp, inp)
    #print(score.size())
    #score = score.view(inp.size(0), -1, inp.size(2), inp.size(3))
    #print(score.size())
    #score = cmha(inp)
    #print(score.size())

    a = torch.randn(size=(4, 4096, 256)) # q - conv feats
    b = torch.randn(size=(4, 128, 786))  # k - GPT output
    d = nn.Linear(786, 256)
    b = d(b)
    c = a @ b.transpose(dim0=1, dim1=2)
    c = nn.functional.softmax(c, dim=1)
    c = c.transpose(dim0=1, dim1=2)
    c = c.view(c.size(0), c.size(1), 64, 64) # v - conv feats
    e = torch.randn(size=(4, 128, 64, 64))
    f = c @ e
    print(f.size())

    mod_dict_1 = nn.ModuleDict({'1': nn.ReLU6(), '2': nn.GLU()})
    mod_dict_2 = nn.ModuleDict({'1': nn.ELU(), '2': nn.GELU()})
    mod_dict = nn.ModuleDict({'a': mod_dict_1, 'b': mod_dict_2})
    print(mod_dict)

    a = torch.randn(size=(4,128,64,64))
    layer = nn.AdaptiveAvgPool2d(output_size=(1))
    result = layer(a)
    result = result.view(-1, 128)
    print(result)

    feats = torch.randn(size=(4,256,64,64))
    conv = nn.Conv2d(256, 512, kernel_size=3, bias=True)
    x = conv(feats)
    #print(layer(a).size())
    #score = mha(inp, inp, inp)


