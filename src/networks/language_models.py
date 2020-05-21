import transformers
import torch.nn as nn
import os.path as path
import torch


class GPT2(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()

        if cfg.tokens_pretrained:
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        else:
            self.tokenizer = transformers.GPT2Tokenizer(cfg.vocab_path, cfg.merge_path)

        if cfg.embeddings_pretrained:
            self.model = transformers.GPT2Model.from_pretrained('gpt2')
        else:
            self.model = transformers.GPT2Model('gpt2')
        self.model = self.model.to(device)

        self.pad_token = 'pad_token'
        self.device = device

        self.max_len = cfg.max_seq_len
        self.trainable = cfg.embeddings_trainable
        self.use_hidden = cfg.use_hidden_language

    def forward(self, input):
        tokens = input[0].strip('.')
        tokens = tokens.strip(',')
        tokens = tokens.split(' ')
        tokens = tokens[:self.max_len] + [self.pad_token] * (self.max_len - len(tokens))
        if not self.trainable:
            with torch.no_grad():
                token_ids = self.tokenizer.encode(tokens)
        else:
            token_ids = self.tokenizer.encode(tokens)

        token_ids = torch.tensor(token_ids).unsqueeze(0)
        token_ids = token_ids.to(self.device)

        if self.use_hidden:
            hidden, _ = self.model(token_ids)
            return hidden
        else:
            return self.model.wte(token_ids)

    def _get_emb_size(self):
        return 768

    def out_size(self):
        return self.max_len * self._get_emb_size()

class BERT(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()

        if cfg.tokens_pretrained:
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = transformers.BertTokenizer(cfg.vocab_path, cfg.merge_path)

        if cfg.embeddings_pretrained:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel('bert-base-uncased')
        self.model = self.model.to(device)

        self.pad_token = 'pad_token'
        self.device = device

        self.max_len = cfg.max_seq_len
        self.trainable = cfg.embeddings_trainable

    def forward(self, input):
        tokens = input[0].strip('.')
        #tokens = tokens.strip(',')
        #tokens = tokens.split(' ')
        if not self.trainable:
            with torch.no_grad():
                token_ids = self.tokenizer.encode(tokens, add_special_tokens=True)
        else:
            token_ids = self.tokenizer.encode(tokens, add_special_tokens=True)

        token_ids = torch.tensor(token_ids).unsqueeze(0)
        token_ids = token_ids.to(self.device)
        hidden, _ = self.model.get_input_embeddings(token_ids)
        return hidden

    def _get_emb_size(self):
        return 768

    def out_size(self):
        return self.max_len * self._get_emb_size()

