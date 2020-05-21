import dataset
import transformers
from transformers import LineByLineTextDataset
from transformers import RobertaConfig, RobertaTokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizerFast
from transformers import pipeline

import torch
import torch.nn as nn
import torch.utils.data.dataset as td
import torch.utils.data.dataloader as dl
from argparse import ArgumentParser
from argparse import Namespace
import yaml

if __name__ == '__main__':
    print(torch.__version__)
    training_args = TrainingArguments(
        output_dir="./bert_small",
        overwrite_output_dir=True,
        num_train_epochs=20,
        do_train=True,
        per_gpu_train_batch_size=512,
        save_steps=100,
        save_total_limit=2,
        local_rank=-1,
        no_cuda=False
    )
    print(training_args.device)
    tokenizer = RobertaTokenizerFast.from_pretrained('./bird_bpe_vocab', max_len=256)
    _config = RobertaConfig(
        vocab_size=tokenizer._tokenizer.get_vocab_size(),
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=256,
        pad_token_id=1,
        eos_token_id=0,
        bos_token_id=2,
        output_attentions=False,
        output_hidden_states=False
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.25
    )
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="language_corpus.txt",
        block_size=256,
    )
    _model = RobertaForMaskedLM(_config)

    print(training_args.n_gpu)
    trainer = Trainer(
        model=_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )
    print(_model.num_parameters())
    #trainer.train()
    #dset = dataset.TextDatasetCreator()
    #dset = dl.DataLoader(dset, num_workers=8)
    #with open('language_corpus.txt', 'w') as lang_file:
    #    for input in dset:
    #        for sentence in input:
    #            lang_file.write(sentence[0] + '\n')
    #            print('step')