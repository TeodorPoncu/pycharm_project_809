from transformers import pipeline
from transformers import RobertaTokenizerFast
import torch
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

def _avg_emb(tensor):
    mean = tensor.mean(dim=1)
    return mean

if __name__ == '__main__':
    tokenizer = RobertaTokenizerFast.from_pretrained('./bird_bpe_vocab', max_len=256)
    # The sun <mask>.
    # =>
    from transformers import RobertaForMaskedLM

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
    _model = RobertaForMaskedLM(_config)
    _model.load_state_dict(torch.load('bert_small/checkpoint-1100/pytorch_model.bin'))
    _model.eval()
    inp = ['this medium sized bird is all black in color and has a large head and beak in comparison with the rest of its body size.',
           'this medium sized bird is all white in color and has a large head and beak in comparison with the rest of its body size.']
    inp = tokenizer.batch_encode_plus(inp, pad_to_max_length=True, return_tensors='pt')
    print(inp)
    emb_out, pool_out = _model.roberta(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'])
    avgs = _avg_emb(emb_out)
    print(avgs.size())
    print(emb_out[0][0] == emb_out[1][0])