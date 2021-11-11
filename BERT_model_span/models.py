import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
#from BERT_model_main import MODELS


MODELS = [
          # 12-layer, 768-hidden, 12-heads, 110M parameters
          (BertModel,       BertTokenizer,      'bert-base-uncased'),
          # 24-layer, 1024-hidden, 16-heads, 340M parameters
          (BertModel,       BertTokenizer,      'bert-large-uncased'),
          # 12-layer, 768-hidden, 12-heads, 125M parameters
          (RobertaModel,    RobertaTokenizer,   'roberta-base'),
          # 24-layer, 1024-hidden, 16-heads, 355M parameters
          (RobertaModel,    RobertaTokenizer,   'roberta-large'),
          ]

class BERT_NN(nn.Module):
    def __init__(self, args):
        super(BERT_NN, self).__init__()
        self.args = args
        model_class, tokenizer_class, pretrained_weights = MODELS[self.args.MODELS_idx]

        # https://huggingface.co/transformers/model_doc/bert.html
        self.bert_encoder = model_class.from_pretrained(
            pretrained_weights, return_dict=False)
        #self.bert_encoder = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)

        #self.bert_encoder.requires_grad_ = False

        self.drop = nn.Dropout(self.args.dropout)

        self.emb_layer = nn.Linear(
            self.args.bert_hidden_size, args.emb_layer_size)

        self.ReLU = nn.ReLU()

    # encoder_flag is a dumb flag here
    def forward(self, input_ids_tensor, mask_tensor, output_mask_tensor, key_len_tensor, encoder_flag):
        # input_ids_tensor: (batch_size, sequence_length, hidden_size)
        # select_idx: (batch_size, hidden_size)
        # return: (batch_size, hidden_size)

        last_hidden_state, pooler_output = self.bert_encoder(
            input_ids=input_ids_tensor, attention_mask=mask_tensor)

        #outputs = self.bert_encoder(input_ids=input_ids_tensor, attention_mask=mask_tensor)
        #last_hidden_state = outputs.hidden_states[-1]

        batch_size, sequence_length, hidden_size = last_hidden_state.size(
            0), last_hidden_state.size(1), last_hidden_state.size(2)

        if self.args.sentence_pooling_method != "cls" and encoder_flag == "gloss":
            if self.args.sentence_pooling_method == "max":
                emb = torch.max(last_hidden_state, 1)[0]
            else:
                emb = torch.mean(last_hidden_state, 1)
        else:
            new_output_mask_tensor = output_mask_tensor.view(
                batch_size, sequence_length, 1).expand(batch_size, sequence_length, hidden_size)
            target_emb = torch.mul(last_hidden_state, new_output_mask_tensor)
            emb = torch.div(torch.sum(target_emb, dim=1), key_len_tensor.view(
                batch_size, 1).expand(batch_size, hidden_size))

        emb = emb.view(batch_size, hidden_size)
        #emb = self.drop(emb.view(batch_size, hidden_size))

        #emb = self.emb_layer(emb)

        return emb


class BERT_NN_SEP(nn.Module):
    def __init__(self, args):
        super(BERT_NN_SEP, self).__init__()
        self.args = args
        model_class, tokenizer_class, pretrained_weights = MODELS[self.args.MODELS_idx]

        # https://huggingface.co/transformers/model_doc/bert.html
        self.bert_encoder_context = model_class.from_pretrained(
            pretrained_weights, return_dict=False)
        self.bert_encoder_gloss = model_class.from_pretrained(
            pretrained_weights, return_dict=False)

        #self.bert_encoder_context = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
        #self.bert_encoder_gloss = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)

        #self.bert_encoder.requires_grad_ = False

        self.drop = nn.Dropout(self.args.dropout)

        self.emb_layer = nn.Linear(
            self.args.bert_hidden_size, args.emb_layer_size)

        self.ReLU = nn.ReLU()

    def forward(self, input_ids_tensor, mask_tensor, select_idx, encoder_flag):
        # input_ids_tensor: (batch_size, sequence_length, hidden_size)
        # select_idx: (batch_size, hidden_size)
        # return: (batch_size, hidden_size)

        if encoder_flag == "gloss":
            last_hidden_state, pooler_output = self.bert_encoder_gloss(
                input_ids=input_ids_tensor, attention_mask=mask_tensor)
            #outputs = self.bert_encoder_gloss(input_ids=input_ids_tensor, attention_mask=mask_tensor)
            #last_hidden_state = outputs.hidden_states[-1]
        elif encoder_flag == "context":
            last_hidden_state, pooler_output = self.bert_encoder_context(
                input_ids=input_ids_tensor, attention_mask=mask_tensor)
            #outputs = self.bert_encoder_context(input_ids=input_ids_tensor, attention_mask=mask_tensor)
            #last_hidden_state = outputs.hidden_states[-1]

        batch_size, sequence_length, hidden_size = last_hidden_state.size(
            0), last_hidden_state.size(1), last_hidden_state.size(2)

        if self.args.sentence_pooling_method != "cls" and encoder_flag == "gloss":
            if self.args.sentence_pooling_method == "max":
                emb = torch.max(last_hidden_state, 1)[0]
            else:
                emb = torch.mean(last_hidden_state, 1)
        else:
            emb = torch.gather(last_hidden_state, dim=1, index=select_idx.view(
                batch_size, 1, 1).expand(batch_size, 1, hidden_size))

        emb = emb.view(batch_size, hidden_size)
        #emb = self.drop(emb.view(batch_size, hidden_size))

        #emb = self.emb_layer(emb)

        return emb
