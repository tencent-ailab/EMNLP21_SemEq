import argparse
import glob
import os
import random
import sys
import torch
import numpy as np
sys.path.append("../utilities/")
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from train import model_main
from model_utils_span import BERT_prepare_data
from models import MODELS



# clean files under current directory
def clean_files(args):
    for file in glob.glob("*.txt"):
        if ("test_" in file or "valid_" in file) and (args.loss in file):
            os.system("rm " + file)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #  (OxfordAdvanced, Webster, Collins, CambridgeAdvanced, LongmanAdvanced, wordnet)
    parser.add_argument("--dict", default="OxfordAdvanced", type=str,
                        help="Which dictionary to train the model.")
    parser.add_argument("--use_dict_knowledge", default=False,
                        type=str2bool, help="Use dictionary knowledge or not.")
    parser.add_argument("--align_dict_knowledge", default=False,
                        type=str2bool, help="Align dictionary knowledge or not.")
    parser.add_argument("--align_dict_t", default=0.5, type=float,
                        help="Dictionary knowledge alignment threshold.")
    parser.add_argument("--dict_ignore_phrase", default=False, type=str2bool,
                        help="Ignore several-word phrases in dictionary knowledge.")
    parser.add_argument("--eval_dataset", default="dictionary", type=str,
                        help="Which evaluation dataset to test the model (WSD, WiC, FEWS, dictionary).")
    parser.add_argument("--loss", default="CrossEntropy", type=str,
                        help="Which loss function to use (CrossEntropy, CosineSimilarity, Contrastive, Triplet).")
    parser.add_argument("--concat", default="rep*diff*mul",
                        type=str, help="CrossEntropy embedding concatenation.")
    parser.add_argument("--sep_encoder", default=False, type=str2bool,
                        help="Use seperate encoders for example sentences and definition sentences.")
    parser.add_argument("--exp_mode", default="train", type=str,
                        help="Train a model or eval on saved model (train, eval).")
    parser.add_argument("--class_num", default=2,
                        type=int, help="Class number.")
    #  Sequences longer than this will be truncated, sequences shorter will be padded.
    parser.add_argument("--sentence_max_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--gloss_max_length", default=64, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--sentence_pooling_method", default="cls",
                        type=str, help="Sentence pooling method (cls, max, mean).")
    parser.add_argument("--max_num_gloss", default=100,
                        type=int, help="The maximum total gloss number.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--bert_model", default="bert_base",
                        type=str, help="BERT model.")
    parser.add_argument("--use_hidden_layer", default=True, type=str2bool,
                        help="Use one more hidden layer before classification or not.")
    parser.add_argument("--emb_layer_size", default=300,
                        type=int, help="Representation emb layer size.")
    parser.add_argument("--fix_embedding_layer", default=False,
                        type=str2bool, help="Fix BERT embedding layer in training.")
    parser.add_argument("--dropout", default=0.2,
                        type=float, help="Dropout rate.")
    parser.add_argument("--optimizer", default="Adam",
                        type=str, help="Adam or AdamW.")
    parser.add_argument("--learning_rate", default=2e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01,
                        type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument("--epochs", default=7, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--cuda", default=True,
                        type=str2bool, help="Train on GPU or not.")
    parser.add_argument("--gpu_id", default=0, type=int,
                        help="GPU id to train models.")
    parser.add_argument("--toy", default=False, type=str2bool,
                        help="Use toy dataset (for fast testing), True means use toy dataset")
    parser.add_argument("--prepare_data", default=True,
                        type=str2bool, help="Prepare data (env.pkl) or not.")
    parser.add_argument("--sample_neg_defs", default=False, type=str2bool,
                        help="Sample negative definitions to create training instances or not.")
    parser.add_argument("--seed", default=11, type=int,
                        help="Random seed for initialization.")
    parser.add_argument("--mask_trigger", default=True, type=str2bool,
                        help="Mask the key word or not in a sentence.")
    parser.add_argument("--scale_weight", default=True,
                        type=str2bool, help="Give smaller class higher weight.")
    parser.add_argument("--encoder1_flag", default="gloss",
                        type=str, help="Encoder to encode the first sentence.")
    parser.add_argument("--encoder2_flag", default="context",
                        type=str, help="Encoder to encode the second sentence.")

    args = parser.parse_args()

    args.dictionary_file_path = "../data/extracted_data_new/"
    args.def_alignment_file = "../run_align_definitions_main/final_keyword_pos2def_alignments.p"

    args.WSD_data_path = "../EWISE/data/"
    args.WSD_gold_data_path = "../datasets/WSD_Evaluation_Framework/"
    args.WiC_data_path = "../datasets/WiC_dataset/"
    args.FEWS_data_path = "../datasets/fews/"

    if args.bert_model == "bert_base":
        args.MODELS_idx = 0
        args.bert_hidden_size = 768
    elif args.bert_model == "bert_large":
        args.MODELS_idx = 1
        args.bert_hidden_size = 1024
    elif args.bert_model == "roberta_base":
        args.MODELS_idx = 2
        args.bert_hidden_size = 768
    elif args.bert_model == "roberta_large":
        args.MODELS_idx = 3
        args.bert_hidden_size = 1024
    else:
        print("MODEL selection error!")

    if "bert_" in args.bert_model:
        args.cls_token = "[CLS]"
        args.sep_token = "[SEP]"
        args.mask_token = "[MASK]"
        args.pad_token = "[PAD]"

    elif "roberta_" in args.bert_model:
        args.cls_token = "<s>"
        args.sep_token = "</s>"
        args.mask_token = "<mask>"
        args.pad_token = "<pad>"

    model_class, tokenizer_class, pretrained_weights = MODELS[args.MODELS_idx]
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    #args.MASK_id = tokenizer._convert_token_to_id(args.mask_token)
    args.MASK_id = tokenizer.convert_tokens_to_ids([args.mask_token])[0]
    args.PAD_id = tokenizer.convert_tokens_to_ids([args.pad_token])[0]


    if args.exp_mode == "train":
        clean_files(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda is True:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda", args.gpu_id)
        print("Using GPU device:{}".format(torch.cuda.current_device()))
    else:
        device = torch.device("cpu")
    args.device = device

    if args.prepare_data is True:
        BERT_prepare_data(args, tokenizer)

    model_main(args, tokenizer)
