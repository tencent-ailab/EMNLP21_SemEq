import argparse
import ast
import copy
import glob
import json
import math
import os
import pickle
import random
import re
import string
import sys
import time
from multiprocessing import Process
import nltk
import numpy as np
import spacy
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm

from utilities import (clean_text, extract_valid_POS, load_dictionary,
                       pos_mapping)

nlp = spacy.load('en_core_web_sm', disable=["parser", "ner", "textcat"])

letters = string.ascii_lowercase


def randomString(stringLength=8):
    return ''.join(random.choice(letters) for i in range(stringLength))


WORDNET_PATH = '../datasets/nltk_data/corpora/wordnet'

# download wordnet: import nltk; nltk.download("wordnet") in readme.txt
wordnet_lemmatizer = WordNetLemmatizer()
wn = WordNetCorpusReader(WORDNET_PATH, '.*')
print('wordnet version %s: %s' % (wn.get_version(), WORDNET_PATH))


def mask_keyword(args, keyword, sentence_str, keyword_not_found_output):
    pattern = []
    for token in nlp(keyword):
        pattern.append(token.lemma_)
    sentence = nlp(sentence_str)
    sentence_lemmas = []
    sentence_words = []
    for token in sentence:
        sentence_lemmas.append(token.lemma_)
        sentence_words.append(token.text)

    masked_sentence = None
    for i in range(0, len(sentence_lemmas)):
        if sentence_lemmas[i] == pattern[0] and sentence_lemmas[i:i+len(pattern)] == pattern:
            masked_sentence = " ".join(
                sentence_words[:i] + [args.mask_token] + sentence_words[i+len(pattern):])
            found_flag = True
            break
    original_sentence = " ".join(sentence_words)
    if masked_sentence is None:
        keyword_not_found_output.write(
            " ".join([keyword, "|", sentence_str, "\n"]))
    return original_sentence, masked_sentence


def normalize_sentence(args, s):
    sentence_wordList = s.split()

    if "." not in sentence_wordList[-1] and "!" not in sentence_wordList[-1] and "?" not in sentence_wordList[-1]:
        sentence_wordList.append(".")

    if "bert_" in args.bert_model:
        return "[CLS] " + " ".join(sentence_wordList) + " [SEP]"
    elif "roberta_" in args.bert_model:
        return "<s>" + " ".join(sentence_wordList) + "</s>"


def tokenize_sentence(args, tokenizer, original_s, masked_s):

    if masked_s is not None:
        original_tokens = tokenizer.tokenize(original_s)
        masked_tokens = tokenizer.tokenize(masked_s)

        original_s_ids = tokenizer.convert_tokens_to_ids(original_tokens)
        masked_s_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

        mask_idx = None
        for i, t_id in enumerate(masked_s_ids):
            if t_id == args.MASK_id:
                mask_idx = i
                break
        start_idx = mask_idx
        end_idx = mask_idx + 1
        for i in range(mask_idx+1, len(original_s_ids)):
            if original_s_ids[i] == masked_s_ids[mask_idx+1]:
                end_idx = i
                break

        return original_s_ids, start_idx, end_idx
    else:
        original_tokens = tokenizer.tokenize(original_s)
        original_s_ids = tokenizer.convert_tokens_to_ids(original_tokens)
        return original_s_ids, 0, 1


# test_files = ['senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL', 'semeval2007']
def process_WSD_evaluation_dataset(args, train_files, dev_files, test_files, tokenizer):
    wn_dict = pickle.load(open(args.WSD_data_path + 'candidatesWN30.p', 'rb'))
    pos_dict = {'NOUN': 'n', 'PROPN': 'n', 'VERB': 'v',
                'AUX': 'v', 'ADJ': 'a', 'ADV': 'r'}

    all_synsets = []
    for x in wn_dict:
        all_synsets += wn_dict[x]
    all_synsets = list(set(all_synsets))

    synset2def_text = {}
    for synset in tqdm(all_synsets, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
        ss = wn.lemma_from_key(synset).synset()
        def_text = ss.definition()
        D_sentence_input_ids, start_idx, end_idx = tokenize_sentence(
            args, tokenizer, normalize_sentence(args, def_text), None)
        synset2def_text[synset] = [
            def_text, D_sentence_input_ids, start_idx, end_idx]

    instance_id2gold_synsets = {}
    for file in train_files:
        input_lines = open(args.WSD_gold_data_path +
                           "Training_Corpora/SemCor/" + file + ".gold.key.txt")
        for line in input_lines:
            fields = line.split()
            instance_id2gold_synsets[file + "@" + fields[0]] = fields[1:]
        input_lines.close()

    for files in [dev_files, test_files]:
        for file in files:
            input_lines = open(
                args.WSD_gold_data_path + "Evaluation_Datasets/" + file + "/" + file + ".gold.key.txt")
            for line in input_lines:
                fields = line.split()
                instance_id2gold_synsets[file + "@" + fields[0]] = fields[1:]
            input_lines.close()

    instanceList = []
    error_sentence_set = set()
    for files in [train_files, dev_files, test_files]:
        for file in files:
            data = json.load(
                open(args.WSD_data_path + file + "_unindexed.json"))
            count = 0
            for sents in tqdm(data, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
                x = sents['original'].strip().split()
                y = sents['annotated'].strip().split()
                count += 1
                # if count > 1000:
                #    break
                for i in range(0, len(sents['offsets'])):
                    offset_temp = sents['offsets'][i]
                    stem_temp = sents['stems'][i]
                    pos_temp = sents['pos'][i]
                    instance_id = sents['doc_offset'] + ".t{0:0=3d}".format(i)
                    E_sentence_org_str = " ".join(
                        x[:offset_temp] + x[offset_temp].split("_") + x[offset_temp+1:])
                    E_sentence_masked_str = " ".join(
                        x[:offset_temp] + [args.mask_token] + x[offset_temp+1:])

                    org_S = normalize_sentence(args, E_sentence_org_str)
                    masked_S = normalize_sentence(args, E_sentence_masked_str)
                    E_sentence_input_ids, E_start_idx, E_end_idx = tokenize_sentence(args, tokenizer, org_S, masked_S)

                    gold_synsets = instance_id2gold_synsets[file + "@" + instance_id]
                    assert y[offset_temp] in gold_synsets

                    if (stem_temp, pos_dict[pos_temp]) in wn_dict:
                        all_synsets = wn_dict[(stem_temp, pos_dict[pos_temp])]
                    else:
                        all_synsets = []
                    """
                    # select a maximum of n=6 context gloss pairs
                    # Paper: Adapting BERT for WSD with Gloss Selection Objective and Example Sentences
                    if file == "semcor":
                        all_synsets = all_synsets[:6]
                    """

                    for synset in all_synsets:
                        def_text, D_sentence_input_ids, _, _ = synset2def_text[synset]

                        new_instance = {"instance_id": "WSD@"+file+"@" +
                                        instance_id+"@"+synset, "keyword": stem_temp}
                        new_instance["input1_text"] = def_text
                        new_instance["input1_masked_text"] = "none"
                        new_instance["input1_pos"] = pos_temp
                        new_instance["input1_ids"] = D_sentence_input_ids
                        new_instance["query1_idx"] = [0, 1]

                        # if E_keyword_idx is None:
                        #    print(sents)

                        new_instance["input2_text"] = sents['original']
                        new_instance["input2_masked_text"] = E_sentence_masked_str
                        new_instance["input2_pos"] = pos_temp
                        new_instance["input2_ids"] = E_sentence_input_ids
                        new_instance["query2_idx"] = [E_start_idx, E_end_idx]

                        if synset in gold_synsets:
                            new_instance["class"] = 1
                        else:
                            new_instance["class"] = 0

                        if (E_start_idx < args.sentence_max_length
                            and E_end_idx < args.sentence_max_length
                            and len(D_sentence_input_ids) < 512
                            and len(E_sentence_input_ids) < 512):
                            instanceList.append(new_instance)
                        else:
                            error_sentence_set.add(sents['original'])

    print("len(error_sentence_set):", len(error_sentence_set))
    trainList, devList, testList = [], [], []
    for instance in instanceList:
        if instance["instance_id"].split("@")[1] in train_files:
            trainList.append(instance)
        elif instance["instance_id"].split("@")[1] in dev_files:
            devList.append(instance)
        elif instance["instance_id"].split("@")[1] in test_files:
            testList.append(instance)
    return trainList, devList, testList

# https://github.com/allenai/kb
# https://github.com/llightts/CSI5138_Project/blob/master/RoBERTa_WiC_baseline.ipynb


def process_WiC_dataset(args, tokenizer):
    instanceList = []
    for data_flag in ["train", "dev", "test"]:
        instance_lines = open(args.WiC_data_path + data_flag + "/" +
                              data_flag + ".data.txt", 'r', encoding="utf-8").readlines()
        if data_flag == "test":
            label_lines = ['F' for i in range(len(instance_lines))]
        else:
            label_lines = open(args.WiC_data_path + data_flag + "/" +
                               data_flag + ".gold.txt", 'r', encoding="utf-8").readlines()

        for i, instance in enumerate(instance_lines):
            if not instance.strip():
                continue
            fields = instance.strip().split("\t")
            keyword = fields[0]
            pos = fields[1]
            idx1, idx2 = fields[2].split('-')
            idx1, idx2 = int(idx1), int(idx2)

            sentence1_words = fields[3].split()
            sentence2_words = fields[4].split()

            sentence1_masked_str = " ".join(
                sentence1_words[:idx1] + [args.mask_token] + sentence1_words[idx1+1:])
            sentence2_masked_str = " ".join(
                sentence2_words[:idx2] + [args.mask_token] + sentence2_words[idx2+1:])

            sentence1_input_ids, s1_keyword_start_idx, s1_keyword_end_idx = tokenize_sentence(
                args, tokenizer, normalize_sentence(args, fields[3]), normalize_sentence(args, sentence1_masked_str))
            sentence2_input_ids, s2_keyword_start_idx, s2_keyword_end_idx = tokenize_sentence(
                args, tokenizer, normalize_sentence(args, fields[4]), normalize_sentence(args, sentence2_masked_str))

            # left <-> right
            new_instance = {"instance_id": "WiC@" +
                            data_flag+"@"+str(i)+"_0", "keyword": keyword}

            new_instance["input1_text"] = fields[3]
            new_instance["input1_masked_text"] = sentence1_masked_str
            new_instance["input1_pos"] = pos
            new_instance["input1_ids"] = sentence1_input_ids
            #new_instance["input1_ids"] = sentence1_masked_input_ids
            new_instance["query1_idx"] = [
                s1_keyword_start_idx, s1_keyword_end_idx]

            new_instance["input2_text"] = fields[4]
            new_instance["input2_masked_text"] = sentence2_masked_str
            new_instance["input2_pos"] = pos
            new_instance["input2_ids"] = sentence2_input_ids
            #new_instance["input2_ids"] = sentence2_masked_input_ids
            new_instance["query2_idx"] = [
                s2_keyword_start_idx, s2_keyword_end_idx]

            if label_lines[i].strip() == "T":
                new_instance["class"] = 1
            else:
                new_instance["class"] = 0
            instanceList.append(new_instance)

            # right <-> left
            new_instance = {"instance_id": "WiC@" +
                            data_flag+"@"+str(i)+"_1", "keyword": keyword}

            new_instance["input2_text"] = fields[3]
            new_instance["input2_masked_text"] = sentence1_masked_str
            new_instance["input2_pos"] = pos
            new_instance["input2_ids"] = sentence1_input_ids
            new_instance["query2_idx"] = [
                s1_keyword_start_idx, s1_keyword_end_idx]

            new_instance["input1_text"] = fields[4]
            new_instance["input1_masked_text"] = sentence2_masked_str
            new_instance["input1_pos"] = pos
            new_instance["input1_ids"] = sentence2_input_ids
            new_instance["query1_idx"] = [
                s2_keyword_start_idx, s2_keyword_end_idx]

            if label_lines[i].strip() == "T":
                new_instance["class"] = 1
            else:
                new_instance["class"] = 0
            instanceList.append(new_instance)

    trainList, devList, testList = [], [], []
    for instance in instanceList:
        if "train" in instance["instance_id"]:
            trainList.append(instance)
        elif "dev" in instance["instance_id"]:
            devList.append(instance)
        elif "test" in instance["instance_id"]:
            testList.append(instance)
    return trainList, devList, testList


# FEWS dataset: load senses into dict from senses.txt file
def FEWS_load_senses(filepath):
    senses = {}
    with open(filepath, 'r') as f:
        s = {}
        for line in f:
            line = line.strip()
            if len(line) == 0:
                senses[s['sense_id']] = s
                s = {}
            else:
                line = line.strip().split(':\t')
                key = line[0]
                if len(line) > 1:
                    value = line[1]
                else:
                    key = key[:-1]
                    value = ''
                s[key] = value
    return senses


def process_FEWS_dataset(args, tokenizer):
    senses = FEWS_load_senses(args.FEWS_data_path + "senses.txt")
    key_pos2sense_ids = {}
    sense_id2def_text = {}

    for sense_id in tqdm(senses, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
        sense = senses[sense_id]
        def_text = sense['gloss']
        if len(def_text.split()) == 0:
            continue
        D_sentence_input_ids, _, _ = tokenize_sentence(
            args, tokenizer, normalize_sentence(args, def_text), None)
        sense_id2def_text[sense['sense_id']] = [def_text, D_sentence_input_ids]
        key_pos = '.'.join(sense['sense_id'].split('.')[:2])
        if key_pos not in key_pos2sense_ids:
            key_pos2sense_ids[key_pos] = []
        key_pos2sense_ids[key_pos].append(sense_id)

    instanceList = []
    error_sentence_set = set()
    for data_flag in ["train", "dev", "test"]:
        input_files = []
        if data_flag == "train":
            input_files.append(args.FEWS_data_path + "train/train.txt")
        else:
            input_files.append(args.FEWS_data_path +
                               data_flag + "/" + data_flag + ".few-shot.txt")
            input_files.append(args.FEWS_data_path +
                               data_flag + "/" + data_flag + ".zero-shot.txt")

        for input_file in tqdm(input_files, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
            file = input_file.split("/")[-1]
            instance_lines = open(input_file, 'r')
            line_count = 0
            for line in instance_lines:
                if not line.strip():
                    continue
                line_count += 1
                fields = line.strip().split("\t")
                original_sentence = fields[0]
                original_sentence = clean_text(original_sentence)
                gold_sense_id = fields[1]

                """
                keyword_idx = None
                
                for i, w in enumerate(original_sentence.split()):
                    if "<WSD>" in w:
                        if w[:5] == "<WSD>":
                            keyword_idx = i
                        else:
                            keyword_idx = i+1
                        break
                """
                replace_str = " " + args.mask_token + " "
                E_sentence_masked_str = re.sub(
                    r"<WSD>.*?</WSD>", replace_str, original_sentence, 1)

                cleaned_sentence = original_sentence.replace(
                    "<WSD>", " ").replace("</WSD>", " ")

                cleaned_S = normalize_sentence(args, cleaned_sentence)
                E_S_masked = normalize_sentence(args, E_sentence_masked_str)
                E_sentence_input_ids, E_keyword_start_idx, E_keyword_end_idx = tokenize_sentence(
                    args, tokenizer, cleaned_S, E_S_masked)

                stem_temp, pos_temp = gold_sense_id.split(".")[:2]
                key_pos = stem_temp + "." + pos_temp
                if key_pos in key_pos2sense_ids:
                    all_synsets = key_pos2sense_ids[key_pos]
                else:
                    all_synsets = []

                for synset in all_synsets:
                    def_text, D_sentence_input_ids = sense_id2def_text[synset]

                    new_instance = {"instance_id": "FEWS@"+data_flag+"@" +
                                    file+":"+str(line_count)+"@"+synset, "keyword": stem_temp}
                    new_instance["input1_text"] = def_text
                    new_instance["input1_masked_text"] = "none"
                    new_instance["input1_pos"] = pos_temp
                    new_instance["input1_ids"] = D_sentence_input_ids
                    new_instance["query1_idx"] = [0, 1]

                    # if E_keyword_idx is None:
                    #    print(sents)

                    new_instance["input2_text"] = original_sentence
                    new_instance["input2_masked_text"] = E_sentence_masked_str
                    new_instance["input2_pos"] = pos_temp
                    new_instance["input2_ids"] = E_sentence_input_ids
                    new_instance["query2_idx"] = [
                        E_keyword_start_idx, E_keyword_end_idx]

                    if synset == gold_sense_id:
                        new_instance["class"] = 1
                    else:
                        new_instance["class"] = 0

                    if (E_keyword_start_idx < args.sentence_max_length
                        and E_keyword_end_idx < args.sentence_max_length
                        and len(D_sentence_input_ids) < 512
                        and len(E_sentence_input_ids) < 512):
                        instanceList.append(new_instance)
                    else:
                        error_sentence_set.add(original_sentence)

            instance_lines.close()

    print("len(error_sentence_set):", len(error_sentence_set))
    trainList, devList, testList = [], [], []
    for instance in instanceList:
        if instance["instance_id"].split("@")[1] == "train":
            trainList.append(instance)
        elif instance["instance_id"].split("@")[1] == "dev":
            devList.append(instance)
        elif instance["instance_id"].split("@")[1] == "test":
            testList.append(instance)
    return trainList, devList, testList


def sample_instances(dict_name, keyword, defList, example_sentencesList, combination):
    instanceList = []
    length = len(defList)
    if "DS->ES" in combination:
        for i in range(length):  # DS
            for j in range(length):  # ES
                if len(defList[i]["POS"] & defList[j]["POS"]) == 0:
                    continue
                for x in range(len(example_sentencesList[j])):
                    new_instance = {"instance_id": dict_name +
                                    "@"+str(i)+":"+str(j), "keyword": keyword}
                    new_instance["input1_text"] = defList[i]["text"]
                    new_instance["input1_masked_text"] = defList[i]["masked_text"]
                    new_instance["input1_pos"] = defList[i]["POS"]
                    new_instance["input1_ids"] = defList[i]["input_ids"]
                    new_instance["query1_idx"] = defList[i]["query_idx"]

                    new_instance["input2_text"] = example_sentencesList[j][x]["text"]
                    new_instance["input2_masked_text"] = example_sentencesList[j][x]["masked_text"]
                    new_instance["input2_pos"] = defList[j]["POS"]
                    new_instance["input2_ids"] = example_sentencesList[j][x]["input_ids"]
                    new_instance["query2_idx"] = example_sentencesList[j][x]["query_idx"]

                    if i == j:
                        new_instance["class"] = 1
                    else:
                        new_instance["class"] = 0
                    instanceList.append(new_instance)

    if "ES->DS" in combination:
        for i in range(length):  # DS
            for j in range(length):  # ES
                if len(defList[i]["POS"] & defList[j]["POS"]) == 0:
                    continue
                for x in range(len(example_sentencesList[j])):
                    new_instance = {"instance_id": dict_name +
                                    "@"+str(i)+":"+str(j), "keyword": keyword}
                    new_instance["input2_text"] = defList[i]["text"]
                    new_instance["input2_masked_text"] = defList[i]["masked_text"]
                    new_instance["input2_pos"] = defList[i]["POS"]
                    new_instance["input2_ids"] = defList[i]["input_ids"]
                    new_instance["query2_idx"] = defList[i]["query_idx"]

                    new_instance["input1_text"] = example_sentencesList[j][x]["text"]
                    new_instance["input1_masked_text"] = example_sentencesList[j][x]["masked_text"]
                    new_instance["input1_pos"] = defList[j]["POS"]
                    new_instance["input1_ids"] = example_sentencesList[j][x]["input_ids"]
                    new_instance["query1_idx"] = example_sentencesList[j][x]["query_idx"]

                    if i == j:
                        new_instance["class"] = 1
                    else:
                        new_instance["class"] = 0
                    instanceList.append(new_instance)

    if "DS->DS" in combination:
        for i in range(length):  # DS
            for j in range(length):  # DS
                if len(defList[i]["POS"] & defList[j]["POS"]) == 0:
                    continue

                new_instance = {"instance_id": dict_name +
                                "@"+str(i)+":"+str(j), "keyword": keyword}
                new_instance["input1_text"] = defList[i]["text"]
                new_instance["input1_masked_text"] = defList[i]["masked_text"]
                new_instance["input1_pos"] = defList[i]["POS"]
                new_instance["input1_ids"] = defList[i]["input_ids"]
                new_instance["query1_idx"] = defList[i]["query_idx"]

                new_instance["input2_text"] = defList[j]["text"]
                new_instance["input2_masked_text"] = defList[j]["masked_text"]
                new_instance["input2_pos"] = defList[j]["POS"]
                new_instance["input2_ids"] = defList[j]["input_ids"]
                new_instance["query2_idx"] = defList[j]["query_idx"]

                if i == j:
                    new_instance["class"] = 1
                else:
                    new_instance["class"] = 0
                instanceList.append(new_instance)

    if "ES->ES" in combination:
        for i in range(length):  # ES
            for j in range(length):  # ES
                if len(defList[i]["POS"] & defList[j]["POS"]) == 0:
                    continue
                for x in range(len(example_sentencesList[i])):
                    for y in range(len(example_sentencesList[j])):
                        new_instance = {"instance_id": dict_name +
                                        "@"+str(i)+":"+str(j), "keyword": keyword}
                        new_instance["input1_text"] = example_sentencesList[i][x]["text"]
                        new_instance["input1_masked_text"] = example_sentencesList[i][x]["masked_text"]
                        new_instance["input1_pos"] = defList[i]["POS"]
                        new_instance["input1_ids"] = example_sentencesList[i][x]["input_ids"]
                        new_instance["query1_idx"] = example_sentencesList[i][x]["query_idx"]

                        new_instance["input2_text"] = example_sentencesList[j][y]["text"]
                        new_instance["input2_masked_text"] = example_sentencesList[j][y]["masked_text"]
                        new_instance["input2_pos"] = defList[j]["POS"]
                        new_instance["input2_ids"] = example_sentencesList[j][y]["input_ids"]
                        new_instance["query2_idx"] = example_sentencesList[j][y]["query_idx"]

                        if i == j:
                            new_instance["class"] = 1
                        else:
                            new_instance["class"] = 0
                        # !!!!!!!!!!!!!!!!!
                        if new_instance["input1_text"] != new_instance["input2_text"]:
                            instanceList.append(new_instance)

    return instanceList


def sample_instances_align(dict_name1, dict_name2, keyword, content1List, content2List, combination, symmetry_flag):
    """
    if keyword == "acceptable":
        for c in content1List:
            print(c)
        for c in content2List:
            print(c)
    """
    instanceList = []
    len1 = len(content1List)
    len2 = len(content2List)

    if "DS->ES" in combination:
        for i in range(len1):  # DS
            for j in range(len2):  # ES
                example_sentencesList = content2List[j][1]
                for x in range(len(example_sentencesList)):
                    new_instance = {"instance_id": dict_name1+"@" +
                                    content1List[i][0]["org_idx"]+":"+dict_name2+"@"+content2List[j][0]["org_idx"], 
                                    "keyword": keyword}
                    new_instance["input1_text"] = content1List[i][0]["text"]
                    new_instance["input1_masked_text"] = content1List[i][0]["masked_text"]
                    new_instance["input1_pos"] = content1List[i][0]["POS"]
                    new_instance["input1_ids"] = content1List[i][0]["input_ids"]
                    new_instance["query1_idx"] = content1List[i][0]["query_idx"]

                    new_instance["input2_text"] = example_sentencesList[x]["text"]
                    new_instance["input2_masked_text"] = example_sentencesList[x]["masked_text"]
                    new_instance["input2_pos"] = content2List[j][0]["POS"]
                    new_instance["input2_ids"] = example_sentencesList[x]["input_ids"]
                    new_instance["query2_idx"] = example_sentencesList[x]["query_idx"]

                    if i == j:
                        new_instance["class"] = 1
                    else:
                        new_instance["class"] = 0
                    instanceList.append(new_instance)

        if symmetry_flag == True:
            for i in range(len1):  # ES
                for j in range(len2):  # DS
                    example_sentencesList = content1List[i][1]
                    for x in range(len(example_sentencesList)):
                        new_instance = {"instance_id": dict_name2+"@" +
                                        content2List[j][0]["org_idx"]+":"+dict_name1+"@"+content1List[i][0]["org_idx"],
                                        "keyword": keyword}
                        new_instance["input1_text"] = content2List[j][0]["text"]
                        new_instance["input1_masked_text"] = content2List[j][0]["masked_text"]
                        new_instance["input1_pos"] = content2List[j][0]["POS"]
                        new_instance["input1_ids"] = content2List[j][0]["input_ids"]
                        new_instance["query1_idx"] = content2List[j][0]["query_idx"]

                        new_instance["input2_text"] = example_sentencesList[x]["text"]
                        new_instance["input2_masked_text"] = example_sentencesList[x]["masked_text"]
                        new_instance["input2_pos"] = content1List[i][0]["POS"]
                        new_instance["input2_ids"] = example_sentencesList[x]["input_ids"]
                        new_instance["query2_idx"] = example_sentencesList[x]["query_idx"]

                        if i == j:
                            new_instance["class"] = 1
                        else:
                            new_instance["class"] = 0
                        instanceList.append(new_instance)

    if "DS->DS" in combination:
        for i in range(len1):  # DS
            for j in range(len2):  # DS
                new_instance = {"instance_id": dict_name1+"@" +
                                content1List[i][0]["org_idx"]+":"+dict_name2+"@"+content2List[j][0]["org_idx"], 
                                "keyword": keyword}
                new_instance["input1_text"] = content1List[i][0]["text"]
                new_instance["input1_masked_text"] = content1List[i][0]["masked_text"]
                new_instance["input1_pos"] = content1List[i][0]["POS"]
                new_instance["input1_ids"] = content1List[i][0]["input_ids"]
                new_instance["query1_idx"] = content1List[i][0]["query_idx"]

                new_instance["input2_text"] = content2List[j][0]["text"]
                new_instance["input2_masked_text"] = content2List[j][0]["masked_text"]
                new_instance["input2_pos"] = content2List[j][0]["POS"]
                new_instance["input2_ids"] = content2List[j][0]["input_ids"]
                new_instance["query2_idx"] = content2List[j][0]["query_idx"]

                if i == j:
                    new_instance["class"] = 1
                else:
                    new_instance["class"] = 0
                instanceList.append(new_instance)

    if "ES->ES" in combination:
        for i in range(len1):  # ES
            for j in range(len2):  # ES
                example_sentencesList_i = content1List[i][1]
                example_sentencesList_j = content2List[j][1]
                for x in range(len(example_sentencesList_i)):
                    for y in range(len(example_sentencesList_j)):
                        new_instance = {"instance_id": dict_name1+"@" +
                                        content1List[i][0]["org_idx"]+":"+dict_name2+"@"+content2List[j][0]["org_idx"],
                                        "keyword": keyword}
                        new_instance["input1_text"] = example_sentencesList_i[x]["text"]
                        new_instance["input1_masked_text"] = example_sentencesList_i[x]["masked_text"]
                        new_instance["input1_pos"] = content1List[i][0]["POS"]
                        new_instance["input1_ids"] = example_sentencesList_i[x]["input_ids"]
                        new_instance["query1_idx"] = example_sentencesList_i[x]["query_idx"]

                        new_instance["input2_text"] = example_sentencesList_j[y]["text"]
                        new_instance["input2_masked_text"] = example_sentencesList_j[y]["masked_text"]
                        new_instance["input2_pos"] = content2List[j][0]["POS"]
                        new_instance["input2_ids"] = example_sentencesList_j[y]["input_ids"]
                        new_instance["query2_idx"] = example_sentencesList_j[y]["query_idx"]

                        if i == j:
                            new_instance["class"] = 1
                        else:
                            new_instance["class"] = 0

                        # !!!!!!!!!!!!!!!!!
                        if new_instance["input1_text"] != new_instance["input2_text"]:
                            instanceList.append(new_instance)

    return instanceList


def sample_neg_instances(dict_name, keyword, all_keywords, keyword2defList, 
                        keyword2example_sentencesList, combination, neg_num):
    instanceList = []
    example_sentencesList = keyword2example_sentencesList[keyword]
    defList = keyword2defList[keyword]
    length = len(example_sentencesList)

    if "DS->ES" in combination:
        for j in range(length):  # ES
            for x in range(len(example_sentencesList[j])):
                count = 0
                for _ in range(0, 10 * neg_num):
                    if count >= neg_num:
                        break
                    neg_keyword = random.sample(all_keywords, 1)[0]
                    if neg_keyword == keyword:
                        continue
                    neg_defList = keyword2defList[neg_keyword]
                    i = random.sample(
                        [n for n in range(len(neg_defList))], 1)[0]
                    if len(neg_defList[i]["POS"] & defList[j]["POS"]) == 0:
                        continue

                    new_instance = {"instance_id": dict_name+"@"+neg_keyword +
                                    "#"+str(i)+":"+keyword+"#"+str(j), "keyword": keyword}
                    new_instance["input1_text"] = neg_defList[i]["text"]
                    new_instance["input1_masked_text"] = neg_defList[i]["masked_text"]
                    new_instance["input1_pos"] = neg_defList[i]["POS"]
                    new_instance["input1_ids"] = neg_defList[i]["input_ids"]
                    new_instance["query1_idx"] = neg_defList[i]["query_idx"]

                    new_instance["input2_text"] = example_sentencesList[j][x]["text"]
                    new_instance["input2_masked_text"] = example_sentencesList[j][x]["masked_text"]
                    new_instance["input2_pos"] = defList[j]["POS"]
                    new_instance["input2_ids"] = example_sentencesList[j][x]["input_ids"]
                    new_instance["query2_idx"] = example_sentencesList[j][x]["query_idx"]

                    new_instance["class"] = 0
                    instanceList.append(new_instance)
                    count += 1

    return instanceList


def process_custom_evaluation_dataset(args, test_files, tokenizer):
    instanceList = []
    error_sentence_set = set()
    for file in test_files:
        input_lines = open("../data_custom/" + file)
        count = 0
        for line in tqdm(input_lines, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
            count += 1
            sents = ast.literal_eval(line)
            E_sentence = sents["sentence"].lower()
            words = E_sentence.split()
            target_word = sents["word"]
            target_idx = words.index(target_word)
            all_synsets = sents["synset_names"]
            assert len(all_synsets) == len(sents["definitions"])

            E_sentence_org_str = E_sentence
            E_sentence_masked_str = " ".join(
                words[:target_idx] + [args.mask_token] + words[target_idx+1:])
            E_sentence_input_ids, E_start_idx, E_end_idx = tokenize_sentence(
                args, tokenizer, normalize_sentence(args, E_sentence_org_str), normalize_sentence(args, E_sentence_masked_str))

            for i, synset in enumerate(all_synsets):
                def_text = sents["definitions"][i]
                D_sentence_input_ids, start_idx, end_idx = tokenize_sentence(
                    args, tokenizer, normalize_sentence(args, def_text), None)

                new_instance = {"instance_id": "CustomData@"+file +
                                ":"+str(count)+"@"+synset, "keyword": target_word}
                new_instance["input1_text"] = def_text
                new_instance["input1_masked_text"] = "none"
                new_instance["input1_pos"] = "x"
                new_instance["input1_ids"] = D_sentence_input_ids
                new_instance["query1_idx"] = [0, 1]

                # if E_keyword_idx is None:
                #    print(sents)

                new_instance["input2_text"] = E_sentence_org_str
                new_instance["input2_masked_text"] = E_sentence_masked_str
                new_instance["input2_pos"] = "x"
                new_instance["input2_ids"] = E_sentence_input_ids
                new_instance["query2_idx"] = [E_start_idx, E_end_idx]

                new_instance["class"] = 0

                if (E_start_idx < args.sentence_max_length
                    and E_end_idx < args.sentence_max_length
                    and len(D_sentence_input_ids) < 512
                    and len(E_sentence_input_ids) < 512):
                    instanceList.append(new_instance)
                else:
                    error_sentence_set.add(E_sentence_org_str)
    return instanceList


def process_dict_instanceList(args, tokenizer, combination):
    instanceList = []
    keyword_pos2def_alignments = pickle.load(
        open(args.def_alignment_file, "rb"))

    #dict_nameList = ["OxfordAdvanced", "Webster", "Collins", "LongmanAdvanced", "CambridgeAdvanced", "wordnet"]
    valid_dict_names = args.dict.split("*")

    keyword2definitionsList = []
    for dict_name in valid_dict_names:
        print(dict_name)
        keyword2definitions = load_dictionary(
            args.dictionary_file_path + dict_name + "_def.txt")
        keyword2definitionsList.append(keyword2definitions)

    def_id2content = {}

    for dict_name in valid_dict_names:
        print("processing " + dict_name)
        keyword_not_found_output = open(
            "keyword_not_found_" + dict_name + ".txt", 'w', encoding="utf-8")
        keyword2definitions = keyword2definitionsList[valid_dict_names.index(
            dict_name)]
        keyword2defList = {}
        keyword2example_sentencesList = {}

        count = 0
        for keyword in tqdm(keyword2definitions, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
            if len(keyword) <= 2:
                continue
            if args.dict_ignore_phrase and len(keyword.split()) > 1:
                continue
            count += 1
            # if count > 5000:
            #    break
            definitions = keyword2definitions[keyword]
            defList = []
            example_sentencesList = []
            for idx, definition in enumerate(definitions):
                if "related_items" in definition:
                    continue
                if len(definition["example_sentences"]) == 0:
                    continue
                if " #idiom# " in definition["definition"]:
                    continue

                pos = set(extract_valid_POS(definition, dict_name))
                def_text = definition["definition"].split(" ### ")[0]

                if len(def_text.split()) == 0:
                    continue

                def_text_input_ids, _, _ = tokenize_sentence(
                    args, tokenizer, normalize_sentence(args, def_text), None)

                if len(def_text_input_ids) >= 512:
                    continue

                example_sentences = []
                for ES in definition["example_sentences"]:
                    example_sentence = ES.split(" ### ")[0]

                    example_sentence, masked_example_sentence = mask_keyword(
                        args, keyword, example_sentence, keyword_not_found_output)

                    if masked_example_sentence is not None:
                        example_S = normalize_sentence(args, example_sentence)
                        masked_example_S = normalize_sentence(args, masked_example_sentence)
                        example_sentence_input_ids, keyword_start_idx, keyword_end_idx = tokenize_sentence(
                            args, tokenizer, example_S, masked_example_S)

                        if (keyword_start_idx < args.sentence_max_length
                            and keyword_end_idx < args.sentence_max_length
                            and len(example_sentence_input_ids) < 512):
                            example_sentences.append({"text": ES, 
                                "masked_text": masked_example_sentence, 
                                "input_ids": example_sentence_input_ids, 
                                "query_idx": [keyword_start_idx, keyword_end_idx]})

                def_id = dict_name + ":" + keyword + ":" + str(idx)
                def_content = {"POS": pos, 
                               "org_idx": str(idx), 
                               "text": definition["definition"], 
                               "masked_text": "none", 
                               "input_ids": def_text_input_ids, 
                               "query_idx": [0, 1]}

                def_id2content[def_id] = [def_content, example_sentences]

                example_sentencesList.append(example_sentences)
                defList.append(def_content)

            assert len(example_sentencesList) == len(defList)
            if len(defList) > 0:
                keyword2defList[keyword] = defList
                keyword2example_sentencesList[keyword] = example_sentencesList

            # print(defList)
            # print(example_sentencesList)

        all_keywords = list(keyword2defList.keys())
        for keyword in tqdm(all_keywords, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
            defList = keyword2defList[keyword]
            example_sentencesList = keyword2example_sentencesList[keyword]

            instanceList += sample_instances(dict_name, keyword,
                                             defList, example_sentencesList, combination)
            if args.sample_neg_defs == True:
                instanceList += sample_neg_instances(dict_name, keyword, all_keywords,
                                                     keyword2defList, keyword2example_sentencesList, combination, 3)

    keyword_not_found_output.close()

    random.shuffle(instanceList)
    trainList, devList, testList = [], [], []
    for i, instance in enumerate(instanceList):
        if i % 10 == 0:
            devList.append(instance)
        elif i % 10 == 1:
            testList.append(instance)
        else:
            trainList.append(instance)

    print("len(trainList):", len(trainList))
    print("len(devList):", len(devList))
    print("len(testList):", len(testList))

    if args.align_dict_knowledge:
        valid_dict_pairs = []
        for i in range(0, len(valid_dict_names)):
            for j in range(0, len(valid_dict_names)):
                if i == j:
                    continue
                valid_dict_pairs.append(
                    valid_dict_names[i] + ":" + valid_dict_names[j])

        for keyword_pos in keyword_pos2def_alignments:
            keyword, pos = keyword_pos.split("@@@")

            def_alignments = keyword_pos2def_alignments[keyword_pos]

            for dict_pair in valid_dict_pairs:
                target_dict1, target_dict2 = dict_pair.split(":")
                content1List = []
                content2List = []
                for def_alignment in def_alignments:
                    if def_alignment["score"] < args.align_dict_t:
                        continue
                    dict_name1, kw1, def_idx1 = def_alignment["def1_id"].split(
                        ":")
                    dict_name2, kw2, def_idx2 = def_alignment["def2_id"].split(
                        ":")

                    if not (dict_name1 == target_dict1 and dict_name2 == target_dict2):
                        continue

                    def1_id = dict_name1 + ":" + keyword + ":" + def_idx1
                    def2_id = dict_name2 + ":" + keyword + ":" + def_idx2

                    if not (def1_id in def_id2content and def2_id in def_id2content):
                        continue
                    content1 = def_id2content[def1_id]
                    content2 = def_id2content[def2_id]

                    content1List.append(content1)
                    content2List.append(content2)
                #print(dict_name1, dict_name2)
                # input("continue?")

                symmetry_flag = True

                trainList += sample_instances_align(target_dict1, target_dict2,
                                                    keyword, content1List, content2List, combination, symmetry_flag)

        print("len(trainList):", len(trainList))

    return trainList, devList, testList


def BERT_prepare_data(args, tokenizer):
    if args.eval_dataset == "dictionary":
        combination = ["DS->ES", "DS->DS", "ES->ES"]
        trainList, devList, testList = process_dict_instanceList(
            args, tokenizer, combination)

    elif args.eval_dataset == "WSD":
        train_files, dev_files, test_files = ["semcor"], ["semeval2007"], [
            'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL', 'semeval2007']
        trainList, devList, testList = process_WSD_evaluation_dataset(
            args, train_files, dev_files, test_files, tokenizer)
        if args.use_dict_knowledge:
            combination = ["DS->ES"]
            dict_data = process_dict_instanceList(args, tokenizer, combination)
            for d in dict_data:
                trainList += d

    elif args.eval_dataset == "WiC":
        trainList, devList, testList = process_WiC_dataset(args, tokenizer)
        if args.use_dict_knowledge:
            combination = ["ES->ES"]
            dict_data = process_dict_instanceList(args, tokenizer, combination)
            for d in dict_data:
                trainList += d

    elif args.eval_dataset == "FEWS":
        trainList, devList, testList = process_FEWS_dataset(args, tokenizer)
        if args.use_dict_knowledge:
            combination = ["DS->ES"]
            dict_data = process_dict_instanceList(args, tokenizer, combination)
            for d in dict_data:
                trainList += d

    elif args.eval_dataset == "transfer_eval_DS":
        trainList, devList, testList = [], [], []

        train_files, dev_files, test_files = [], ["semeval2007"], [
            'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL', 'semeval2007']
        data = process_WSD_evaluation_dataset(
            args, train_files, dev_files, test_files, tokenizer)
        devList += data[1]
        testList += data[2]

        data = process_FEWS_dataset(args, tokenizer)
        devList += data[1]
        testList += data[2]

        combination = ["DS->ES"]
        dict_data = process_dict_instanceList(args, tokenizer, combination)
        for d in dict_data:
            trainList += d

    elif args.eval_dataset == "transfer_eval_ES":
        trainList, devList, testList = [], [], []

        data = process_WiC_dataset(args, tokenizer)
        devList += data[1]
        testList += data[2]

        combination = ["ES->ES"]
        dict_data = process_dict_instanceList(args, tokenizer, combination)
        for d in dict_data:
            trainList += d

    elif args.eval_dataset == "transfer_eval_all":
        trainList, devList, testList = [], [], []

        train_files, dev_files, test_files = [], ["semeval2007"], [
            'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL', 'semeval2007']
        data = process_WSD_evaluation_dataset(
            args, train_files, dev_files, test_files, tokenizer)
        devList += data[1]
        testList += data[2]

        data = process_WiC_dataset(args, tokenizer)
        devList += data[1]
        testList += data[2]

        data = process_FEWS_dataset(args, tokenizer)
        devList += data[1]
        testList += data[2]

        combination = ["DS->ES", "ES->ES"]
        dict_data = process_dict_instanceList(args, tokenizer, combination)
        for d in dict_data:
            trainList += d

    if args.eval_dataset == "custom_data":
        trainList, devList, testList = [], [], []
        testList += process_custom_evaluation_dataset(
            args, ["adj_wsd_set.json", "noun_wsd_set.json"], tokenizer)

    random.shuffle(trainList)

    output = open("log_" + args.loss + ".txt", "w")
    output.write(str(args)+"\n\n\n")
    for instance in trainList[:5000]:
        output.write(str(instance) + "\n\n")
    output.close()

    # random.shuffle(trainList)

    print("{} train instances".format(len(trainList)))
    print("{} valid instances".format(len(devList)))
    print("{} test instances".format(len(testList)))

    env = {}
    env["train"] = trainList
    env["dev"] = devList
    env["test"] = testList

    pickle.dump(env, open("env.pkl", "wb"))
