import argparse
import glob
import pickle
import random
import time
import sys
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from transformers import *
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from models import BERT_NN, BERT_NN_SEP
from loss_functions import SoftmaxLoss



def tensor_to_numpy(x):
    ''' Need to cast before calling numpy()
    '''
    # return (Variable(x).data).cpu().numpy()
    return x.data.type(torch.DoubleTensor).cpu().numpy()


def seq_padding(input_ids, max_seq_length, pad_token_id):
    #pad_token = 0

    if len(input_ids) < max_seq_length:
        padding_length = max_seq_length - len(input_ids)
    else:
        padding_length = 0
        input_ids = input_ids[:max_seq_length]

    input_mask = [1] * len(input_ids) + [0] * padding_length
    input_ids = input_ids + [pad_token_id] * padding_length

    # print(input_ids)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return input_ids, input_mask


def prepare_input_tensor(input_ids_list, keyword_idx_list, sentence_max_length, pad_token_id):
    input_lenList = [len(input_ids) for input_ids in input_ids_list]
    # use given sentence_max_length

    max_len = sentence_max_length
    """
    # use batch longest length
    max_len = max(input_lenList)
    #max_len = min([sentence_max_length, max_len])
    """
    padded_input_ids_list = []
    mask_list = []
    for input_ids in input_ids_list:
        padded_input_ids, mask = seq_padding(input_ids, max_len, pad_token_id)
        padded_input_ids_list.append(padded_input_ids)
        mask_list.append(mask)
    output_mask_list = []
    key_len_list = []
    for keyword_idx in keyword_idx_list:
        output_mask_list.append([0.0] * (keyword_idx[0]) + [1.0] * (
            keyword_idx[1]-keyword_idx[0]) + [0.0] * (max_len-keyword_idx[1]))
        key_len_list.append(float(keyword_idx[1]-keyword_idx[0]))
    return (torch.tensor(padded_input_ids_list),
           torch.tensor(mask_list),
           torch.tensor(output_mask_list),
           torch.tensor(key_len_list))


class Experiment:
    def __init__(self, args, tokenizer):
        self.args = args
        self.exp_log = open("exp_log_" + args.loss + ".txt", 'w', 1)

        self.env = pickle.load(open("env.pkl", "rb"))

        self.train_set = self.env['train']
        self.dev_set = self.env['dev']
        self.test_set = self.env['test']

        if(self.args.toy is True):
            print("Using toy mode...")
            random.shuffle(self.train_set)
            random.shuffle(self.dev_set)
            random.shuffle(self.test_set)

            self.train_set = self.train_set[:500]
            self.dev_set = self.dev_set[:100]
            self.test_set = self.test_set[:100]

        classes_freq = [1 for i in range(0, self.args.class_num)]
        for instance in self.train_set:
            classes_freq[instance["class"]] += 1
        classes_freq_sum = sum(classes_freq)

        #classes_weight = [math.log(float(classes_freq_sum)/float(freq)) for freq in classes_freq]
        print(self.args.scale_weight)
        self.exp_log.write(str(self.args.scale_weight) + "\n")
        if self.args.scale_weight is True:
            classes_weight = [float(classes_freq_sum)/float(freq)
                              for freq in classes_freq]
        else:
            classes_weight = [1.0 for freq in classes_freq]
            #classes_weight = [2.0, 1.0]

        self.classes_weight = torch.from_numpy(
            np.array(classes_weight, dtype='float32'))
        print("classes_freq:", classes_freq)
        self.exp_log.write("classes_freq: " + str(classes_freq) + "\n")
        print("classes_weight:", classes_weight)
        self.exp_log.write("classes_weight: " + str(classes_weight) + "\n")

        # if self.args.experiment == "BERT":

        if self.args.sep_encoder is True:
            self.mdl = BERT_NN_SEP(args)
        else:
            # !!!!!!!!!!!!!! should not mix gloss/context and context/context pairs in training !!!!!!!!!!!!!!!
            self.mdl = BERT_NN(args)

        self.encoder1_flag = self.args.encoder1_flag
        self.encoder2_flag = self.args.encoder2_flag

        # loss functions

        if self.args.loss == "CrossEntropy":
            if "rep" in self.args.concat:
                concat_rep = True
            else:
                concat_rep = False
            if "diff" in self.args.concat:
                concat_difference = True
            else:
                concat_difference = False
            if "mul" in self.args.concat:
                concat_multiplication = True
            else:
                concat_multiplication = False

            self.criterion = SoftmaxLoss(
                args=self.args,
                classes_weight=self.classes_weight,
                concatenation_sent_rep=concat_rep,
                concatenation_sent_difference=concat_difference,
                concatenation_sent_multiplication=concat_multiplication)

        if self.args.cuda is True:
            # self.mdl = nn.DataParallel(self.mdl) # !!!!!
            self.mdl.to(self.args.device)
            self.classes_weight = self.classes_weight.to(self.args.device)
            self.criterion.to(self.args.device)

    def select_optimizer(self):
        if self.args.optimizer == "Adam":
            parameters = filter(lambda p: p.requires_grad,
                                self.mdl.parameters())
            parameters = list(parameters) + list(self.criterion.parameters())
            self.optimizer = optim.Adam(parameters, lr=self.args.learning_rate)
            #self.optimizer = optim.AdamW(parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        elif self.args.optimizer == "AdamW":
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.mdl.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in self.mdl.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0},
                {'params': self.criterion.parameters()}
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
            self.schedule = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=10000, num_training_steps=len(self.train_set)*self.args.epochs)

    def make_batch(self, x, i, batch_size):
        '''
        :param x: input sentences
        :param i: select the ith batch (-1 to take all)
        :return: sentences, targets, actual_batch_size
        '''
        batch = x[int(i * batch_size):int((i + 1) * batch_size)]

        if self.encoder1_flag == "gloss":
            (input1_ids_tensor, input1_mask_tensor,
            input1_output_mask_tensor, input1_key_len_tensor) = prepare_input_tensor(
                [instance['input1_ids'] for instance in batch],
                [instance["query1_idx"] for instance in batch],
                self.args.gloss_max_length,
                self.args.PAD_id)
        else:
            (input1_ids_tensor, input1_mask_tensor,
            input1_output_mask_tensor, input1_key_len_tensor) = prepare_input_tensor(
                [instance['input1_ids'] for instance in batch],
                [instance["query1_idx"] for instance in batch],
                self.args.sentence_max_length,
                self.args.PAD_id)

        (input2_ids_tensor, input2_mask_tensor,
        input2_output_mask_tensor, input2_key_len_tensor) = prepare_input_tensor(
                [instance['input2_ids'] for instance in batch],
                [instance["query2_idx"] for instance in batch],
                self.args.sentence_max_length,
                self.args.PAD_id)

        targets = torch.LongTensor(
            np.array([instance['class'] for instance in batch], dtype=np.int32).tolist())

        if self.args.cuda is True:
            input1_ids_tensor = input1_ids_tensor.to(self.args.device)
            input2_ids_tensor = input2_ids_tensor.to(self.args.device)

            input1_mask_tensor = input1_mask_tensor.to(self.args.device)
            input2_mask_tensor = input2_mask_tensor.to(self.args.device)

            input1_output_mask_tensor = input1_output_mask_tensor.to(
                self.args.device)
            input2_output_mask_tensor = input2_output_mask_tensor.to(
                self.args.device)

            input1_key_len_tensor = input1_key_len_tensor.to(self.args.device)
            input2_key_len_tensor = input2_key_len_tensor.to(self.args.device)

            targets = targets.to(self.args.device)

        actual_batch_size = input1_ids_tensor.size(0)

        return {"targets": targets,
                "input1_ids_tensor": input1_ids_tensor, "input2_ids_tensor": input2_ids_tensor,
                "input1_mask_tensor": input1_mask_tensor, "input2_mask_tensor": input2_mask_tensor,
                "input1_output_mask_tensor": input1_output_mask_tensor,
                "input2_output_mask_tensor": input2_output_mask_tensor,
                "input1_key_len_tensor": input1_key_len_tensor,
                "input2_key_len_tensor": input2_key_len_tensor,
                "actual_batch_size": actual_batch_size}


    def train_batch(self, i, batch_size):
        self.mdl.train()
        self.criterion.train()

        # self.mdl.zero_grad()
        self.optimizer.zero_grad()

        # if self.args.experiment == "BERT":

        #loss = self.criterion(output, targets)

        batch = self.make_batch(self.train_set, i, batch_size)
        emb1 = self.mdl(batch["input1_ids_tensor"], batch["input1_mask_tensor"],
                        batch["input1_output_mask_tensor"], batch["input1_key_len_tensor"], self.encoder1_flag)
        emb2 = self.mdl(batch["input2_ids_tensor"], batch["input2_mask_tensor"],
                        batch["input2_output_mask_tensor"], batch["input2_key_len_tensor"], self.encoder2_flag)
        loss = self.criterion(emb1, emb2, batch["targets"])

        loss.backward()

        nn.utils.clip_grad_norm_(parameters=list(self.mdl.parameters(
        ))+list(self.criterion.parameters()), max_norm=self.args.max_grad_norm)
        self.optimizer.step()
        if self.args.optimizer == "AdamW":
            self.schedule.step()

        # return loss.data[0]
        return loss.item()

    def train(self):
        """
        This is the main train function
        """

        print(self.args)
        self.exp_log.write(str(self.args) + "\n")

        if len(self.train_set) % self.args.batch_size == 0:
            num_batches = int(len(self.train_set) / self.args.batch_size)
        else:
            num_batches = int(len(self.train_set) / self.args.batch_size) + 1

        print("len(self.train_set)", len(self.train_set))
        self.exp_log.write("len(self.train_set): " +
                           str(len(self.train_set)) + "\n")
        print("num_batches:", num_batches)
        self.exp_log.write("num_batches: " + str(num_batches) + "\n")
        self.select_optimizer()


        final_results_strList = []
        for epoch in range(1, self.args.epochs+1):
            self.mdl.train()
            self.criterion.train()
            print("epoch: ", epoch)
            self.exp_log.write("epoch: " + str(epoch) + "\n")
            #t0 = time.clock()
            t0 = time.perf_counter()
            random.shuffle(self.train_set)
            print(
                "========================================================================")
            self.exp_log.write("=====================================\n")
            losses = []
            for i in tqdm(range(num_batches), bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
                loss = self.train_batch(i, self.args.batch_size)
                if(loss is None):
                    continue
                losses.append(loss)
            #t1 = time.clock()
            t1 = time.perf_counter()
            print("[Epoch {}] Train Loss={} T={}s".format(
                epoch, np.mean(losses), t1-t0))

            if len(self.dev_set) != 0:
                print("Evaluate on dev set...")
                self.exp_log.write("Evaluate on dev set...\n")
                avg_P, avg_R, avg_F, results_str = self.test(epoch, "dev")
                print(results_str)
                self.exp_log.write(results_str + "\n")

            if len(self.test_set) != 0:
                print("Evaluate on test set...")
                self.exp_log.write("Evaluate on test set...\n")
                avg_P, avg_R, avg_F, results_str = self.test(epoch, "test")
                print(results_str)
                self.exp_log.write(results_str + "\n")

            if "transfer_eval" in self.args.eval_dataset:
                torch.save(
                    {
                    "args": self.args,
                    "mdl": self.mdl.state_dict(),
                    "criterion": self.criterion.state_dict()
                    },
                    "epoch" + str(epoch) + "_model_" + self.args.loss + ".pt")

    def predict(self):
        exp_log_output = open("eval_trained_model.txt", "w")
        if len(self.dev_set) != 0:
            print("Evaluate on dev set...")
            exp_log_output.write("Evaluate on dev set...\n")
            avg_P, avg_R, avg_F, results_str = self.test("saved_model", "dev")
            print(results_str)
            exp_log_output.write(results_str + "\n")

        if len(self.test_set) != 0:
            print("Evaluate on test set...")
            exp_log_output.write("Evaluate on test set...\n")
            avg_P, avg_R, avg_F, results_str = self.test("saved_model", "test")
            print(results_str)
            exp_log_output.write(results_str + "\n")
        exp_log_output.close()

    def test(self, epoch, data_flag):
        if data_flag == "dev":
            dataset = self.dev_set
            output_file = open("valid_TruePred_" + str(epoch) +
                               "_" + self.args.loss + ".txt", "w")

        elif data_flag == "test":
            dataset = self.test_set
            output_file = open("test_TruePred_" + str(epoch) +
                               "_" + self.args.loss + ".txt", "w")

        all_probs, all_preds, acc, avg_P, avg_R, avg_F, results_str = self.evaluate(dataset)

        if output_file is None:
            return avg_P, avg_R, avg_F, results_str

        for i, instance in enumerate(dataset):
            output_file.write(str(instance["instance_id"])
                + "\t" + str(instance["keyword"])
                + "\t" + str(instance["class"])
                + "\t" + str(all_preds[i])
                + "\t" + str(all_probs[i])
                + "\t" + instance["input1_text"]
                + "\t" + instance["input2_text"] + "\n")
        output_file.close()
        return avg_P, avg_R, avg_F, results_str

    def evaluate(self, x):
        self.mdl.eval()
        self.criterion.eval()
        BS = self.args.batch_size // 2
        if BS == 0:
            BS = 1

        if len(x) % BS == 0:
            num_batches = int(len(x) / BS)
        else:
            num_batches = int(len(x) / BS) + 1

        all_probs = []
        all_preds = []
        all_targets = []

        for instance in x:
            all_targets.append(instance["class"])

        for i in range(num_batches):
            batch = self.make_batch(x, i, BS)
            emb1 = self.mdl(batch["input1_ids_tensor"], batch["input1_mask_tensor"],
                            batch["input1_output_mask_tensor"], batch["input1_key_len_tensor"],
                            self.encoder1_flag)
            emb2 = self.mdl(batch["input2_ids_tensor"], batch["input2_mask_tensor"],
                            batch["input2_output_mask_tensor"], batch["input2_key_len_tensor"],
                            self.encoder2_flag)
            """
            emb1 = self.mdl(input1_ids_tensor, query1_idx)
            emb2 = self.mdl(input2_ids_tensor, query2_idx)
            """
            output = self.criterion(emb1, emb2, targets=None)
            if self.args.loss == "CrossEntropy":
                output = nn.functional.softmax(output, dim=1)

            # print(actual_batch_size)
            all_probs += tensor_to_numpy(output).tolist()

        if self.args.loss == "CrossEntropy":
            for probs in all_probs:
                all_preds.append(probs.index(max(probs)))

        # print("len(all_targets):", len(all_targets), "len(all_preds):", len(all_preds))
        confusion_matrix = {}
        matches = 0
        for i in range(len(all_targets)):
            if all_targets[i] == all_preds[i]:
                matches += 1
            string = str(all_targets[i]) + " --> " + str(all_preds[i])
            if string in confusion_matrix:
                confusion_matrix[string] += 1
            else:
                confusion_matrix[string] = 1
        acc = float(matches) / float(len(all_targets))
        print("accuracy:", acc)
        print("confusion_matrix[target --> pred]:", confusion_matrix)

        labelList = [label for label in range(0, self.args.class_num)]
        results = precision_recall_fscore_support(
            all_targets, all_preds, average=None, labels=labelList)
        #results = precision_recall_fscore_support(all_targets, all_preds, average=None)

        results_str = []
        results_str.append("accuracy: " + str(acc))
        results_str.append("\t".join([str(l) for l in labelList]))
        results_str.append("\t".join(["%0.4f" % p for p in results[0]]))
        results_str.append("\t".join(["%0.4f" % r for r in results[1]]))
        results_str.append("\t".join(["%0.4f" % f for f in results[2]]))
        results_str = "\n".join(results_str)
        avg = precision_recall_fscore_support(
            all_targets, all_preds, average='macro')
        avg_P, avg_R, avg_F = avg[0], avg[1], avg[2]
        # print(results_str)

        return all_probs, all_preds, acc, avg_P, avg_R, avg_F, results_str


def model_main(args, tokenizer):
    exp = Experiment(args, tokenizer)
    if args.exp_mode == "train":
        print("Training...")
        exp.train()
        torch.save({
            "args": exp.args,
            "mdl": exp.mdl.state_dict(),
            "criterion": exp.criterion.state_dict()},
            "trained_model_" + args.loss + ".pt")
    elif args.exp_mode == "twoStageTune":
        print("Two stages tuning...")
        checkpoint = torch.load("pretrained_model_" + args.loss + ".pt")
        exp.mdl.load_state_dict(checkpoint["mdl"])
        exp.criterion.load_state_dict(checkpoint["criterion"])
        exp.train()
    elif args.exp_mode == "eval":
        print("Evaluating...")
        checkpoint = torch.load("trained_model_" + args.loss + ".pt")
        #exp.args = checkpoint["args"]
        exp.mdl.load_state_dict(checkpoint["mdl"])
        exp.criterion.load_state_dict(checkpoint["criterion"])
        exp.predict()
