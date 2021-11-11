import torch
from torch import nn

DROPOUT = 0.2

# https://github.com/UKPLab/sentence-transformers/tree/master/sentence_transformers/losses


class SoftmaxLoss(nn.Module):
    def __init__(self,
                 args,
                 classes_weight,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = True):
        super(SoftmaxLoss, self).__init__()
        sentence_embedding_dimension = args.bert_hidden_size
        self.num_labels = args.class_num
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        self.use_hidden_layer = args.use_hidden_layer

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1

        #self.linear_layer = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)

        self.linear_layer = nn.Linear(
            num_vectors_concatenated * sentence_embedding_dimension, 300)
        self.linear_layer2 = nn.Linear(300, self.num_labels)

        self.linear_layer3 = nn.Linear(
            num_vectors_concatenated * sentence_embedding_dimension, self.num_labels)

        self.drop = nn.Dropout(args.dropout)
        self.activation = nn.ReLU()

        self.loss_fct = nn.CrossEntropyLoss(weight=classes_weight)

    def forward(self, rep_a, rep_b, targets):

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))
            #vectors_concat.append(rep_a - rep_b)

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        if self.use_hidden_layer:
            output = self.drop(self.linear_layer(features))
            #output = self.linear_layer(features)
            output = self.linear_layer2(self.activation(output))
        else:
            output = self.linear_layer3(features)

        if targets is not None:
            loss = self.loss_fct(output, targets)
            return loss
        else:
            return output
