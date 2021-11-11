import os, sys, ast, subprocess, argparse

def evaluate_output(file):
    resultList = []
    with open(file, 'r') as input_lines:
        for line in input_lines:
            fields = line.strip().split()
            if fields[1] == fields[2].replace("?.",""):
                resultList.append(1)
            else:
                resultList.append(0)
    return float(sum(resultList))/float(len(resultList))

def eval_prediction(prediction_file, data_prefix):
    input_lines = open(prediction_file, 'r')
    instance_id2prediction = {}
    for line in input_lines:

        fields = line.strip().split("\t")
        if fields[0].split("@")[0] != "FEWS":
            continue

        mdl_output = ast.literal_eval(fields[4])
        if len(mdl_output) == 1:
            score = mdl_output
        else:
            score = mdl_output[1]

        instance_id, synset = fields[0].split("@")[2], fields[0].split("@")[3]
        if instance_id.split(":")[0] == data_prefix + "-shot.txt":
            if instance_id not in instance_id2prediction:
                instance_id2prediction[instance_id] = []
            instance_id2prediction[instance_id].append([synset, score])

    input_lines.close()
    gold_file = "../datasets/fews/" + data_prefix.split(".")[0] + "/" + data_prefix + "-shot.txt"
    gold_lines = open(gold_file, 'r')
    output = open(data_prefix + ".TruePred.txt", "w")
    count = 0
    for line in gold_lines:
        if not line.strip():
            continue
        count += 1
        #print(line)
        instance_id = gold_file.split("/")[-1] + ":" + str(count)
        gold_synset = line.strip().split("\t")[1]
        if instance_id in instance_id2prediction:
            scores = [x[1] for x in instance_id2prediction[instance_id]]
            index = scores.index(max(scores))
            pred_synset = instance_id2prediction[instance_id][index][0]
            output.write('{} {} {}\n'.format(instance_id, gold_synset, pred_synset))
        else:
            #pred_synset = "xxxx"
            pred_synset = "?." + ".".join(gold_synset.split(".")[:2] + ["0"])
            output.write('{} {} {}\n'.format(instance_id, gold_synset, pred_synset))

    gold_lines.close()
    output.close()

    return evaluate_output(data_prefix + ".TruePred.txt")


# python ../BERT_model/evaluate_FEWS.py --loss CrossEntropy --epoch 1
if __name__ == "__main__":
    #os.system("cp ../EWISE/Scorer.class .")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default="10", type=str, help="Which epoch to evaluate.")
    parser.add_argument("--loss", default="CrossEntropy", type=str, help="Which loss function to evaluate (CrossEntropy, CosineSimilarity, Contrastive).")
    args = parser.parse_args()

    dev_prefixList = ['dev.few', 'dev.zero']
    test_prefixList = ['test.few', 'test.zero']

    for dev_prefix in dev_prefixList:
        print(dev_prefix, eval_prediction("valid_TruePred_" + args.epoch + "_" + args.loss + ".txt", dev_prefix))

    for test_prefix in test_prefixList:
        print(test_prefix, eval_prediction("test_TruePred_" + args.epoch + "_" + args.loss + ".txt", test_prefix))
    


