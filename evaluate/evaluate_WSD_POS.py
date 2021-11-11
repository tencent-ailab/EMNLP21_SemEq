import os, sys, ast, subprocess, argparse

def evaluate_output(gold_file, out_file):
    eval_cmd = ['java', 'Scorer', gold_file, out_file]
    #print (eval_cmd)
    output = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE).communicate()[0]
    output = str(output, 'utf-8')
    output = output.splitlines()
    p,r,f1 =  [float(output[i].split('=')[-1].strip()[:-1]) for i in range(3)]
    return p, r, f1

def read_instances_pos(xml_file):
    instance_id2pos = {}
    input_lines = open(xml_file, 'r')
    for line in input_lines:
        if not line.strip():
            continue
        if "<instance" in line:
            instance_id = None
            pos = None
            words = line.split()
            for w in words:
                if "id=" in w:
                    instance_id = w.split("\"")[1]
                if "pos=" in w:
                    pos = w.split("\"")[1]
            assert instance_id is not None
            assert pos is not None
            instance_id2pos[instance_id] = pos

    input_lines.close()
    return instance_id2pos

def synset_occurrence(gold_file):
    synset2num = {}
    input_lines = open(gold_file, "r")
    for line in input_lines:
        for synset in line.strip().split()[1:]:
            if synset not in synset2num:
                synset2num[synset] = 0
            synset2num[synset] += 1
    input_lines.close()
    return synset2num

def eval_prediction(test_data_prefix, train_synset2num):
    instance_id2pos = read_instances_pos("../datasets/WSD_Evaluation_Framework/Evaluation_Datasets/" + test_data_prefix + "/" + test_data_prefix + ".data.xml")
    gold_file = "../datasets/WSD_Evaluation_Framework/Evaluation_Datasets/" + test_data_prefix + "/" + test_data_prefix + ".gold.key.txt"
    pred_file = test_data_prefix + "_output.key.txt"
    instance_id2gold_line = {}
    instance_id2pred_line = {}
    instance_idList = []
    range2instance_ids = {"0":set(), "1-2":set(), "3-5":set(), "6-10":set(), "10+":set()}
    with open(gold_file, 'r') as input_lines:
        for line in input_lines:
            instance_id = line.split()[0]
            instance_id2gold_line[instance_id] = line
            instance_idList.append(instance_id)

            synsets = line.strip().split()[1:]

            for synset in synsets:
                if synset not in train_synset2num:
                    range2instance_ids["0"].add(instance_id)
                elif 1 <= train_synset2num[synset] and train_synset2num[synset] <= 2:
                    range2instance_ids["1-2"].add(instance_id)
                elif 3 <= train_synset2num[synset] and train_synset2num[synset] <= 5:
                    range2instance_ids["3-5"].add(instance_id)
                elif 6 <= train_synset2num[synset] and train_synset2num[synset] <= 10:
                    range2instance_ids["6-10"].add(instance_id)
                elif 10 < train_synset2num[synset]:
                    range2instance_ids["10+"].add(instance_id)


    with open(pred_file, 'r') as input_lines:
        for line in input_lines:
            instance_id = line.split()[0]
            instance_id2pred_line[instance_id] = line

    for POS in ["NOUN", "VERB", "ADJ", "ADV"]:
        gold_file = test_data_prefix + "_" + POS + ".gold.key.txt"
        pred_file = test_data_prefix + "_" + POS + ".pred.key.txt"
        gold_output = open(gold_file, "w")
        pred_output = open(pred_file, "w")
        for instance_id in instance_idList:
            if instance_id2pos[instance_id] == POS:
                gold_output.write(instance_id2gold_line[instance_id])
                pred_output.write(instance_id2pred_line[instance_id])
        gold_output.close()
        pred_output.close()

        print(POS, evaluate_output(gold_file, pred_file))

    for rg in ["0", "1-2", "3-5", "6-10", "10+"]:
        gold_file = test_data_prefix + "_" + rg + ".gold.key.txt"
        pred_file = test_data_prefix + "_" + rg + ".pred.key.txt"
        gold_output = open(gold_file, "w")
        pred_output = open(pred_file, "w")
        for instance_id in instance_idList:
            if instance_id in range2instance_ids[rg]:
                gold_output.write(instance_id2gold_line[instance_id])
                pred_output.write(instance_id2pred_line[instance_id])
        gold_output.close()
        pred_output.close()
        print("Number of instances in this range:", len(range2instance_ids[rg]))
        print(rg, evaluate_output(gold_file, pred_file))


# python ../BERT_model/evaluate_WSD.py --loss CrossEntropy --epoch 1
# python ../BERT_model/evaluate_WSD_POS.py --loss CrossEntropy --epoch 1
if __name__ == "__main__":
    os.system("cp ../EWISE/Scorer.class .")

    """
    test_data_prefix = "semeval2007"
    gold_file = "../EWISE/external/WSD_Evaluation_Framework/Evaluation_Datasets/" + test_data_prefix + "/" + test_data_prefix + ".gold.key.txt"
    print("P, R, F1:", evaluate_output(gold_file, gold_file))
    """
    parser = argparse.ArgumentParser()
    #parser.add_argument("--epoch", default="10", type=str, help="Which epoch to evaluate.")
    #parser.add_argument("--loss", default="CrossEntropy", type=str, help="Which loss function to evaluate (CrossEntropy, CosineSimilarity, Contrastive).")
    args = parser.parse_args()

    test_prefixList = ['ALL']

    train_synset2num = synset_occurrence("../datasets/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt")

    for test_prefix in test_prefixList:
        #print(test_prefix, eval_prediction("test_TruePred_" + args.epoch + "_" + args.loss + ".txt", test_prefix)[-1])
        print(test_prefix,)
        eval_prediction(test_prefix, train_synset2num)
    


