import os, sys, ast, subprocess, argparse

def evaluate_output(gold_file, out_file):
    eval_cmd = ['java', 'Scorer', gold_file, out_file]
    #print (eval_cmd)
    output = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE).communicate()[0]
    output = str(output, 'utf-8')
    output = output.splitlines()
    p,r,f1 =  [float(output[i].split('=')[-1].strip()[:-1]) for i in range(3)]
    return p, r, f1

def eval_prediction(prediction_file, test_data_prefix):
    input_lines = open(prediction_file, 'r')
    instance_id2prediction = {}
    instance_id2true = {}
    for line in input_lines:
        fields = line.strip().split("\t")
        if fields[0].split("@")[0] != "WSD":
            continue
        mdl_output = ast.literal_eval(fields[4])
        if len(mdl_output) == 1:
            score = mdl_output
        else:
            score = mdl_output[1]

        data_prefix, instance_id, synset = fields[0].split("@")[1], fields[0].split("@")[2], fields[0].split("@")[3]
        if fields[2] == "1":
            instance_id2true[instance_id] = synset
        if data_prefix == test_data_prefix:
            if instance_id not in instance_id2prediction:
                instance_id2prediction[instance_id] = []
            instance_id2prediction[instance_id].append([synset, score])

    input_lines.close()
    gold_file = "../datasets/WSD_Evaluation_Framework/Evaluation_Datasets/" + test_data_prefix + "/" + test_data_prefix + ".gold.key.txt"
    gold_lines = open(gold_file, 'r')
    output = open(test_data_prefix + "_output.key.txt", "w")
    count1 = 0
    count2 = 0
    for line in gold_lines:
        if not line.strip():
            continue
        #print(line)
        instance_id = line.split()[0]
        if instance_id in instance_id2prediction:
            
            scores = [x[1] for x in instance_id2prediction[instance_id]]
            index = scores.index(max(scores))

            pred_synset = instance_id2prediction[instance_id][index][0]
            #output.write('{} {}\n'.format(instance_id, pred_synset))
            if max(scores) < 0.5:
                #print(instance_id, instance_id2true[instance_id], pred_synset, instance_id2prediction[instance_id][0][0])
                #print("scores:", scores)
                if instance_id2true[instance_id] == pred_synset:
                    count1 += 1
                if instance_id2true[instance_id] == instance_id2prediction[instance_id][0][0]:
                    count2 += 1
            
            pred_synsetList = [pred_synset]
            """
            for p in instance_id2prediction[instance_id]:
                if p[1] > 0.95:
                    pred_synsetList.append(p[0])
            """
            pred_synsetList = list(set(pred_synsetList))
            output.write(instance_id + " " + " ".join(pred_synsetList) + "\n")
            
        else:
            pred_synset = "xxxx"
            output.write('{} {}\n'.format(instance_id, pred_synset))
    #print(count1, count2)

    gold_lines.close()
    output.close()
    return evaluate_output(gold_file, test_data_prefix + "_output.key.txt")

    #print("P, R, F1:", evaluate_output(gold_file, test_data_prefix + "_output.key.txt"))


# python ../BERT_model/evaluate_WSD.py --loss CrossEntropy --epoch 1
if __name__ == "__main__":
    os.system("cp ../EWISE/Scorer.class .")

    """
    test_data_prefix = "semeval2007"
    gold_file = "../EWISE/external/WSD_Evaluation_Framework/Evaluation_Datasets/" + test_data_prefix + "/" + test_data_prefix + ".gold.key.txt"
    print("P, R, F1:", evaluate_output(gold_file, gold_file))
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default="10", type=str, help="Which epoch to evaluate.")
    parser.add_argument("--loss", default="CrossEntropy", type=str, help="Which loss function to evaluate (CrossEntropy, CosineSimilarity, Contrastive).")
    args = parser.parse_args()

    dev_prefixList = ['semeval2007']

    for dev_prefix in dev_prefixList:
        #print(dev_prefix, eval_prediction("valid_TruePred_" + args.epoch + "_" + args.loss + ".txt", dev_prefix)[-1])
        print(dev_prefix, eval_prediction("valid_TruePred_" + args.epoch + "_" + args.loss + ".txt", dev_prefix))

    test_prefixList = ['senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL']

    for test_prefix in test_prefixList:
        #print(test_prefix, eval_prediction("test_TruePred_" + args.epoch + "_" + args.loss + ".txt", test_prefix)[-1])
        print(test_prefix, eval_prediction("test_TruePred_" + args.epoch + "_" + args.loss + ".txt", test_prefix))
    


