import os, sys, ast, subprocess, argparse

def calculate_acc(preds, golds):
    count = 0.0
    for i in range(len(preds)):
        if preds[i] == golds[i]:
            count += 1
    return count/float(len(preds))


def eval_prediction(args):
    test_pred_file = "test_TruePred_" + args.epoch + "_" + args.loss + ".txt"
    dev_pred_file = "valid_TruePred_" + args.epoch + "_" + args.loss + ".txt"
    input_lines = open(dev_pred_file, 'r')
    instance_id2prediction = {}
    for line in input_lines:
        fields = line.strip().split("\t")
        if fields[0].split("@")[0] != "WiC":
            continue
        instance_id = fields[0].split("_")[0]
        mdl_output = ast.literal_eval(fields[4])
        if len(mdl_output) == 1:
            score = mdl_output
        else:
            score = mdl_output[1]
        if instance_id not in instance_id2prediction:
            instance_id2prediction[instance_id] = {"gold_label": int(fields[2]), "scores": [score]}
        else:
            instance_id2prediction[instance_id]["scores"].append(score)
    input_lines.close()

    trues = []
    preds = []
    for instance_id in instance_id2prediction:
        prediction = instance_id2prediction[instance_id]
        trues.append(prediction["gold_label"])
        if sum(prediction["scores"])/float(len(prediction["scores"])) >= 0.5:
            preds.append(1)
        else:
            preds.append(0)

    assert len(trues) == 638
    assert len(preds) == 638
    count = 0
    for i in range(len(trues)):
        if trues[i] == preds[i]:
            count += 1

    print("val acc:", float(count)/float(len(trues)))

    #best_t = search_decision_t(instance_id2prediction, args.start_t, args.step_size, args.max_steps)
    best_t = 0.5

    

    input_lines = open(test_pred_file, 'r')
    instance_id2prediction = {}
    
    for line in input_lines:
        fields = line.strip().split("\t")
        if fields[0].split("@")[0] != "WiC":
            continue

        instance_id = fields[0].split("_")[0]
        mdl_output = ast.literal_eval(fields[4])
        if len(mdl_output) == 1:
            score = mdl_output
        else:
            score = mdl_output[1]
        
        if instance_id not in instance_id2prediction:
            instance_id2prediction[instance_id] = {"scores": [score]}
        else:
            instance_id2prediction[instance_id]["scores"].append(score)

    input_lines.close()

    assert len(instance_id2prediction) == 1400

    pseudo_gold_lines = open("../datasets/WiC_test_pseudo_gold/output.txt").readlines()
    output = open("submit_output_" + args.epoch + "_" + args.loss + ".txt", "w")

    count = 0
    for i in range(0, 1400):
        instance_id = "WiC@test@"+str(i)
        score = sum(instance_id2prediction[instance_id]["scores"]) / float(len(instance_id2prediction[instance_id]["scores"]))
        if score >= best_t:
            if "T" in pseudo_gold_lines[i]:
                count += 1
            output.write("T\n")
        else:
            if "F" in pseudo_gold_lines[i]:
                count += 1
            output.write("F\n")

    output.close()
    print("pseudo test acc:", float(count) / 1400.0)


# python ../BERT_model/evaluate_WiC.py --loss CrossEntropy --epoch 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default="10", type=str, help="Which epoch to evaluate.")
    parser.add_argument("--loss", default="CrossEntropy", type=str, help="Which loss function to evaluate (CrossEntropy, CosineSimilarity, Contrastive).")
    args = parser.parse_args()

    if args.loss == "CrossEntropy":
        args.start_t = 0.5
        args.step_size = 0.01
        args.max_steps = 30

    
    eval_prediction(args)
    


