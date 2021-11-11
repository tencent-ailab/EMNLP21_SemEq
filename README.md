# Connect-the-Dots: Bridging Semantics between Words and Definitions via Aligning Word Sense Inventories

This repo is the code release of EMNLP 2021 conference paper "Connect-the-Dots: Bridging Semantics between Words and Definitions via Aligning Word Sense Inventories".

## 1. install python environment.
Follow the instruction of "env_install.txt" to create python virtual environment and install necessary packages. The environment is tested on python >=3.6 and pytorch >=1.8. 

## 2. Gloss alignment algorithm.
Change your dictionary data format into the data format of "wordnet_def.txt" in "data/". Run the following commands to get gloss alignment results.
```bash
cd run_align_definitions_main/
python ../model/align_definitions_main.py
```

## 3. Download the pretrained model.
Download the pretrained model (SemEq-General-Large which is based on Roberta-Large) and put it under run_robertaLarge_model_span_WSD_twoStageTune/ and also run_robertaLarge_model_span_FEWS_twoStageTune/. Please make sure that the downloaded model file name is "pretrained_model_CrossEntropy.pt".
The script will load the general model and fine-tune on specific WSD datasets to get the expert model.

## 4. Fine-tune the general model to get an expert model (SemEq-Expert-Large).
### All-words WSD:
```bash
cd run_robertaLarge_model_span_WSD_twoStageTune/
python ../BERT_model_span/BERT_model_main.py --gpu_id 0 --prepare_data True --eval_dataset WSD --exp_mode twoStageTune --optimizer AdamW --learning_rate 2e-6 --bert_model roberta_large --batch_size 16
```

### Few-shot WSD (FEWS):
```bash
cd run_robertaLarge_model_span_FEWS_twoStageTune/
python ../BERT_model_span/BERT_model_main.py --gpu_id 0 --prepare_data True --eval_dataset FEWS --exp_mode twoStageTune --optimizer AdamW --learning_rate 5e-6 --bert_model roberta_large --batch_size 16
```


## 5. Evaluate results.
### All-words WSD: (you can try different epochs)
```bash
cd run_robertaLarge_model_span_WSD_twoStageTune/
python ../evaluate/evaluate_WSD.py --loss CrossEntropy --epoch 1
python ../evaluate/evaluate_WSD_POS.py
```

### Few-shot WSD (FEWS): (you can try different epochs)
```bash
cd run_robertaLarge_model_span_FEWS_twoStageTune/
python ../evaluate/evaluate_FEWS.py --loss CrossEntropy --epoch 1
```
Note that the best results of test set on few-shot setting or zero-shot setting are selected based on dev set across epochs, respectively.

## Extra. Apply the trained model to any given sentences to do WSD.
After training, you can apply the trained model (trained_model_CrossEntropy.pt) to any sentences. Examples are included in data_custom/. Examples are based on glosses in WordNet3.0.
```bash
cd run_BERT_model_span_CustomData/
python ../BERT_model_span/BERT_model_main.py --gpu_id 0 --prepare_data True --eval_dataset custom_data --exp_mode eval --bert_model roberta_large --batch_size 16
```

If you think this repo is useful, please cite our work. Thanks!

```
@inproceedings{yao-etal-2021-connect,
    title = "Connect-the-Dots: Bridging Semantics between Words and Definitions via Aligning Word Sense Inventories",
    author = "Yao, Wenlin  and
      Pan, Xiaoman  and
      Jin, Lifeng  and
      Chen, Jianshu  and
      Yu, Dian  and
      Yu, Dong",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.610",
    pages = "7741--7751",
}
```
