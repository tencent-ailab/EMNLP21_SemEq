import pickle
import sys
sys.path.append("../utilities/")
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment
from nltk.corpus import stopwords
import numpy as np
from torch.multiprocessing import Process, set_start_method
from utilities import extract_valid_POS, load_dictionary

#from multiprocessing import Process

try:
    set_start_method('spawn')
except RuntimeError:
    pass


# nltk.download('stopwords')
nltk_stopwords = set(stopwords.words('english'))


def extract_def_texts(definitions, dict_name):
    pos2def_textList = {}
    for idx, definition in enumerate(definitions):
        if " #idiom# " not in definition["definition"]:
            valid_POSList = extract_valid_POS(definition, dict_name)
            def_text = definition["definition"].split(" ### ")[0]
            for pos in valid_POSList:
                if pos not in pos2def_textList:
                    pos2def_textList[pos] = [[idx, def_text]]
                else:
                    pos2def_textList[pos] += [[idx, def_text]]
    return pos2def_textList


def extract_all_POS(keyword2definitions):
    all_POS = []
    for keyword in keyword2definitions:
        if keyword in nltk_stopwords:
            continue
        definitions = keyword2definitions[keyword]
        for definition in definitions:
            all_POS.append(definition["POS"])
    all_POS = set(all_POS)
    return all_POS


def align_two_dict(model, dict_nameList, keyword2definitionsList, i, j):
    print(dict_nameList[i] + " v.s. " + dict_nameList[j])

    common_keywords = list(set(keyword2definitionsList[i].keys()) & set(
        keyword2definitionsList[j].keys()))
    print("len(common_keywords):", len(common_keywords))
    common_keywords.sort()

    keyword_pos2def_alignments = {}

    for keyword in tqdm(common_keywords, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
        pos2def_textList1 = extract_def_texts(
            keyword2definitionsList[i][keyword], dict_nameList[i])
        pos2def_textList2 = extract_def_texts(
            keyword2definitionsList[j][keyword], dict_nameList[j])

        common_pos = set(pos2def_textList1.keys()) & set(
            pos2def_textList2.keys())

        for pos in common_pos:
            keyword_pos = keyword+"@@@"+pos
            if keyword_pos not in keyword_pos2def_alignments:
                keyword_pos2def_alignments[keyword_pos] = []

            sentences1 = pos2def_textList1[pos]
            sentences2 = pos2def_textList2[pos]

            # Compute embedding for both lists
            embeddings1 = model.encode(
                [s for _, s in sentences1], convert_to_tensor=True)
            embeddings2 = model.encode(
                [s for _, s in sentences2], convert_to_tensor=True)
            # Compute cosine-similarits
            cosine_scores = util.pytorch_cos_sim(
                embeddings1, embeddings2).cpu()

            row_ind, col_ind = linear_sum_assignment(
                np.array(cosine_scores), maximize=True)

            for x in range(len(row_ind)):
                m = row_ind[x]
                n = col_ind[x]
                keyword_pos2def_alignments[keyword_pos].append(
                    {"score": cosine_scores[m][n].item(),
                    "def1_id": dict_nameList[i]+":"+keyword+":"+str(sentences1[m][0]),
                    "def2_id": dict_nameList[j]+":"+keyword+":"+str(sentences2[n][0]),
                    "def1_text": sentences1[m][1], "def2_text": sentences2[n][1]})

    pickle.dump(keyword_pos2def_alignments, open(
        str(i)+"_"+str(j)+"_keyword_pos2def_alignments.p", "wb"))



# https://www.sbert.net/docs/usage/semantic_textual_similarity.html
if __name__ == "__main__":
    dict_nameList = ["OxfordAdvanced", "Webster", "Collins",
                     "LongmanAdvanced", "CambridgeAdvanced", "wordnet"]

    #dict_nameList = ["OxfordAdvanced", "wordnet"]

    idx_combinations = []

    for i in range(0, len(dict_nameList)):
        for j in range(i+1, len(dict_nameList)):
            # if not (i == 1 and j == 2):
            #    continue
            idx_combinations.append([i, j])

    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    keyword2definitionsList = []
    for i in range(len(dict_nameList)):
        print(dict_nameList[i])
        keyword2definitions = load_dictionary(
            "../data/extracted_data_new/" + dict_nameList[i] + "_def.txt")
        keyword2definitionsList.append(keyword2definitions)
        all_POS = extract_all_POS(keyword2definitions)
        # print(all_POS)

    keyword_pos2def_alignments = {}

    processV = []
    for x in range(0, len(idx_combinations)):
        i, j = idx_combinations[x]
        processV.append(Process(target=align_two_dict, args=(
            model, dict_nameList, keyword2definitionsList, i, j, )))
    for x in range(0, len(processV)):
        processV[x].start()
    for x in range(0, len(processV)):
        processV[x].join()

    final_keyword_pos2def_alignments = {}
    for x in range(0, len(idx_combinations)):
        print(i, j, "load keyword_pos2def_alignments.p")
        i, j = idx_combinations[x]
        keyword_pos2def_alignments = pickle.load(
            open(str(i)+"_"+str(j)+"_keyword_pos2def_alignments.p", "rb"))
        for keyword_pos in keyword_pos2def_alignments:
            if keyword_pos not in final_keyword_pos2def_alignments:
                final_keyword_pos2def_alignments[keyword_pos] = []
            final_keyword_pos2def_alignments[keyword_pos] += keyword_pos2def_alignments[keyword_pos]

    pickle.dump(final_keyword_pos2def_alignments, open(
        "final_keyword_pos2def_alignments.p", "wb"))

    output = open("keyword_pos2def_alignments.txt", "w", encoding="utf-8")
    all_keyword_pos = list(final_keyword_pos2def_alignments.keys())
    all_keyword_pos.sort()
    COUNT = 0
    for keyword_pos in all_keyword_pos:
        COUNT += 1
        # if COUNT > 1000:
        #    break
        output.write(keyword_pos + " | " +
                     str(final_keyword_pos2def_alignments[keyword_pos]) + "\n")
    output.close()
