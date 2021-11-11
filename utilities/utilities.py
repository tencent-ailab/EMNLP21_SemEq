import ast

from cleantext import clean

invalid_tokens = ["&#91;", "&#93;", "[...]", "[…]", "…", "<!--", "-->"]

# https://pypi.org/project/clean-text/


def clean_text(s):
    s = clean(s,
              fix_unicode=True,               # fix various unicode errors
              to_ascii=True,                  # transliterate to closest ASCII representation
              lower=False,                     # lowercase text
              # fully strip line breaks as opposed to only normalizing them
              no_line_breaks=True,
              no_urls=True,                  # replace all URLs with a special token
              no_emails=True,                # replace all email addresses with a special token
              no_phone_numbers=True,         # replace all phone numbers with a special token
              no_numbers=False,               # replace all numbers with a special token
              no_digits=False,                # replace all digits with a special token
              no_currency_symbols=False,      # replace all currency symbols with a special token
              no_punct=False,                 # remove punctuations
              replace_with_punct="",          # instead of removing punctuations you may replace them
              replace_with_url="<URL>",
              replace_with_email="<EMAIL>",
              replace_with_phone_number="<PHONE>",
              replace_with_number="<NUMBER>",
              replace_with_digit="0",
              replace_with_currency_symbol="<CUR>",
              lang="en"                       # set to 'de' for German special handling
              )
    for invalid_token in invalid_tokens:
        s = s.replace(invalid_token, " ")
    return s


def load_dictionary(dict_file):
    input_lines = open(dict_file, 'r', encoding='utf-8')
    keyword2definitions = {}
    for line in input_lines:
        if not line.strip():
            continue
        keyword = line.split(" | ")[0]
        keyword2definitions[keyword] = ast.literal_eval(
            line.split(" | ")[1])
    input_lines.close()

    return keyword2definitions


pos_mapping = {"noun": "n", "n": "n",
               "pronoun": "pron", "pron": "pron",
               "verb": "v", "v": "v",
               "adjective": "adj", "adj": "adj",
               "adverb": "adv", "adv": "adv",
               "preposition": "prep", "prep": "prep",
               "conjunction": "conj",
               "abbreviation": "abbr",
               "prefix": "prefix",
               "suffix": "suffix"}


def extract_valid_POS(definition, dict_name):
    if dict_name in ["OxfordAdvanced", "Webster", "LongmanAdvanced", "CambridgeAdvanced", "wordnet"]:
        definition_pos_words = definition["POS"].split()
        if dict_name == "CambridgeAdvanced":
            definition_pos_words = definition["POS"].split()[:3]
        valid_POSList = []
        for p in definition_pos_words:
            pos = p.replace(",", "")
            if pos in pos_mapping:
                valid_POSList.append(pos_mapping[pos])
        if len(valid_POSList) > 0:
            if "suffix" in valid_POSList:
                return ["suffix"]
            elif "prefix" in valid_POSList:
                return ["prefix"]
            else:
                return list(set(valid_POSList))
        else:
            return ["Other"]

    elif dict_name == "Collins":
        valid_POSList = []
        for p in definition["POS"].split("-"):
            if p in pos_mapping:
                valid_POSList.append(pos_mapping[p])
        if definition["POS"] == "verb:":
            valid_POSList.append("v")

        if len(valid_POSList) > 0:
            if "suffix" in valid_POSList:
                return ["suffix"]
            elif "prefix" in valid_POSList:
                return ["prefix"]
            else:
                return list(set(valid_POSList))
        else:
            return ["Other"]
    else:
        return ["unk"]
