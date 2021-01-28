import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs
import re
from collections import Counter
import json
import math
from transformers import XLMRobertaTokenizer

'''
lang_id={"sanwu":0,"tamil":1,"mal":2,"kan":3}
'''

# 去除emoji
def demoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U00010000-\U0010ffff"
                               "]+", flags=re.UNICODE)
    return (emoji_pattern.sub(r'', text))


def merge_all_train(train_file_path_list, output_file_path, label_path, label_weight_path,label_freq_path):
    all_label_dict = {}

    idx=1
    for path in train_file_path_list:
        train_df = pd.read_csv(path, encoding="utf-8", header=None, sep="\t", names=["text", "label", "NaN"])
        train_df["text"] = train_df["text"].astype(str)
        train_df["text"] = train_df["text"].apply(lambda x: demoji(x))
        train_df = train_df[["text", "label"]]
        lang_id=[idx if x not in ["not-Tamil","not-malayalam","not-Kannada"] else 0 for x in train_df["label"]]
        idx+=1
        train_df["lang_id"]=lang_id
        print(output_file_path)
        train_df.to_csv(output_file_path, mode="a", encoding="utf-8", header=None, sep="\t", index=False)

        # 处理label
        label_np = train_df["label"].values
        label_dict = Counter(label_np)

        for label, count in label_dict.items():
            if label not in all_label_dict:
                all_label_dict[label] = count
            else:
                all_label_dict[label] += count

    total_num = sum([count for label, count in all_label_dict.items()])
    all_label_weight = {label: math.log(total_num / count) for label, count in all_label_dict.items()}
    all_label_weight["Not-in-indented-language"] = math.log( \
        total_num / (all_label_dict["not-Tamil"] + all_label_dict["not-Kannada"] + all_label_dict["not-malayalam"]))

    all_label_freq={label:(1.0*count/total_num) for label,count in all_label_dict.items()}
    all_label_freq["Not-in-indented-language"]=1.0*(all_label_dict["not-Tamil"] + all_label_dict["not-Kannada"] + all_label_dict["not-malayalam"])/total_num

    with codecs.open(label_path, "w", encoding="utf-8") as f:
        json.dump(all_label_dict, f)
    with codecs.open(label_weight_path, "w", encoding="utf-8") as f:
        json.dump(all_label_weight, f)
    with codecs.open(label_freq_path,"w",encoding="utf-8") as f:
        json.dump(all_label_freq,f)

def merge_all_dev(dev_file_path_list,output_file_path):

    idx=0
    for path in dev_file_path_list:
        dev_df=pd.read_csv(path,sep="\t",encoding="utf-8",header=None,names=["text","label"])
        lang_id=[idx if x not in ["not-Tamil","not-malayalam","not-Kannada"] else 0 for x in dev_df["label"]]
        idx+=1
        dev_df["lang_id"]=lang_id

        dev_df.to_csv(output_file_path,sep="\t",mode="a",encoding="utf-8",header=None,index=False)


def merge_train_dev(train_file_path,dev_file_path,output_file_path):

    train_df=pd.read_csv(train_file_path,sep="\t",encoding="utf-8",header=None)
    train_df.to_csv(output_file_path,mode="a",sep="\t",encoding="utf-8",header=None,index=False)
    dev_df=pd.read_csv(dev_file_path,sep="\t",encoding="utf-8",header=None)
    dev_df.to_csv(output_file_path,mode="a",sep="\t",encoding="utf-8",header=None,index=False)


def longest_text(data_path,name="train"):
    tokenizer=XLMRobertaTokenizer(vocab_file="/Users/codewithzichao/Desktop/competitions/EACL2021/xlm-roberta-base/sentencepiece.bpe.model")
    final_data=dict()

    train_df = pd.read_csv(data_path, encoding="utf-8", header=None, sep="\t", names=["text", "label","NaN"])
    text = train_df["text"].values
    for item in text:
        input_ids=tokenizer.tokenize(item)
        if len(input_ids) in final_data:
            final_data[len(input_ids)]+=1
        else:
            final_data[len(input_ids)]=1

    final_data=dict(sorted(final_data.items(),key=lambda x:x[0],reverse=True))
    with codecs.open("%s_longest.json"%name,"w",encoding="utf-8") as f:
        json.dump(final_data,f,ensure_ascii=False,indent=4)


def convert_label_to_id(label_path):
    with codecs.open(label_path, "r", encoding="utf-8") as f:
        label_dict = json.load(f)

    final_label_dict = {}
    for idx, (label, count) in enumerate(label_dict.items()):
        final_label_dict[label] = idx

    final_label_dict["not-Tamil"] = 1
    final_label_dict["not-malayalam"] = 1
    final_label_dict["not-Kannada"] = 1
    final_label_dict["Not-in-indented-language"] = 1

    return final_label_dict


def process_dev_data(dev_path, output_path):
    dev_df = pd.read_csv(dev_path, sep="\t", encoding="utf-8", header=None, names=["text", "label", "NaN"])
    dev_df = dev_df[["text", "label"]]
    dev_df["text"] = dev_df["text"].astype(str)
    dev_df["text"] = dev_df["text"].apply(lambda x: demoji(x))

    dev_df.to_csv(output_path, sep="\t", encoding="utf-8", header=None, index=False)


def split_train_dev(data,dev_num=5000):

    np.random.shuffle(data)

    dev_data=data[:dev_num]
    train_data=data[dev_num:]

    return train_data,dev_data


if __name__ == "__main__":
    base_path = "/Users/codewithzichao/Desktop/competitions/EACL2021"

    tamil_train_path = base_path + "/data/tamil_offensive_full_train.csv"
    tamil_dev_path = base_path + "/data/tamil_offensive_full_dev.csv"

    mal_train_path = base_path + "/data/mal_full_offensive_train.csv"
    mal_dev_path = base_path + "/data/mal_full_offensive_dev.csv"

    kan_train_path = base_path + "/data/kannada_offensive_train.csv"
    kan_dev_path = base_path + "/data/kannada_offensive_dev.csv"

    total_train_path = base_path + "/data/train.csv"
    tamil_dev_path_output = base_path + "/data/tamil_dev.csv"
    mal_dev_path_output = base_path + "/data/mal_dev.csv"
    kan_dev_path_output = base_path + "/data/kan_dev.csv"

    label_path = base_path + "/data/label.json"
    label_weight_path = base_path + "/data/label_weight.json"
    label_freq_path=base_path+"/data/label_freq.json"

    train_file_path_list = [tamil_train_path, mal_train_path, kan_train_path]
    merge_all_train(train_file_path_list,total_train_path,label_path,label_weight_path,label_freq_path)

    label_dict = convert_label_to_id(label_path)
    print(label_dict)

    process_dev_data(tamil_dev_path, tamil_dev_path_output)
    process_dev_data(mal_dev_path, mal_dev_path_output)
    process_dev_data(kan_dev_path, kan_dev_path_output)

    dev_list=[tamil_dev_path_output,mal_dev_path_output,kan_dev_path_output]
    all_dev_output=base_path+"/data/dev.csv"
    merge_all_dev(dev_list,all_dev_output)

    final_data_path=base_path+"/data/data.csv"
    merge_train_dev(total_train_path,all_dev_output,final_data_path)
