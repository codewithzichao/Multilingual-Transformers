import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gen_data import MyTestDataset
from preprocessing import demoji, convert_label_to_id
import numpy as np
import pandas as pd
from transformers import XLMRobertaTokenizer, XLMTokenizer,BertTokenizer
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from model import MyModel
import codecs
from tqdm import tqdm
import torch.nn.functional as F

def convert_to_context(data_path):
    data_df = pd.read_csv(data_path, encoding="utf-8", header=None, sep="\t", names=["text", "label", "NaN"])
    train_data = data_df[["text", "label"]].values
    return train_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试集数据路径
base_path = "/root/off_eacl2021"
tamil_test_path = base_path + "/test_data/tamil_offensive_full_test.csv"
mal_test_path = base_path + "/test_data/mal_full_offensive_test.csv"
kan_test_path = base_path + "/test_data/kannada_offensive_test.csv"

# 测试集预测结果路径
# for xlm-bert
bert_tamil_test_pred_path = base_path + "/pred_data/xlm_bert/tamil_pred.csv"
bert_mal_test_pred_path = base_path + "/pred_data/xlm_bert/mal_pred.csv"
bert_kan_test_pred_path = base_path + "/pred_data/xlm_bert/kan_pred.csv"

# for xlm-roberta
roberta_tamil_test_pred_path = base_path + "/pred_data/xlm-roberta/tamil_pred.csv"
roberta_mal_test_pred_path = base_path + "/pred_data/xlm-roberta/mal_pred.csv"
roberta_kan_test_pred_path = base_path + "/pred_data/xlm-roberta/kan_pred.csv"

# for both
all_tamil_test_pred_path = base_path + "/pred_data/mean/tamil_pred.csv"
all_mal_test_pred_path = base_path + "/pred_data/mean/mal_pred.csv"
all_kan_test_pred_path = base_path + "/pred_data/mean/kan_pred.csv"

# 标签路径
label_path = base_path + "/data/label.json"
label_weight_path = base_path + "/data/label_weight.json"
label_freq_path = base_path + "/data/label_freq.json"

# 得到标签
label2idx = convert_label_to_id(label_path)
label2idx_tamil = {'Not_offensive': 0, 'not-Tamil': 1, 'Offensive_Targeted_Insult_Other': 2, \
                   'Offensive_Targeted_Insult_Group': 3, 'Offensive_Untargetede': 4, \
                   'Offensive_Targeted_Insult_Individual': 5}
label2idx_mal = {'Not_offensive': 0, 'Offensive_Targeted_Insult_Other': 2, \
                 'Offensive_Targeted_Insult_Group': 3, 'Offensive_Untargetede': 4, \
                 'Offensive_Targeted_Insult_Individual': 5, 'not-malayalam': 1}
label2idx_kan = {'Not_offensive': 0, 'Offensive_Targeted_Insult_Other': 2, \
                 'Offensive_Targeted_Insult_Group': 3, 'Offensive_Untargetede': 4, \
                 'Offensive_Targeted_Insult_Individual': 5, 'not-Kannada': 1}

# 得到numpy型的数据
tamil_data_origin = convert_to_context(tamil_test_path)
mal_data_origin = convert_to_context(mal_test_path)
kan_data_origin = convert_to_context(kan_test_path)

# 预训练模型路径
# for roberta
roberta_path = base_path + "/pretrained_weights/xlm-roberta-base/"
roberta_vocab_path = roberta_path + "sentencepiece.bpe.model"
roberta_tokenizer = XLMRobertaTokenizer(vocab_file=roberta_vocab_path)

# for bert
bert_path = base_path + "/pretrained_weights/bert-base-multilingual-cased/"
bert_vocab_path = bert_path + "vocab.txt"
tokenizer = BertTokenizer(vocab_file=bert_vocab_path)

# 构建数据
# for bert
bert_tamil_data = MyTestDataset(tamil_data_origin, label2idx, tokenizer)
bert_mal_data = MyTestDataset(mal_data_origin, label2idx, tokenizer)
bert_kan_data = MyTestDataset(kan_data_origin, label2idx, tokenizer)

# for roberta
roberta_tamil_data = MyTestDataset(tamil_data_origin, label2idx, roberta_tokenizer)
roberta_mal_data = MyTestDataset(mal_data_origin, label2idx, roberta_tokenizer)
roberta_kan_data = MyTestDataset(kan_data_origin, label2idx, roberta_tokenizer)

# 构建dataloader
# for bert
bert_tamil_loader = DataLoader(bert_tamil_data, batch_size=1, shuffle=False)
bert_mal_loader = DataLoader(bert_mal_data, batch_size=1, shuffle=False)
bert_kan_loader = DataLoader(bert_kan_data, batch_size=1, shuffle=False)

# for roberta
roberta_tamil_loader = DataLoader(roberta_tamil_data, batch_size=1, shuffle=False)
roberta_mal_loader = DataLoader(roberta_mal_data, batch_size=1, shuffle=False)
roberta_kan_loader = DataLoader(roberta_kan_data, batch_size=1, shuffle=False)

# 构建模型
roberta_model_ckpt = base_path + "/roberta_new_ckpt_12/xlm_roberta_best.ckpt"
bert_model_ckpt = base_path + "/bert_best_ckpt_12/xlm_bert_best.ckpt"

# for roberta
roberta_model = MyModel(model_name="xlm-roberta", bert_path=roberta_path, num_class=6)
roberta_model.load_state_dict(torch.load(roberta_model_ckpt, map_location=device)["model"], strict=False)
roberta_model.to(device)

# for bert
bert_model = MyModel(model_name="xlm-bert", bert_path=bert_path, num_class=6)
bert_model.load_state_dict(torch.load(bert_model_ckpt, map_location=device)["model"], strict=False)
bert_model.to(device)


#######################################################################################################
# 构建函数
def test_and_to_tsv(model, test_loader, tsv_path, name="tamil"):
    model.eval()

    if name == "tamil":
        label2idx = label2idx_tamil
    elif name == "mal":
        label2idx = label2idx_mal
    elif name == "kan":
        label2idx = label2idx_kan

    idx2label = {id: label for label, id in label2idx.items()}
    y_pred = []
    y_true = []
    origin_datas = []
    prob = []
    with torch.no_grad():
        for idx, batch_data in tqdm(enumerate(test_loader)):
            input_ids, attention_mask, token_type_ids, label, origin_data = batch_data[0], batch_data[1], batch_data[2], \
                                                                            batch_data[3], batch_data[4]
            logits, _ = model(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device))

            if torch.cuda.is_available():
                y_true.extend(label.cpu().numpy())
                logits = logits.cpu()
            else:
                y_true.extend(label)
                logits = logits
            for item in logits:
                y_pred.append(np.argmax(item))

            origin_datas.append(origin_data)

            prob.append(F.softmax(logits, dim=-1).numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    p = precision_score(y_true, y_pred, average="weighted")
    r = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    cls_report = classification_report(y_true, y_pred)
    print("precision:{p},recall:{r},f1:{f1}".format(p=p, r=r, f1=f1))
    print(cls_report)

    str_format = "{id}\t{text}\t{label}\n"
    with codecs.open(tsv_path, "w", encoding="utf-8") as f:
        f.write("id\ttext\tlabel\n")
        for idx, item in tqdm(enumerate(y_pred)):
            f.write(str_format.format(id="%d" % (idx + 1), text=origin_datas[idx][0], \
                                      label=idx2label[y_pred[idx]]))

        f.flush()
        f.close()

    return prob, y_true


def merge_all_test_and_to_tsv(bert_prob, roberta_prob, test_loader, y_true, tsv_path, name="tamil"):
    if name == "tamil":
        label2idx = label2idx_tamil
    elif name == "mal":
        label2idx = label2idx_mal
    elif name == "kan":
        label2idx = label2idx_kan

    idx2label = {id: label for label, id in label2idx.items()}

    y_pred = []
    y_true = y_true
    assert len(bert_prob)==len(roberta_prob)
    for i in range(len(bert_prob)):
        temp = (bert_prob[i] + roberta_prob[i]) / 2.0
        y_pred.append(np.argmax(temp))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    origin_datas = []
    for idx, batch_data in tqdm(enumerate(test_loader)):
        input_ids, attention_mask, token_type_ids, label, origin_data = batch_data[0], batch_data[1], batch_data[2], \
                                                                        batch_data[3], batch_data[4]

        origin_datas.append(origin_data)

    p = precision_score(y_true, y_pred, average="weighted")
    r = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    cls_report = classification_report(y_true, y_pred)
    print("precision:{p},recall:{r},f1:{f1}".format(p=p, r=r, f1=f1))
    print(cls_report)

    str_format = "{id}\t{text}\t{label}\n"
    with codecs.open(tsv_path, "w", encoding="utf-8") as f:
        f.write("id\ttext\tlabel\n")
        for idx, item in tqdm(enumerate(y_pred)):
            f.write(str_format.format(id="%d" % (idx + 1), text=origin_datas[idx][0], \
                                      label=idx2label[y_pred[idx]]))

        f.flush()
        f.close()

#######################################################################################################
print("start testing tamil data!")
print("test in xlm-bert!")
bert_tamil_prob, bert_tamil_y_true = test_and_to_tsv(bert_model, bert_tamil_loader, bert_tamil_test_pred_path, \
                                                     name="tamil")
print("test in xlm-roberta!")
roberta_tamil_prob, roberta_tamil_y_true = test_and_to_tsv(roberta_model, roberta_tamil_loader, \
                                                           roberta_tamil_test_pred_path, name="tamil")
print("test in both mean!")
merge_all_test_and_to_tsv(bert_tamil_prob, roberta_tamil_prob, roberta_tamil_loader, bert_tamil_y_true, \
                          all_tamil_test_pred_path,name="tamil")
print("end testing tamil data!")
print("#" * 50)

#######################################################################################################
print("start testing mal data!")
print("test in xlm-bert!")
bert_mal_prob, bert_mal_y_true = test_and_to_tsv(bert_model, bert_mal_loader, bert_mal_test_pred_path, name="mal")
print("test in xlm-roberta!")
roberta_mal_prob, roberta_mal_y_true = test_and_to_tsv(roberta_model, roberta_mal_loader, roberta_mal_test_pred_path, \
                                                       name="mal")
print("test in both mean!")
merge_all_test_and_to_tsv(bert_mal_prob, roberta_mal_prob, roberta_mal_loader, bert_mal_y_true, \
                          all_mal_test_pred_path,name="mal")
print("end testing mal data!")
print("#" * 50)

#######################################################################################################
print("start testing kan data!")
print("test in xlm-bert!")
bert_kan_prob, bert_kan_y_true = test_and_to_tsv(bert_model, bert_kan_loader, bert_kan_test_pred_path, name="kan")
print("test in xlm-roberta!")
roberta_kan_prob, roberta_kan_y_true = test_and_to_tsv(roberta_model, roberta_kan_loader, \
                                                       roberta_kan_test_pred_path, name="kan")
print("test in both mean!")
merge_all_test_and_to_tsv(bert_kan_prob, roberta_kan_prob, roberta_kan_loader, bert_kan_y_true, \
                          all_kan_test_pred_path, name="kan")
print("end testing kan data!")
print("#" * 50)

#######################################################################################################
print("finish all things!lucky!")
