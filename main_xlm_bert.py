import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from preprocessing import *
from gen_data import MyDataset
from model import MyModel, FGM
from train import Trainer
import numpy as np
import pandas as pd
import json
from transformers import BertTokenizer
from tensorboardX import SummaryWriter
import argparse
from loss import MyLoss,FocalLoss
from sklearn.model_selection import StratifiedKFold
from scheduler import WarmUp_LinearDecay
from apex import amp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--base_path", default="/Users/codewithzichao/Desktop/competitions/EACL2021", type=str,
                    help="please input base_path", required=False)
parser.add_argument("--lang_id_task",default=False,type=bool,help="whether to add language identification task",required=False)
parser.add_argument("--mmi",default=False,type=bool,help="whether to add mmi loss",required=False)
parser.add_argument("--batch_size", default=4, type=int, help="please input batch size", required=False)
parser.add_argument("--epochs", default=50, type=int, help="please input epoch", required=False)
parser.add_argument("--logfile", default="logfile/1", type=str, help="please input logfile", required=False)
args = parser.parse_args()

# 数据路径
base_path = args.base_path
total_train_path = base_path + "/data/train.csv"
tamil_dev_path_output = base_path + "/data/tamil_dev.csv"
mal_dev_path_output = base_path + "/data/mal_dev.csv"
kan_dev_path_output = base_path + "/data/kan_dev.csv"
total_dev_path = base_path + "/data/dev.csv"
data_path=base_path+"/data/data.csv"


label_path = base_path + "/data/label.json"
label_weight_path = base_path + "/data/label_weight.json"
label_freq_path = base_path + "/data/label_freq.json"

train_data = pd.read_csv(total_train_path, sep="\t", encoding="utf-8").values
tamil_dev_data = pd.read_csv(tamil_dev_path_output, sep="\t", encoding="utf-8").values
mal_dev_data = pd.read_csv(mal_dev_path_output, sep="\t", encoding="utf-8").values
kan_dev_data = pd.read_csv(kan_dev_path_output, sep="\t", encoding="utf-8").values
dev_data = pd.read_csv(total_dev_path, sep="\t", encoding="utf-8").values

label_dict = convert_label_to_id(label_path)

#---------混合数据分割
data=pd.read_csv(data_path,sep="\t",encoding="utf-8").values
train_data,dev_data=split_train_dev(data)
#---------混合数据分割

###计算label weight
label_weight = json.load(codecs.open(label_weight_path, "r", encoding="utf-8"))
label_weight_list = list(label_weight.items())
final_weight = [label_weight_list[0][-1], label_weight_list[-1][-1], \
                label_weight_list[2][-1], label_weight_list[3][-1], \
                label_weight_list[4][-1], label_weight_list[5][-1]]

final_weight = np.array(final_weight).astype("float32")
final_weight = torch.from_numpy(final_weight)
final_weight = torch.FloatTensor(final_weight).to(device)
###计算label weight

###计算prior
temp = json.load(codecs.open(label_freq_path, "r", encoding="utf-8"))
temp = list(temp.items())
temp = [temp[0][-1], temp[-1][-1], \
        temp[2][-1], temp[3][-1], \
        temp[4][-1], temp[5][-1]]
prior = np.array(temp).astype("float32")
prior = torch.from_numpy(prior)
prior.requires_grad = False
prior = torch.FloatTensor(prior).to(device)
###计算prior

# bert文件路径
bert_path = base_path + "/bert-base-multilingual-cased/"
bert_vocab_path = base_path + "/bert-base-multilingual-cased/vocab.txt"

# tokenizer
tokenizer = BertTokenizer(vocab_file=bert_vocab_path)

# 生成batch data
train_data = MyDataset(train_data, label_dict, tokenizer, max_length=432)
dev_data = MyDataset(dev_data, label_dict, tokenizer, max_length=432)

# 生成loader
batch_size = args.batch_size
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, drop_last=True)

# 模型
num_class = 6
epochs = args.epochs
mymodel = MyModel(model_name="xlm-bert",bert_path=bert_path, num_class=num_class,lang_id_task=args.lang_id_task, requires_grad=True)

# 定义优化器
optimizer = optim.AdamW(mymodel.parameters(),lr=2e-5)
warm_up_scheduler=CosineAnnealingLR(optimizer,T_max=(epochs//4)+1,eta_min=1e-8)

loss_fn = FocalLoss(weight=final_weight)
save_path = base_path + "/ckpt/xlm_bert_best.ckpt"

writer = SummaryWriter(args.logfile)
max_norm = 0.25
eval_step_interval = 400

trainer = Trainer(model=mymodel, fgm=FGM, train_loader=train_loader, dev_loader=dev_loader, test_loader=None, \
                  optimizer=optimizer, scheduler=warm_up_scheduler, loss_fn=loss_fn, save_path=save_path, epochs=epochs, \
                  writer=writer, max_norm=max_norm, eval_step_interval=eval_step_interval, device=device)

print("start training")
trainer.train()
print("end training")
print("dev set best f1:", trainer.best_f1)
