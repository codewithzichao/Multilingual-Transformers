import os
import codecs
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from preprocessing import convert_label_to_id,demoji


class MyDataset(Dataset):
    def __init__(self, train_data, label2idx, tokenizer, max_length=512):
        super(MyDataset, self).__init__()
        self.train_data = train_data
        self.label2idx = label2idx
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        data = self.train_data[item][0]
        label = self.train_data[item][1]
        lang_id = self.train_data[item][-1]

        tokenizer_result = self.tokenizer(data, add_special_tokens=True, \
                                          max_length=self.max_length, \
                                          padding="max_length", \
                                          return_tensors='pt')

        input_ids = tokenizer_result["input_ids"].squeeze(0)
        #print(input_ids.shape)
        attention_mask = tokenizer_result["attention_mask"].squeeze(0)
        #token_type_ids = tokenizer_result["token_type_ids"].squeeze(0)

        if input_ids.shape[-1] != self.max_length:
            input_ids = input_ids[:self.max_length]
        if attention_mask.shape[-1] != self.max_length:
            attention_mask = attention_mask[:self.max_length]
        #if token_type_ids.shape[-1] != self.max_length:
        #   token_type_ids = token_type_ids[:self.max_length]

        label = self.label2idx[label]

        token_type_ids=0

        return input_ids, attention_mask, token_type_ids,label, lang_id


class MyTestDataset(Dataset):
    def __init__(self, train_data, label2idx, tokenizer, max_length=512):
        super(MyTestDataset, self).__init__()
        self.train_data = train_data
        self.label2idx = label2idx
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):

        data_origin = self.train_data[item][0]
        data = demoji(data_origin)

        label = self.train_data[item][1]
        label = self.label2idx[label]

        tokenizer_result = self.tokenizer.encode_plus(data, add_special_tokens=True, \
                                                      max_length=self.max_length, \
                                                      padding="max_length", \
                                                      return_tensors='pt')

        input_ids = tokenizer_result["input_ids"].squeeze(0)
        attention_mask = tokenizer_result["attention_mask"].squeeze(0)


        if input_ids.shape[-1] != self.max_length:
            input_ids = input_ids[:self.max_length]
        if attention_mask.shape[-1] != self.max_length:
            attention_mask = attention_mask[:self.max_length]

        token_type_ids=0
        return input_ids, attention_mask, token_type_ids, label, data_origin
