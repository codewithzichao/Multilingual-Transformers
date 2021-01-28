import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel, XLMRobertaModel


class MyModel(nn.Module):
    def __init__(self, model_name, bert_path, num_class, lang_id_task=False, lang_class=4, requires_grad=False):
        super(MyModel, self).__init__()

        self.model_name = model_name
        self.bert_path = bert_path
        self.num_class = num_class
        self.lang_id_task = lang_id_task
        self.lang_class = lang_class
        self.requires_grad = requires_grad

        if self.model_name == "xlm-bert":
            self.bert = BertModel.from_pretrained(self.bert_path)
        elif self.model_name == "xlm-roberta":
            self.bert = XLMRobertaModel.from_pretrained(self.bert_path)
        else:
            raise NotImplementedError

        for name, params in self.bert.named_parameters():
            if "emb" in name:
                params.requires_grad = True
            else:
                params.requires_grad = self.requires_grad

        self.dropout=nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=self.bert.config.hidden_size, out_features=self.num_class)
        self.avg_pool_layer=nn.AdaptiveAvgPool1d(output_size=1)
        self.lang_fc = nn.Linear(in_features=self.bert.config.hidden_size, out_features=self.lang_class)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):


        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding=self.avg_pool_layer(bert_output[0].permute(0,2,1)).squeeze(dim=-1)
        #cls_embedding = bert_output[0][:, 0, :].squeeze(dim=1)
        cls_embedding=self.dropout(cls_embedding)
        x = self.fc(cls_embedding)

        if self.lang_id_task is True:
            lang_logits = self.lang_fc(cls_embedding)
        else:
            lang_logits = None

        return x, lang_logits


class MyModel_origin(nn.Module):
    def __init__(self, model_name, bert_path, num_class, lang_id_task=False, lang_class=4, requires_grad=False):
        super(MyModel_origin, self).__init__()

        self.model_name = model_name
        self.bert_path = bert_path
        self.num_class = num_class
        self.lang_id_task = lang_id_task
        self.lang_class = lang_class
        self.requires_grad = requires_grad

        if self.model_name == "xlm-bert":
            self.bert = BertModel.from_pretrained(self.bert_path)
        elif self.model_name == "xlm-roberta":
            self.bert = XLMRobertaModel.from_pretrained(self.bert_path)
        else:
            raise NotImplementedError

        for name, params in self.bert.named_parameters():
            if "emb" in name:
                params.requires_grad = True
            else:
                params.requires_grad = self.requires_grad

        self.dropout=nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=self.bert.config.hidden_size, out_features=self.num_class)
        #self.avg_pool_layer=nn.AdaptiveAvgPool1d(output_size=1)
        self.lang_fc = nn.Linear(in_features=self.bert.config.hidden_size, out_features=self.lang_class)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):


        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #cls_embedding=self.avg_pool_layer(bert_output[0].permute(0,2,1)).squeeze(dim=-1)
        cls_embedding = bert_output[0][:, 0, :].squeeze(dim=1)
        cls_embedding=self.dropout(cls_embedding)
        x = self.fc(cls_embedding)

        if self.lang_id_task is True:
            lang_logits = self.lang_fc(cls_embedding)
        else:
            lang_logits = None

        return x, lang_logits

class FGM(object):
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emd_name="emb"):
        for name, params in self.model.named_parameters():
            if params.requires_grad is True and emd_name in name:
                self.backup[name] = params.data.clone()
                norm = torch.norm(params.grad)
                if norm != 0:
                    r_at = epsilon * params.grad / norm
                    params.grad.add_(r_at)

    def restore(self, emd_name="emb"):
        for name, params in self.model.named_parameters():
            if params.requires_grad is True and emd_name in name:
                assert name in self.backup
                params.data = self.backup[name]

        self.backup = {}


if __name__ == "__main__":
    bert_path = "/Users/codewithzichao/Desktop/competitions/EACL2021/bert-base-multilingual-cased/"
    num_class = 2

    my_model = MyModel(bert_path, num_class)

    for name, params in my_model.named_parameters():
        if "emb" in name:
            print(name)
