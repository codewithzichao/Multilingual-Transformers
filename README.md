# Offensive Language Identification in Dravidian Languages at EACL2021 Workshop
Our source code for EACL2021 workshop: Offensive Language Identification in Dravidian Languages. 
We ranked 4th,4th and 3th in Tamil, Malayalam and Kannada language of this task finally!🥳


**Updated:** Source code is released!🤩

> I will release the code very soon.

## Repository structure
```shell
├── README.md
├── ckpt                        # store model weights during training
│   └── README.md
├── data                        # store the data
│   └── README.md
├── gen_data.py                 # generate Dataset
├── install_cli.sh              # install required package
├── loss.py                     # loss function
├── main_xlm_bert.py            # train mulingual-BERT
├── main_xlm_roberta.py         # train XLM-RoBERTa
├── model.py                    # model implementation
├── pred_data
│   └── README.md
├── preprocessing.py            # preprocess the data
├── pretrained_weights          # store the pretrained weights
│   └── README.md
└── train.py                    # define training and validation loop
```
## Installation
Use the following command so that you can install all of required packages:
```shell
sh install_cli.sh
```

## Preprocessing
The first step is to preprocess the data. Just use the following command:
```shell
python3 -u preprocessing.py
```

## Training
The second step is to train our model. In our solution, We trained two models which use multilingual-BERT and XLM-RoBERTa as the encoder, respectively.

If you want to train model which use multilingual-BERT as the encoder, use the following command:
```shell
nohup python3 -u main_xlm_bert.py \
        --base_path your base path \
        --batch_size 8 \
        --epochs 50 \
        > train_xlm_bert_log.log 2>&1 &
```
If you want to train model which use XLM-RoBERTa as the encoder, use the following command:
```shell
nohup python3 -u main_xlm_roberta.py \
        --base_path your base path \
        --batch_size 8 \
        --epochs 50 \
        > train_xlm_roberta_log.log 2>&1 &
```

## Inference
The final step is inference after training. Use the following command:
```shell
nohup python3 -u inference.py > inference.log 2>&1 &
```
Congralutions! You have got the final results!🤩




> If you use our code, please indicate the source.