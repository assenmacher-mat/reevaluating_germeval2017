#!/usr/bin/env python3
import os
import sys
import json
import logging
import argparse
import time
import pickle
import torch
import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm, trange
import torch

from utils import set_all_seeds, initialize_device_settings, format_time
from data_prep import bio_tagging_df
from data_handler import (get_tags_list, get_sentences_biotags, split_train_dev, padding_token)

from modeling_token import TokenBERT, TokenDistilBERT

from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, SequentialSampler)
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          BertTokenizer, BertConfig,
                          BertModel, BertPreTrainedModel, 
                          DistilBertTokenizer, DistilBertConfig,
                          DistilBertModel, DistilBertPreTrainedModel)

from seqeval_metrics import (seq_accuracy_score, seq_f1_score, 
                             seq_classification_report)

logger = logging.getLogger(__name__)
logging.basicConfig(filename="subtaskD.log", level=logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
set_all_seeds()


def training(train_dataloader, model, device, optimizer, scheduler, max_grad_norm=1.0):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    #predictions_train, true_labels_train = [], []

    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        model.zero_grad()
        # forward pass
        loss = model(
            b_input_ids,
            attention_mask=b_input_mask, 
            labels=b_labels
        )
        
        # backward pass
        loss.backward()        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # update learning rate
        scheduler.step()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
        
    return model, optimizer, scheduler, tr_loss


def evaluation(sample_dataloader, model, device, tag_values):
    model.eval()
    true_labels, predictions, validation_loss_values, tokenized_texts = [], [], [], []
    
    for step, batch in enumerate(sample_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_label_ids = batch
        
        with torch.no_grad():
            tags = model(
                b_input_ids,
                attention_mask=b_input_mask)
        
        # Move logits and labels to CPU
        input_ids = b_input_ids.cpu().numpy()
        label_ids = b_label_ids.cpu().numpy()
        if type(tags[0])!=list:
            tags = tags.detach().cpu().numpy() # statt 1

        # Calculate the accuracy for this batch of test sentences.
        tokenized_texts.extend(input_ids)
        true_labels.extend(label_ids)
        predictions.extend(tags)
    
    dev_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    # calculate accuracy score
    dev_accuracy_score = seq_accuracy_score(dev_tags, pred_tags)
    # calculate f1 score exact & overlap
    dev_f1_score = seq_f1_score(dev_tags, pred_tags, overlap = False)
    dev_f1_score_overlap = seq_f1_score(dev_tags, pred_tags, overlap = True)
    
    return dev_tags, pred_tags, dev_f1_score, dev_f1_score_overlap


def main():
    """
    main function for conducting Subtask D. Parameters are parsed with argparse.
    Language model should be one of the following:
    (    
        'bert-base-multilingual-uncased', 
        'bert-base-multilingual-cased',              
        'bert-base-german-cased', 
        'bert-base-german-dbmdz-cased',
        'bert-base-german-dbmdz-uncased',
        'distilbert-base-german-cased',
        'distilbert-base-multilingual-cased'
    )
    """
    parser = argparse.ArgumentParser(description='Run Subtask D of GermEval 2017 Using Pre-Trained Language Model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=5e-5, help='The learning rate.')
    parser.add_argument('--max_len', type=int, default=512, help='The maximum sequence length of the input text.')
    parser.add_argument('--batch_size', type=int, default=16, help='Your train set batch size.')
    parser.add_argument('--df_path', type=str, default='./data/', help='The data directory.')    
    parser.add_argument('--train_data', type=str, default='train_df_opinion.tsv', help='The filename of the input train data.')
    parser.add_argument('--dev_data', type=str, default='dev_df_opinion.tsv', help='The filename of the input development data.')
    parser.add_argument('--test_data1', type=str, default='test_syn_df_opinion.tsv', help='The filename of the first input test data (synchronic).')
    parser.add_argument('--test_data2', type=str, default='test_dia_df_opinion.tsv', help='The filename of the second input test data (diachronic).')
    parser.add_argument('--output_path', type=str, default='./output/subtaskD/', help='The output directory of the model and predictions.')
    parser.add_argument('--config_path', type=str, default='./saved_models/subtaskD/', help='The configuration directory of the model config.')
    parser.add_argument('--lang_model', type=str, default='bert-base-german-dbmdz-uncased', help='The pre-trained language model.')
    parser.add_argument("--use_crf", default=False, action="store_true", help="Flag for CRF usage.")
    parser.add_argument("--train", default=True, action="store_true", help="Flag for training.")
    parser.add_argument("--eval", default=False, action="store_true", help="Flag for evaluation.")
    parser.add_argument("--save_model", default=False, action="store_true", help="Flag for saving.")
    parser.add_argument("--save_prediction", default=False, action="store_true", help="Flag for saving predictions.")
    parser.add_argument("--save_cr", default=False, action="store_true", help="Flag for saving classification report.")
    args = parser.parse_args()

    #############################################################################
    # Settings
    device, n_gpu = initialize_device_settings(use_cuda=True)
    set_all_seeds(args.seed)
    lm = args.lang_model
    if args.use_crf:
        lm = args.lang_model+"_crf"

    MODEL_PATH = os.path.join(args.config_path,'{}_token.pt'.format(lm))
    print(MODEL_PATH)
    CONFIG_PATH = os.path.join(args.config_path,'{}_config.json'.format(lm))
    print(CONFIG_PATH)
    PREDICTIONS_DEV = os.path.join(args.config_path,'{}_predictions_dev.json'.format(lm))
    print(PREDICTIONS_DEV)
    PREDICTIONS_TEST = os.path.join(args.config_path,'{}_predictions_test.json'.format(lm))
    print(PREDICTIONS_TEST)

    #############################################################################
    # Load and prepare data by adding BIO tags
    train_df = bio_tagging_df(pd.read_csv(args.df_path + args.train_data, delimiter = '\t'))
    dev_df = bio_tagging_df(pd.read_csv(args.df_path + args.dev_data, delimiter = '\t'))
    test_syn_df = bio_tagging_df(pd.read_csv(args.df_path + args.test_data1, delimiter = '\t'))
    test_dia_df = bio_tagging_df(pd.read_csv(args.df_path + args.test_data2, delimiter = '\t'))
    
    # 1. Create a tokenizer
    lower_case = False
    if args.lang_model[-7:] == "uncased":
        lower_case = True

    if args.lang_model[:4] == "bert":
        model_class = "BERT"
        tokenizer = BertTokenizer.from_pretrained(args.lang_model, do_lower_case = lower_case, max_length=args.max_len)
    
    if args.lang_model[:10] == "distilbert":
        model_class = "DistilBERT"
        tokenizer = DistilBertTokenizer.from_pretrained(args.lang_model, do_lower_case = lower_case, max_length=args.max_len)

    # get training features
    tokenized_texts, labels = get_sentences_biotags(tokenizer, train_df, dev_df)

    # get tag values and dictionary
    tag_values, tag2idx, entities = get_tags_list(args.df_path)
    print("entities:", entities)
    
    # pad input_ids and tags, create attention masks
    input_ids, tags = padding_token(tokenized_texts, tokenizer, args.max_len, tag2idx, labels)
    
    attention_masks= [[int(token_id > 0) for token_id in sent] for sent in input_ids]
    
    print("tokenized_texts",tokenized_texts[0])
    print("input_ids",input_ids[0])
    print("labels",labels[0])

    # split train, dev
    train_inputs, train_labels, dev_inputs, dev_labels, train_masks, dev_masks = split_train_dev(
        train_df, dev_df, attention_masks, input_ids, tags)

    # transform to torch tensor
    train_inputs = torch.tensor(train_inputs, dtype = torch.long)
    dev_inputs = torch.tensor(dev_inputs, dtype = torch.long)

    train_labels = torch.tensor(train_labels, dtype = torch.long)
    dev_labels = torch.tensor(dev_labels, dtype = torch.long)

    train_masks = torch.tensor(train_masks, dtype = torch.uint8)
    dev_masks = torch.tensor(dev_masks, dtype = torch.uint8)

    # create DataLoader
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.batch_size)
    #print(len(train_sampler), len(train_dataloader))
    #print(len(dev_sampler), len(dev_dataloader))

    # create dataLoader for test syn set
    tokenized_texts_syn, labels_syn = get_sentences_biotags(tokenizer, test_syn_df)
    
    # padding
    input_ids_syn, tags_syn = padding_token(tokenized_texts_syn, tokenizer, args.max_len, tag2idx, labels_syn)
    
    # Create attention masks
    attention_masks_syn = [[int(token_id > 0) for token_id in sent] for sent in input_ids_syn]

    # Convert to tensors
    test_syn_inputs = torch.tensor(input_ids_syn, dtype = torch.long)
    test_syn_masks = torch.tensor(attention_masks_syn, dtype = torch.uint8)
    test_syn_labels = torch.tensor(tags_syn, dtype = torch.long)
    # Create the DataLoader
    test_syn_data = TensorDataset(test_syn_inputs, test_syn_masks, test_syn_labels)
    test_syn_sampler = SequentialSampler(test_syn_data)
    test_syn_dataloader = DataLoader(test_syn_data, sampler=test_syn_sampler, batch_size=args.batch_size)
    #print(len(test_syn_sampler), len(test_syn_dataloader))

    # create dataLoader for test dia set
    tokenized_texts_dia, labels_dia = get_sentences_biotags(tokenizer, test_dia_df)
    # padding
    input_ids_dia, tags_dia = padding_token(tokenized_texts_dia, tokenizer, args.max_len, tag2idx, labels_dia)

    # Create attention masks
    attention_masks_dia = [[int(token_id > 0) for token_id in sent] for sent in input_ids_dia]

    # Convert to tensors
    test_dia_inputs = torch.tensor(input_ids_dia, dtype = torch.long)
    test_dia_masks = torch.tensor(attention_masks_dia, dtype = torch.uint8)
    test_dia_labels = torch.tensor(tags_dia, dtype = torch.long)
    # Create the DataLoader
    test_dia_data = TensorDataset(test_dia_inputs, test_dia_masks, test_dia_labels)
    test_dia_sampler = SequentialSampler(test_dia_data)
    test_dia_dataloader = DataLoader(test_dia_data, sampler=test_dia_sampler, batch_size=args.batch_size)
    #print(len(test_dia_sampler), len(test_dia_dataloader))

    #############################################################################
    # Training
    if args.train:
        # Load Config
        if model_class=="BERT":
            config = BertConfig.from_pretrained(args.lang_model, num_labels=len(tag2idx))
            config.hidden_dropout_prob = 0.1 # dropout probability for all fully connected layers
                                             # in the embeddings, encoder, and pooler; default = 0.1
            model = TokenBERT(
                model_name=args.lang_model, 
                num_labels=len(tag2idx),
                use_crf=args.use_crf)

        if model_class=="DistilBERT":
            config = DistilBertConfig.from_pretrained(args.lang_model, num_labels=len(tag2idx))   
            config.hidden_dropout_prob = 0.1       
            model = TokenDistilBERT(
                model_name=args.lang_model, 
                num_labels=len(tag2idx),
                use_crf=args.use_crf)
        
        model.cuda() 

        # Create an optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=1e-8
        )
        # Total number of training steps = number of batches * number of epochs
        total_steps = len(train_dataloader) * args.epochs
        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Main Loop
        print("=================== Train ================")
        print("##### Language Model:", args.lang_model, ",", "use CRF:", args.use_crf, ",", "learning rate:", args.lr, ",", "DROPOUT:", config.hidden_dropout_prob)
        print()

        track_time = time.time()
                
        for epoch in trange(args.epochs, desc="Epoch"):
            print("Epoch: %4i"%epoch, dt.datetime.now())
            
            # TRAINING
            model, optimizer, scheduler, tr_loss = training(
                train_dataloader, 
                model=model, 
                device=device, 
                optimizer=optimizer, 
                scheduler=scheduler
                )
            
            # EVALUATION: TRAIN SET
            y_true_train, y_pred_train, f1s_train, f1s_overlap_train = evaluation(
                    train_dataloader, model=model, device=device, tag_values=tag_values)
            print("TRAIN: F1 Exact %.3f | F1 Overlap %.3f"%(f1s_train, f1s_overlap_train))
            
            # EVALUATION: DEV SET
            y_true_dev, y_pred_dev, f1s_dev, f1s_overlap_dev = evaluation(
                    dev_dataloader, model=model, device=device, tag_values=tag_values)
            print("EVAL: F1 Exact %.3f | F1 Overlap %.3f"%(f1s_dev, f1s_overlap_dev))
        
        print("  Training and validation took in total: {:}".format(format_time(time.time()-track_time)))

        # EVALUATION: TEST SYN SET
        y_true_test_syn, y_pred_test_syn, f1s_test_syn, f1s_overlap_test_syn = evaluation(
                test_syn_dataloader, model=model, device=device, tag_values=tag_values)
        print("TEST SYN: F1 Exact %.3f | F1 Overlap %.3f"%(f1s_test_syn, f1s_overlap_test_syn))
                
        # EVALUATION: TEST DIA SET
        y_true_test_dia, y_pred_test_dia, f1s_test_dia, f1s_overlap_test_dia = evaluation(
                test_dia_dataloader, model=model, device=device, tag_values=tag_values)
        print("TEST DIA: F1 Exact %.3f | F1 Overlap %.3f"%(f1s_test_dia, f1s_overlap_test_dia))
        
        
        if args.save_model:
            # Save Model
            torch.save( model.state_dict(), MODEL_PATH )
            # Save Config
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config.to_json_string(), f, sort_keys=True, indent=4, separators=(',', ': '))
                    
        if args.save_prediction:
            # SAVE PREDICTED DATA
            # DEV
            DATA_DEV = dev_df # as input tokens # get_data_with_labels(all_input_tokens_dev, y_true_dev, y_pred_dev)
            with open(PREDICTIONS_DEV, 'w') as f:
                json.dump(DATA_DEV, f, sort_keys=True, indent=4, separators=(',', ': '))
            # TEST SYN
            DATA_TEST_SYN = test_syn_df # as input tokens #get_data_with_labels(all_input_tokens_test, y_true_test, y_pred_test)
            with open(PREDICTIONS_TEST, 'w') as f:
                json.dump(DATA_TEST_SYN, f, sort_keys=True, indent=4, separators=(',', ': '))
            # TEST DIA
            DATA_TEST_DIA = test_dia_df # as input tokens #get_data_with_labels(all_input_tokens_test, y_true_test, y_pred_test)
            with open(PREDICTIONS_TEST, 'w') as f:
                json.dump(DATA_TEST_DIA, f, sort_keys=True, indent=4, separators=(',', ': '))
            
        if args.save_cr:
            # Print and save classification report
            cr_report_syn = seq_classification_report(y_true_test_syn, y_pred_test_syn, digits = 4)
            cr_report_dia = seq_classification_report(y_true_test_dia, y_pred_test_dia, digits = 4)
            cr_report_syn_overlap = seq_classification_report(y_true_test_syn, y_pred_test_syn, digits = 4, overlap = True)
            cr_report_dia_overlap = seq_classification_report(y_true_test_dia, y_pred_test_dia, digits = 4, overlap = True)
            #print(cr_report_syn)
            #print(cr_report_dia)
            pickle.dump(cr_report_syn, open(args.output_path+'classification_report_'+lm+'_test_syn_exact.txt','wb'))
            pickle.dump(cr_report_dia, open(args.output_path+'classification_report_'+lm+'_test_dia_exact.txt','wb'))
            pickle.dump(cr_report_syn_overlap, open(args.output_path+'classification_report_'+lm+'_test_syn_overlap.txt','wb'))
            pickle.dump(cr_report_dia_overlap, open(args.output_path+'classification_report_'+lm+'_test_dia_overlap.txt','wb'))


    #################################################################################
    # Load the fine-tuned model:
    if args.eval:
        logging.basicConfig(level=logging.INFO)

        if model_class=="BERT":
            # Model
            model = TokenBERT(
                model_name=args.lang_model, 
                num_labels=len(tag2idx),
                use_crf=args.use_crf)
        
        if model_class=="DistilBERT":
            # Model
            model = TokenDistilBERT(
                model_name=args.lang_model, 
                num_labels=len(tag2idx),
                use_crf=args.use_crf)

        model.load_state_dict( torch.load(MODEL_PATH) )
        model.to(device)  

        print("=================== Evaluate ================")

        # EVALUATION: TRAIN SET
        y_true_train, y_pred_train, f1s_train, f1s_overlap_train = evaluation(
                train_dataloader, model=model, device=device, tokenizer=tokenizer, tag_values=tag_values)
        print("TRAIN: F1 Exact %.3f | F1 Overlap %.3f"%(f1s_train, f1s_overlap_train))
        
        # EVALUATION: DEV SET
        y_true_dev, y_pred_dev, f1s_dev, f1s_overlap_dev = evaluation(
                dev_dataloader, model=model, device=device, tokenizer=tokenizer, tag_values=tag_values)
        print("EVAL: F1 Exact %.3f | F1 Overlap %.3f"%(f1s_dev, f1s_overlap_dev))
        
        # EVALUATION: TEST SYN SET
        y_true_test_syn, y_pred_test_syn, f1s_test_syn, f1s_overlap_test_syn = evaluation(
                test_syn_dataloader, model=model, device=device, tokenizer=tokenizer, tag_values=tag_values)
        print("TEST SYN: F1 Exact %.3f | F1 Overlap %.3f"%(f1s_test_syn, f1s_overlap_test_syn))

        # EVALUATION: TEST DIA SET
        y_true_test_dia, y_pred_test_dia, f1s_test_dia, f1s_overlap_test_dia = evaluation(
                test_dia_dataloader, model=model, device=device, tokenizer=tokenizer, tag_values=tag_values)
        print("TEST DIA: F1 Exact %.3f | F1 Overlap %.3f"%(f1s_test_dia, f1s_overlap_test_dia))

        if args.save_cr:
            # Print and save classification report
            cr_report_syn = seq_classification_report(y_true_test_syn, y_pred_test_syn, digits = 3)
            cr_report_dia = seq_classification_report(y_true_test_dia, y_pred_test_dia, digits = 3)
            cr_report_syn_overlap = seq_classification_report(y_true_test_syn, y_pred_test_syn, digits = 3, overlap = True)
            cr_report_dia_overlap = seq_classification_report(y_true_test_dia, y_pred_test_dia, digits = 3, overlap = True)
            #print(cr_report_syn)
            #print(cr_report_dia)
            pickle.dump(cr_report_syn, open(args.output_path+'classification_report_'+lm+'_test_syn_exact.txt','wb'))
            pickle.dump(cr_report_dia, open(args.output_path+'classification_report_'+lm+'_test_dia_exact.txt','wb'))
            pickle.dump(cr_report_syn_overlap, open(args.output_path+'classification_report_'+lm+'_test_syn_overlap.txt','wb'))
            pickle.dump(cr_report_dia_overlap, open(args.output_path+'classification_report_'+lm+'_test_dia_overlap.txt','wb'))


if __name__ == "__main__":
    set_all_seeds()
    main()