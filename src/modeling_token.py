#!/usr/bin/env python3
from torch import nn, sigmoid
from transformers import BertForTokenClassification, DistilBertForTokenClassification
from torchcrf import CRF

class TokenBERT(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fields (CRF)
    '''
    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True):
        super(TokenBERT, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.tokenbert = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            problem_type="multi_label_classification"
        )
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=self.batch_first)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.tokenbert.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.tokenbert.dropout(sequence_output)
        logits = self.tokenbert.classifier(sequence_output)

        if self.use_crf: # not for multi-labeling
            if labels is not None: # training
                return -self.crf(logits, labels, attention_mask.byte())
            else: # inference
                return self.crf.decode(logits, attention_mask.byte())
        else:
            if labels is not None: # training
                labels = nn.functional.one_hot(labels)
                loss_fct = nn.BCEWithLogitsLoss() # BCEwithLogitsLoss: combined Sigmoid layer and binary cross entropy
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.type_as(logits).view(-1, self.num_labels)
                )
                return loss
            else: # inference
                #return argmax(logits, dim=2)
                return sigmoid(logits)


class TokenDistilBERT(nn.Module):
    '''
        Token DistilBERT with (optional) Conditional Random Fields (CRF)
    '''
    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True):
        super(TokenDistilBERT, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.tokendistilbert = DistilBertForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            problem_type="multi_label_classification"
        )
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=self.batch_first)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.tokendistilbert.distilbert(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        sequence_output = self.tokendistilbert.dropout(sequence_output)
        logits = self.tokendistilbert.classifier(sequence_output)

        if self.use_crf:
            if labels is not None: # training
                return -self.crf(logits, labels, attention_mask.byte())
            else: # inference
                return self.crf.decode(logits, attention_mask.byte())
        else:
            if labels is not None: # training
                labels = nn.functional.one_hot(labels)
                loss_fct = nn.BCEWithLogitsLoss() # BCEwithLogitsLoss: combined Sigmoid layer and binary cross entropy
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.type_as(logits).view(-1, self.num_labels)
                )
                return loss
            else: # inference
                return sigmoid(logits)