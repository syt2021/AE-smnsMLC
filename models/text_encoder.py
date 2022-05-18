#!/usr/bin/env python
# coding:utf-8
import os

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):
    def __init__(self, config):
        """
        :param config: helper.configure, Configure Object
        """
        super(TextEncoder, self).__init__()
        self.config = config
        self.cur_path = os.getcwd()
        print('text encoder path: ', self.cur_path)
        self.model_name = os.path.join(self.cur_path, 'bert_base_pretrained')
        print('pretrained bert model path: ', self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(self.model_name)
        
        self.kernel_sizes = config.text_encoder.CNN.kernel_size
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(torch.nn.Conv1d(
                hidden_dimension,
                config.text_encoder.CNN.num_kernel,
                kernel_size,
                padding=kernel_size // 2
                )
            )
                

    def forward(self, inputs, seq_lens):
        """
        :param inputs: torch.FloatTensor, embedding, (batch, max_len, embedding_dim)
        :param seq_lens: torch.LongTensor, (batch, max_len)
        :return:
        """
        #print('original inputs: ', inputs)
        #inputs_forbert = self.tokenizer(inputs, return_tensors="pt")
        #print('bert tokenized inputs: ', inputs_forbert.data)
        with torch.no_grad():
            outputs = self.bert_model(inputs)
            last_hidden_states = outputs.last_hidden_state
            #print('last hidden states: ', last_hidden_states)
            #print('last hidden states shape: ', last_hidden_states.shape)
        text_output_bert = last_hidden_states[:,:,:]
        
        text_output = text_output_bert.transpose(1, 2)
        
        for _, conv in enumerate(self.convs):
            convolution = F.relu(conv(text_output))
            #print('convolution shape: ', convolution.shape)
            text_output = convolution.transpose(1,2)
        
        return text_output
