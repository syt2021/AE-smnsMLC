#!/usr/bin/env python
# coding:utf-8

import torch
from torch import nn
from torch.nn import functional
from models.label_embedding_layer import EmbeddingLayer
from torch.nn.parameter import Parameter


class HiAGMLA(nn.Module):
    def __init__(self, config, label_map, model_mode, device):
        """
        Hierarchy-Aware Global Model : (Parallel) Multi-label attention Variant
		paper: Jie Zhou, Chunping Ma, Dingkun Long, Guangwei Xu, Ning Ding, Haoyu Zhang, Pengjun Xie, and Gongshen Liu. Hierarchy-aware global model for hierarchical text classification. ACL 2020.
        :param config: helper.configure, Configure Object
        :param label_map: helper.vocab.Vocab.v2i['label'] -> Dict{str:int}
        :param model_mode: 'TRAIN'， 'EVAL'
        :param graph_model: computational graph for graph model
        :param device: torch.device, config.train.device_setting.device
        """
        super(HiAGMLA, self).__init__()

        self.config = config
        self.device = device
        self.label_map = label_map

        self.label_embedding = EmbeddingLayer(
            vocab_map=self.label_map,
            embedding_dim=config.embedding.label.dimension,
            vocab_name='label',
            config=config,
            padding_index=None,
            pretrained_dir=config.embedding.label.pretrained_file,
            model_mode=model_mode,
            initial_type=config.embedding.label.init_type
        ).to(device)

        # classifier
        self.linear = nn.Linear(len(self.label_map) * config.embedding.label.dimension,
                                len(self.label_map)).to(device)

        # dropout
        self.dropout = nn.Dropout(p=config.model.classifier.dropout).to(device)

        self.model_mode = model_mode


    @staticmethod
    def _soft_attention(text_f, label_f):
        """
        soft attention module
        :param text_f -> torch.FloatTensor, (batch_size, K, dim)
        :param label_f ->  torch.FloatTensor, (N, dim)
        :return: label_align ->  torch.FloatTensor, (batch, N, dim)
        """
        att = torch.matmul(text_f, label_f.transpose(0, 1))
        weight_label = functional.softmax(att.transpose(1, 2), dim=-1)
        label_align = torch.matmul(weight_label, text_f)
        return label_align

    def forward(self, text_feature):
        """
        forward pass with multi-label attention
        :param text_feature ->  torch.FloatTensor, (batch_size, K0, text_dim)
        :return: logits ->  torch.FloatTensor, (batch, N)
        """
        

        if self.model_mode == 'TEST':
            label_feature = self.label_feature
        else:
            label_embedding = self.label_embedding(torch.arange(0, len(self.label_map)).long().to(self.device))
            label_feature = label_embedding
            

        #print('text feature shape: ', text_feature.shape)
        text_feature = torch.mean(text_feature, dim=1, keepdim=True)
        label_aware_text_feature = self._soft_attention(text_feature, label_feature)
        #label_aware_text_feature = label_aware_text_feature.to('cuda:7')
        label_aware_text_feature_linear = self.linear(label_aware_text_feature.view(label_aware_text_feature.shape[0], -1))
        #label_aware_text_feature_linear = label_aware_text_feature_linear.to('cuda:7')

        logits = self.dropout(label_aware_text_feature_linear)
        return label_feature, logits

