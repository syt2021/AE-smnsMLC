#!/usr/bin/env python
# coding:utf-8

import torch.nn as nn
import torch
import numpy as np
from models.text_encoder import TextEncoder
from models.embedding_layer import EmbeddingLayer
from models.multi_label_attention import HiAGMLA

from models.labelprior_discriminator import LabelPriorDiscriminator
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
import json



class AEsmnsMLC(nn.Module):
    def __init__(self, config, vocab, model_mode='TRAIN'):
        """
        AEsmnsMLC Model class
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(AEsmnsMLC, self).__init__()
        self.config = config
        self.vocab = vocab
        #self.vocab = self.vocab.to(config.train.device_setting.device)
        self.device = config.train.device_setting.device

        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']
        self.index2label = vocab.i2v['label']

        #self.token_map = self.token_map.to(config.train.device_setting.device)

        self.token_embedding = EmbeddingLayer(
            vocab_map=self.token_map,
            embedding_dim=config.embedding.token.dimension,
            vocab_name='token',
            config=config,
            padding_index=vocab.padding_index,
            pretrained_dir=config.embedding.token.pretrained_file,
            model_mode=model_mode,
            initial_type=config.embedding.token.init_type
        )
        self.token_embedding.to(self.device)


        self.labelpriorweight_linear = nn.Linear(len(self.label_map) * config.embedding.label.dimension, 1)
        
        self.label_weight_estimator = Parameter(torch.Tensor(config.embedding.label.dimension, 1))
        nn.init.xavier_uniform_(self.label_weight_estimator)


        self.text_encoder = TextEncoder(config)
        self.text_encoder.to(self.device)
        
        self.label_prior_d = LabelPriorDiscriminator()

        self.aesmnsmlc = HiAGMLA(config=config,
                                 device=self.device,
                                 label_map=self.label_map,
                                 model_mode=model_mode)
        self.aesmnsmlc.to(self.device)

        # load tagmaster
        self.cur_path = os.getcwd()
        self.tagmaster_file_path = os.path.join(self.cur_path, 'tagmaster_154.json')
        with open(self.tagmaster_file_path, 'r') as jf:
            self.tagmaster = json.load(jf)
        
    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                           'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                          }]
        """
        params = list()
        params.append({'params': self.text_encoder.parameters()})
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.aesmnsmlc.parameters()})
        return params

    def forward(self, batch):
        """
        forward pass of the overall architecture
        :param batch: DataLoader._DataLoaderIter[Dict{'token_len': List}], each batch sampled from the current epoch
        :return: 
        """

        # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
        #print('input shape: ', batch['token'].shape)#embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))
        #print('input format: ', batch['token'])#embedding = self.token_embedding(batch['token'].to("cuda:7"))
        seq_len = batch['token_len']
        token_output = self.text_encoder(batch['token'].to(self.config.train.device_setting.device), seq_len)
        #token_output = token_output.to('cuda:7')

        all_labels_feature, logits = self.aesmnsmlc(token_output)
        #logits = logits.to(self.device)

        text_feature = token_output
        idx = np.random.permutation(text_feature.shape[0])
        negative_text = text_feature[idx, :, :]

        similarity_loss = 0
        num_samples = len(batch['label_list'])
        i = 0
        for label_index, labels_nestedlist in zip(batch['label_list'], batch['label_nestedlist']):
            # Label Selector: select the corresponding labels for each text sample
            label_feature = all_labels_feature[label_index,:]

            # max pooling of ground truth labels' embedding
            weighted_sum_label_emb = torch.topk(label_feature, 1, dim=0)[0].squeeze(0)

            # calculate the similartiy between text feature and combined label embeddings
            each_text_feature = text_feature[i]
            each_text_feature = torch.mean(each_text_feature, dim=0)
            sim_score = F.cosine_similarity(each_text_feature, weighted_sum_label_emb, dim=0)
            #print('cosine_similartiy: ', sim_score)
            if not torch.isnan(sim_score):
                similarity_loss += -sim_score

            # select negative labels for each sample, and calculate the similarity 
            # between sample and its negative labels
            neg_labels_ids = []
            for each_group in labels_nestedlist:
                groupid = each_group[0]
                tagids = each_group[1:]
                if groupid in self.tagmaster:
                    tagids_in_same_group = self.tagmaster[groupid]
                    #remove its gold label ids from the candidates
                    neg_labels_candidates = [x for x in tagids_in_same_group if x not in tagids]
                    # randomly generate neg labels
                    if len(neg_labels_candidates) != 0:
                        equal_probs = [1.0/len(neg_labels_candidates)] * len(neg_labels_candidates)
                        neg_label_num = len(tagids)
                        if len(neg_labels_candidates) < neg_label_num:
                            neg_label_num = len(neg_labels_candidates)
                        selected_candidate_indices = np.random.choice(np.arange(len(neg_labels_candidates)), size=neg_label_num, p=equal_probs)
                        neg_label_ids_for_eachgroup =  [self.label_map[neg_labels_candidates[each]] for each in selected_candidate_indices]
                        neg_labels_ids += neg_label_ids_for_eachgroup
            neg_labels_feature = all_labels_feature[neg_labels_ids,:]
            neg_label_combinedemb = torch.mean(neg_labels_feature, dim=0)
            # calculate similarity between text feature and negative labels feature
            neg_sim_score = F.cosine_similarity(each_text_feature, neg_label_combinedemb, dim=0)
            #print('neg_cosine_similartiy: ', neg_sim_score)
            if not torch.isnan(neg_sim_score):
                similarity_loss += neg_sim_score
            i += 1

        similarity_loss /= num_samples

        # compute the label prior matching loss
        label_totalnum = all_labels_feature.shape[0]
        #print('Generated label embedding size: ', all_labels_feature.shape)
        label_prior_loss = 0.0
        for i in range(label_totalnum):
            label_y = all_labels_feature[i]
            label_prior = torch.rand_like(label_y)
            term_a = torch.log(self.label_prior_d(label_prior)).mean()
            term_b = torch.log(1.0 - self.label_prior_d(label_y)).mean()
            label_prior_loss += - (term_a + term_b)
        label_prior_loss /= label_totalnum
        #label_prior_loss = label_prior_loss.to('cuda:6')
    
        # loss weight estimator: compute the weight for label_prior_loss
        labelprior_weightlogit = self.labelpriorweight_linear(all_labels_feature.view(-1))
        labelprior_weight = F.sigmoid(labelprior_weightlogit)        

        return similarity_loss, label_prior_loss, logits, labelprior_weight
