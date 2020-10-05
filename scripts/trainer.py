#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kaivalya mannam
"""
import torch
torch.manual_seed(9)
torch.backends.cudnn_deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(9)
import random
random.seed(9)

import sys
import os
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings,WordEmbeddings, TransformerWordEmbeddings, ELMoEmbeddings, FlairEmbeddings, StackedEmbeddings
from typing import List


# this is the folder in which train, test and dev files reside
data_folder = 'data'

mini_batch_size = 16
dataset_number = 1
arch = 0
dropout = 0
embedding_names = ["cb", "ep", "fp", "fpd", "bb"]
embedding = "ep" # one of embedding_names
modes = ["new", "resume"]
resume = False # True
run_id = ""

def usage():
    print("Arg 1 (embedding): " + " or ".join(embedding_names))            
    sys.exit(1)

if len(sys.argv)<2:
    usage()

embedding = sys.argv[1]
if str.lower(embedding) not in embedding_names:
    usage()    
    
if embedding=="fp" and not (os.path.exists('./forward-lm.pt') and os.path.exists('./backward-lm.pt')):
        print("forward-lm.pt / backward-lm.pt files are missing")
        print("Please copy and restart")
        sys.exit()
        
if embedding=="cb" and not os.path.exists('./bert-base-clinical-cased'):
        print("bert-base-clinical-cased directory is missing")
        print("Please copy and restart")
        sys.exit()    

if embedding=="bb" and not os.path.exists('./biobert_v1.1_pubmed'):
    print("biobert_v1.1_pubmed directory is missing")
    print("Please copy and restart")
    sys.exit()   
        
print("Embedding: " + embedding)

resume = False

if not os.path.exists(data_folder+'/dataset'+str(dataset_number)+'_train.txt') or not os.path.exists(data_folder+'/dataset'+str(dataset_number)+'_test.txt'):
    print("Data-set training and test files missing")
    print("Please copy them to the current directory")
    sys.exit()
            
# define columns
columns = {0: 'file', 1: 'line', 2: 'index', 3: 'sequence', 4: 'start', 5: 'end', 6: 'orig', 7: 'text',  8: 'ner'}

# 1. get the corpus
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='dataset'+str(dataset_number)+'_train.txt',
                              test_file='dataset'+str(dataset_number)+'_test.txt',
                              dev_file=None)
#corpus.downsample(0.1)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)
print(corpus.train[0].to_tagged_string('ner'))

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = []


if embedding == "ep":
    embedding_types = [WordEmbeddings('glove'),ELMoEmbeddings('pubmed')]
elif embedding == "fp":        
    embedding_types = [WordEmbeddings('glove'),FlairEmbeddings('./forward-lm.pt'),FlairEmbeddings('./backward-lm.pt')]    
elif embedding == "fpd":        
    embedding_types = [WordEmbeddings('glove'),FlairEmbeddings('pubmed-forward'),FlairEmbeddings('pubmed-backward')]    
elif embedding == "cb":        
    embedding_types =  [WordEmbeddings('glove'), TransformerWordEmbeddings('./bert-base-clinical-cased',layers="all",allow_long_sentences=True, use_scalar_mix=True, pooling_operation="mean", fine_tune=False)]
    mini_batch_size = 8    
elif embedding == "bb":    
    embedding_types =  [WordEmbeddings('glove'), TransformerWordEmbeddings('./biobert_v1.1_pubmed',layers="all",allow_long_sentences=True, use_scalar_mix=True, pooling_operation="mean", fine_tune=False)]
    mini_batch_size = 8    
elif embedding == "bblarge":    
    embedding_types =  [WordEmbeddings('glove'), TransformerWordEmbeddings('./biobert_large_v1.1_pubmed',layers="all",allow_long_sentences=True, use_scalar_mix=True, pooling_operation="mean", fine_tune=False)]
    mini_batch_size = 8    

print("mini_batch_size is -" + str(mini_batch_size))

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger = None

# 5. initialize sequence tagger
from flair.models import SequenceTagger    

tagger: SequenceTagger = SequenceTagger(hidden_size=256, rnn_layers=1, dropout = dropout,
                            embeddings=embeddings, tag_dictionary=tag_dictionary,
                            tag_type=tag_type, use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer
   
run_id = 'model'

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train(run_id,
          mini_batch_size=mini_batch_size, 
          learning_rate=0.1,
          checkpoint = True, 
          patience=3,
          train_with_dev=True, 
          anneal_factor=0.5, max_epochs=1)

