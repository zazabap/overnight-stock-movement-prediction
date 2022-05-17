# Author: Shiwen An 
# Date: 2022/05/17
# Purpose: FinNLP Submission

import tensorflow as tf
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

#df = pd.read_csv("../data/2013/01/01/2013-01-01.csv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
df = pd.read_csv("../data/2013/01/01/2013-01-01.csv", 
        delimiter=',', header=None, skiprows=[0], 
        names=[
            'Unique_Story_Index',
            'PNAC',
            'Story_Date_Time',
            'Take_Date_Time',
            'Headline',
            'Story_Body',
            'Products',
            'Topics',
            'Related_RICs',
            'Attribution',
            'Language',
            'Issuer_OrgID',
            'PILC',
            'DATETIME',
            'year',
            'month',
            'day'])

def tutorial():
  print(df)
  print(df.shape)    
  print(df.sample(0))
  
  sentences = df.Headline.values
  sentences = ["[CLS] " + Headline + " [SEP]" for Headline in sentences]
  labels = df.day.values

  # Bring BERT Models in
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

  print ("Tokenize the first 10 sentence:")
  for i in range(10):
    print (tokenized_texts[i])
  
  MAX_LEN = 64
  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
  #input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

  # input_ids print; change the tokens to ids
  # Currently Just use maximum length of 64 
  print("input_ids")
  for i in range(1):
    print (input_ids[i])

  # What is the 
  attention_masks = []
  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
  print("Attention Mask")
  print(attention_masks[0])

  # The actual training process or something else?
  train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                            random_state=2018, test_size=0.1)
  train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)



  # Convert all of our data into torch tensors, the required datatype for our model

  train_inputs = torch.tensor(train_inputs)
  validation_inputs = torch.tensor(validation_inputs)
  train_labels = torch.tensor(train_labels)
  validation_labels = torch.tensor(validation_labels)
  train_masks = torch.tensor(train_masks)
  validation_masks = torch.tensor(validation_masks)

  # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
  batch_size = 32

  # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
  # with an iterator the entire dataset does not need to be loaded into memory
  
  train_data = TensorDataset(train_inputs, train_masks, train_labels)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
  
  validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
  validation_sampler = SequentialSampler(validation_data)
  validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

  print("End Tutorial")

if __name__ == "__main__":
  tutorial()
