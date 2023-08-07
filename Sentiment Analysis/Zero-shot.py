# run on Google colab

# pip install -U sentence-transformers
# !wget -O test_set.csv "https://www.dropbox.com/scl/fi/kfz0xefhis2ltdv4yfl97/testing_set.csv?rlkey=z71iaaycrkuy7i12gyqu6azp5&dl=0"

# importing packages
import pandas as pd
import numpy as np
import regex as re
import string
import plotly.express as px
import datetime

# for the model
import torch

# Column width total display
pd.set_option('display.max_colwidth', None)

# all columns total display
pd.set_option('display.max_columns', None)

# cleaning
test_df = pd.read_csv(r"test_set.csv")
test_df = test_df[~test_df.full_text.isna()]

# importing the model

from sentence_transformers import SentenceTransformer, util

# one of the good models
# Dimensions:	384
# see other models at : https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer('all-mpnet-base-v2')

# we need to tokenize and see if most tweets are shorter than the
# maximum sequence length that this model accepts
print("Max Sequence Length:", model.max_seq_length)


labels_to_int = {'anger': 0, 'joy': 1, 'optimism': 2, 'sadness': 3}


# texts to compare
sentences1 = test_df.loc[:, "preprocessed"]
sentences2 = ['anger', 'joy', 'optimism', 'sadness']

# Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

from collections import defaultdict

result_dict = defaultdict(dict)

for i in range(cosine_scores.size()[0]):
    result_dict[i] = {
        "indices": torch.topk(cosine_scores[i], 4).indices,
        "probs": torch.topk(cosine_scores[i], 4).values,
        "text": test_df.loc[i, "preprocessed"]
        }

indices_list = []
probs_list = []
texts_list = []
preds_list = []

for i in result_dict.values():

    # print(i['text'])
    # print(i['probs'].cpu() .numpy())
    # print(i['indices'].cpu() .numpy())

    probs_list.append(i['probs'].cpu() .numpy())
    pred_idx = i['indices'].cpu() .numpy()
    indices_list.append(pred_idx)
    preds_list.append(pred_idx[0])
    texts_list.append(i['text'])

pred_df = pd.DataFrame(
    {
        'text': texts_list, 'preds': preds_list, 'indices': indices_list,
     'probs': probs_list, 'present_label': test_df.label
     }
     )

acc = len(pred_df[pred_df['preds']==pred_df['present_label']]) / len(pred_df)
print('accuracy is ', acc)

from sklearn.metrics import confusion_matrix

# Step 4: Create the confusion matrix
cm = confusion_matrix(pred_df['present_label'], pred_df['preds'])

# Step 5: Convert the confusion matrix to a DataFrame (optional)
cm_df = pd.DataFrame(cm, index=sentences2, columns=['pred_' + i for i in sentences2])
print(cm_df)