# importing packages 
import pandas as pd
import numpy as np
import regex as re
import ast
import os
import string
import pickle
import datetime

from functools import partial
import torch.optim as optim

from itertools import chain
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import wandb

# for word2vec
import nltk

# for the model
import torch
from sklearn.model_selection import train_test_split


# Column width total display
pd.set_option('display.max_colwidth', None)

# all columns total display
pd.set_option('display.max_columns', None)


DATE = datetime.datetime.today().strftime("%b_%d_%Y")
print(DATE)


PROJECT_SWEEP_NAME = f"keywords-approach-calgary-{DATE}"
# DATASET CREATION------------------------------------------------------
BASE_FOLDER = "---/split 80 - 20 /"

TRAIN_PATH = BASE_FOLDER + f'training_set.csv'
TEST_PATH = BASE_FOLDER + f'testing_set.csv'


max_len_tokens = 32
bert_embeddings_len = 256
col = "preprocessed"
LABEL = 4

# DATASET CREATION------------------------------------------------------
test_df_main = pd.read_csv(TEST_PATH)
test_text = test_df_main.preprocessed.values.tolist()
#PROJECT_SWEEP_NAME = f"few-shot-optim-sweep-2labeled-label{LABEL}"


FULL_DATA_PATH = "---/calgary_filtered_2020_2023_Jul_17_2023.csv"
df = pd.read_csv(FULL_DATA_PATH)

labels_to_int = {'anger': 0, 'joy': 1, 'optimism': 2, 'sadness': 3}


df.drop_duplicates(subset=['full_text'], inplace=True)
df.reset_index(inplace=True, drop=True)


# making sure that the model sees the test set/manually created dataset for the
# first time

# this is a different cleaning than what we have for other methods!
# in here, we only want words and nothing else.


def clean(t):
    """
    cleans a given tweet
    """
    t = t.lower()

    # list of some extra words: add gradually to this list
    extra = ["&gt;", "&lt;", "&amp;", "”", "“", "#"] # "\"", ','
    for patt in extra: t = re.sub(patt, '', t)

    # URL removal: Go on untill you hit a space
    t = re.sub(r"\S*https?:\S*", "", t)


    # removes all @s and the text right after them; mentions
    # Question: should we remove hashtags too?
    t = re.sub(r'@\w*\_*' , '', t)

    # substitute extra space with only one space
    t = re.sub(r' \s+', ' ', t)

    return t


df.loc[:, "cleaned_text"] = df.loc[:, "full_text"].apply(lambda x: clean(x))
df = df[df['cleaned_text']!=''].reset_index(drop=True)

df[['cleaned_text', 'full_text']]



df.drop_duplicates(subset=['cleaned_text'], inplace=True)
df.reset_index(inplace=True, drop=True)
len(df)



# making sure that the model sees the test set/manually created dataset for the
# first time

df_indices = []
for i in range(len(df)):
    if df.loc[i, "cleaned_text"] not in test_df_main.preprocessed.values:
        df_indices.append(i)

df = df.loc[df_indices, :]
df.reset_index(inplace=True, drop=True)
len(df)



keywords_and_emojis = {
    'anger': ['rage', 'anger', 'frustration', 'irritation', 'outrage', 'annoyance', 'resentment', 'indignation', '🤬', '😡', '😠', '👿'],
    'joy': ['happiness', 'joy', 'glee', 'excitement', 'euphoria', 'delight', 'bliss', 'ecstasy', '🌟', '😄', '😊', '🥳', '🎉'],
    'optimism': ['optimism', 'hope', 'confidence', 'positivity', 'enthusiasm', 'faith', 'assurance', 'expectation', '🌞', '🌈', '😃', '🙌'],
    'sadness': ['sadness', 'grief', 'despair', 'heartbreak', 'sorrow', 'depression', 'melancholy', 'anguish', '😔', '😢', '😭', '💔'],
}




# Finding emotions ----------------------------

print('Finding out the labels with emotion...')
emotion_idx = {}
all_emotions = keywords_and_emojis.keys()


def create_regex(list_):
    """"
    creating the appropriate regex
    """
    final_patt = r'\b(?:'

    for i, patt in enumerate(list_):
        if i == len(list_)-1: final_patt += patt
        else: final_patt += patt + r'|'

    final_patt = final_patt + r')\b'
    return final_patt


for emotion in all_emotions:
    print(f'emotion: {emotion}')

    keywords = keywords_and_emojis[emotion]
    print(f'list of words we look for {keywords}')

    inclusion_criteria = create_regex(keywords)
    indices = df[df['full_text'].str.contains(inclusion_criteria, na=False, case=False)].index
    assert len(indices) == len(set(indices))
    emotion_idx[emotion] = indices
    print(f'number of records: {len(indices)}')
    print()


print('')
i = 0
all_dfs = []

for emotion in all_emotions:

    indices = emotion_idx[emotion]
    assert(len(indices) ==  len( list(set(indices))))
    cleaned_series = df.loc[indices, 'cleaned_text'].values
    #full_txt_series = df.loc[indices, "full_text"].values
    # UNCOMMENT
    # ids = df.loc[indices, "id"].values
    emotion_i_df = pd.DataFrame({
        #'full_text': full_txt_series,
         'text': cleaned_series,
        'label': emotion,
        # 'id': ids
        })
    emotion_i_df.loc[:, "label"] = i

    print(f'{emotion} -> {i}')
    all_dfs.append(emotion_i_df)
    i += 1

labeled_df = pd.concat(all_dfs, axis=0)
labeled_df.reset_index(inplace=True, drop=True)



labeled_df_optim = labeled_df[labeled_df.loc[:, 'label'] ==2].sample(2000)
labeled_df_not_optim = labeled_df[labeled_df.loc[:, 'label'] !=2]



labeled_df_total = pd.concat([labeled_df_optim, labeled_df_not_optim ])
print('length of labeled_df_total = ', len(labeled_df_total))

# first identifying the duplicates in text
all_duplicates_df = labeled_df_total[labeled_df_total.duplicated(subset=['text'], keep=False)]

# then identifying the ones that have different labels in those duplicates
difference_df = all_duplicates_df[all_duplicates_df.duplicated(subset=['label'])==False]

# saving the index
difference_idx = difference_df.index

# removing those indices from the dataset
labeled_df_total = labeled_df_total[~labeled_df_total.index.isin(difference_idx)]
labeled_df_total.reset_index(drop=True, inplace=True)

print(' length of labeled_df_total after removing the duplicates in labels = ', len(labeled_df_total))


# creating the model --------------------------------------
print('creating the semantic model')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers import BertTokenizer

# Select CPU/GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\ndevice = ', device)

model_checkpoint = "cardiffnlp/twitter-roberta-base"



tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, do_lower_case=True)


# training and testing set creation ----------------------
# splitting the dataset 

print('training and testing set creation')
ratio = 0.1
# val_df = labeled_df_total.sample(frac = ratio, random_state=42)
# train_indices = list(set(labeled_df_total.index) - set(val_df.index))

# train_df = labeled_df_total.loc[train_indices]


train_df, val_df = train_test_split(labeled_df_total, test_size=ratio, random_state=42, shuffle=True)
print('training examples: ', len(train_df), ' and for validation we have ', len(val_df), ' records')

print('length of train_df = ', len(train_df))


# training the model -------------------------------------
def tokenize(sequences):
    '''
    tokenize all of the sentences and map the tokens to their word IDs.
    '''
    input_ids = []
    attention_masks = []

    # For every caption...
    for seq in sequences:
        
        encoded_dict = tokenizer.encode_plus(
                            seq,                       # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_len_tokens + 2, # Pad & truncate all sentences.
                            truncation=True,
                            padding='max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',      # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    
    return input_ids, attention_masks




class mydataset():    

    '''
    Dataloader 
    For Training: Returns (Tweet, Input_id, Attention_mask, label)
    For Testing: Returns (Tweet, Input_id, Attention_mask)
    '''
    def __init__(self, classification_df, name = 'train'):

        super(mydataset).__init__()
        self.name = name
        self.tweet = []
        self.Y = []
                    
        for index, rows in classification_df.iterrows():

            text = rows['text']
            self.tweet.append(text)  # Question: why ''.join(tweet) ????
            
            # if name == 'train' or self.name == 'valid':
            label = rows['label']
            self.Y.append(label)


        
        
        '''
        Tokenize all of the tweets 
        get the input ids and attention masks.
        '''
        self.input_ids, self.attention_masks = tokenize(self.tweet)
        
        
    
    def __getitem__(self,index): 
        '''
        For tweets, Input ids and Attention mask
        '''
        tweet = self.tweet[index]
        input_id = self.input_ids[index]
        attention_masks = self.attention_masks[index]
        
        
        '''
        For Labels during training
        '''      
        # if self.name == 'train' or self.name == 'valid' :
        label = float(self.Y[index])
        
        return tweet, input_id, attention_masks, torch.as_tensor(label).long()

    def __len__(self):
        return len(self.tweet)
    



'''
Function to perform inference on validation set
Returns: validation loss, top1 accuracy
'''

def test_classify(model, valid_loader, criterion, device):
    model.eval()
    test_loss = []
    top1_accuracy = 0
    total = 0

    for batch_num, (tweet, input_id, attention_masks, target) in enumerate(valid_loader):
               
        input_ids, attention_masks, target = input_id.to(device), attention_masks.to(device), target.to(device)
            
        '''
        Compute output and loss from BERT
        '''
        loss, logits = model(input_ids, 
                         token_type_ids=None, 
                         attention_mask=attention_masks, 
                         labels=target,
                         return_dict=False)

        test_loss.extend([loss.item()]*input_id.size()[0])
        
        predictions = F.softmax(logits, dim=1)
        
        _, top1_pred_labels = torch.max(predictions,1)
        top1_pred_labels = top1_pred_labels.view(-1)
        
        top1_accuracy += torch.sum(torch.eq(top1_pred_labels, target)).item()
        total += len(target)

    return np.mean(test_loss), top1_accuracy/total


# testing on test data ---------------------------------

print('testing set distribution: ')
print(test_df_main.groupby('label').count())



def test_acc(model, device, config):

    model.eval()
    preds = []
    texts = []
    logits_list = []
    preset_labels = []

    '''
    TEST dataloader
    '''
    test_dataset = mydataset(test_df_main, name = 'test')
    test_dataloader = data.DataLoader(test_dataset, shuffle= True, batch_size = config['batch_size'], num_workers=2, pin_memory=True)


    for batch_num, (batch_text, input_id, attention_masks, batch_labels) in enumerate(test_dataloader):
     
        
        input_ids, attention_masks = input_id.to(device), attention_masks.to(device)
            
        '''
        Compute prediction outputs from BERT
        '''
        output_dictionary = model(input_ids, 
                         token_type_ids=None, 
                         attention_mask=attention_masks, 
                         return_dict=True)
        
        predictions = F.softmax(output_dictionary['logits'], dim=1)
        
        labels_pred = torch.max(predictions, 1).indices.cpu().numpy()
        logits = torch.max(predictions, 1).values.cpu().detach().numpy()
        preds.extend(labels_pred)
        logits_list.extend(logits)
        texts.extend(batch_text)
        preset_labels.extend(batch_labels.numpy())


    assert len(test_df_main) == len(preset_labels)
    print('len(text) = ', len(texts))
    print('len(test_df_main) =', len(test_df_main))

    final_df = pd.DataFrame({'text': test_df_main.preprocessed.values,
                             'prediction': preds, 
                             'preset_label': preset_labels})    
    # we're saving the output of the model on our testing set 
    # instead of saving the model which can be memory consuming
    
    
    test_accuracy = len(final_df[final_df['preset_label'] == final_df['prediction']]) / len(final_df)
    print("Best trial test set accuracy: {}".format(test_accuracy))
    
    
    
    # y_true = [0,0,1,1,1]
    # y_preds = [0,0,0,0,1]
    # array([[1.        , 0.        ],
      #      [0.66666667, 0.33333333]])

    
    cm = confusion_matrix(
        final_df['preset_label'], 
        final_df['prediction'], 
        normalize="true")
    acc_1 = cm[1, 1]
    print('Best trial test set accuracy on one label: ', acc_1)

    return test_accuracy, cm




# HYPERPARAMETER SPACE DEFINITION---------------------------------------------

# Refer for distributions: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration

sweep_config = {
    
    # define the search method
    # one of "grid", "random" or "bayes"
    'method': 'bayes',
    
    # define the metric (useful for bayesian sweeps)
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    }
}

parameters = {
    # defining constant parameters
    # 'dataset': {'value': 'SetFit/sst2'},

    # define different types of losses for contrastive learning
    # these losses comes from sentence_transformers library

    'batch_size': {
        # integers between 4 and 64
        'distribution': 'categorical',
        'values': [4, 8, 16, 24, 32, 64]
    },
    'num_epochs': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 5
    },

    'learning_rate': {
        'distribution': 'categorical',
        'values': [1e-6, 5e-6, 1e-5, 5e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4],
    },

    'warmup_steps': {'values': [0, 1, 3, 8]},
}

# adding the hyperparameters to the parameters field in the sweep_config dictionary
sweep_config['parameters'] = parameters



def train(config=None):
    
    
    #print(config.name)

    '''
    Train Dataloader
    ''' 
    print('entering the training')
    train_dataset = mydataset(train_df, name = 'train')
    train_loader = data.DataLoader(train_dataset, shuffle= True, batch_size = int(config["batch_size"]), num_workers=2, pin_memory=True)
    print('got the training set')
    '''
    Validation_Dataloader
    '''
    validation_dataset = mydataset(val_df, name = 'valid')
    valid_loader = data.DataLoader(validation_dataset, shuffle= True, batch_size = int(config["batch_size"]), num_workers=2, pin_memory=True)


    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels = 4, #Number of Classes
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )   
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr = config["learning_rate"], eps = 1e-8)
    criterion = nn.CrossEntropyLoss()

    '''
    Create the learning rate scheduler.
    Total number of training steps is [number of batches] x [number of epochs].
    '''
    total_steps = len(train_loader) * config['num_epochs']
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = config['warmup_steps'],  num_training_steps = total_steps)


    model.train()


    # train_loss = []
    # train_acc = []
    # valid_loss = []
    # valid_acc = []


    for epoch in range(config['num_epochs']):
        avg_loss = 0.0
                
        
        for batch_num, (tweet, input_id, attention_masks, target) in enumerate(train_loader):
            print('tweet = ', tweet)
            
            input_ids, attention_masks, target = input_id.to(device), attention_masks.to(device), target.to(device)
                
            '''
            Compute output and loss from BERT
            '''
            loss, logits = model(input_ids, 
                            token_type_ids=None, 
                            attention_mask=attention_masks, 
                            labels=target,
                            return_dict=False
                                )
            
            '''
            Take Step
            '''
            optimizer.zero_grad()
            loss.backward()
            '''
            Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
            '''
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            
            optimizer.step()
            avg_loss += loss.item()

            '''
            linear_schedule_with_warmup take step after each batch
            '''
            lr_scheduler.step()
                
                            
        training_loss = avg_loss/len(train_loader)
    
        print('Epoch: ', epoch+1)            
        #print('training loss = ', training_loss)

        '''
        check performance on training set
        '''
        
        _, top1_acc_train= test_classify(model, train_loader, criterion, device)
        
        print('training loss : {:.4f}\t Train Accuracy: {:.4f}'.format(training_loss, top1_acc_train))

        '''
        Check performance on validation set after an Epoch
        '''
        validation_loss, top1_acc_valid= test_classify(model, valid_loader, criterion, device)
        # valid_loss.append(validation_loss)
        # valid_acc.append(top1_acc_valid)
        print('Validation Loss: {:.4f}\tTop 1 Validation Accuracy: {:.4f}'.format(validation_loss, top1_acc_valid))
        
    metrics = {'valid_accuracy':top1_acc_valid, 'valid_loss':validation_loss,
                'training_acc':top1_acc_train, 'training_loss':training_loss}
    print(metrics)
    
    print("Finished Training")
    test_accuracy, cm = test_acc(model, device, config)
    metrics['test_acc'] = test_accuracy
    metrics['cm'] = cm


    wandb.log(metrics)


    
# PARAMETERS 

# Loss Function
criterion = nn.CrossEntropyLoss()


'''
Create the learning rate scheduler.
Total number of training steps is [number of batches] x [number of epochs].
'''



sweep_id = wandb.sweep(sweep_config, project=PROJECT_SWEEP_NAME)
wandb.agent(sweep_id, train)