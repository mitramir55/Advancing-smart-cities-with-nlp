# importing packages 
import pandas as pd
import numpy as np
import regex as re
import string
import datetime
import nltk
import pickle

# stopwords
# nltk stopwords

nltk.download('punkt')
max_seq_len = 512


DATE = datetime.datetime.today().strftime("%b_%d_%Y")
print(DATE)




# import files ---------------------------------------
BASE_FOLDER = "/home/mitrasadat.mirshafie/Thesis/June 26th - first round/sentiment analysis/fewshot files/"
TRAIN_PATH = BASE_FOLDER + f'split 80 - 20 /training_set.csv'
TEST_PATH = BASE_FOLDER + f'split 80 - 20 /testing_set.csv'
df = pd.read_csv(TEST_PATH)

# remove any nulls in text for cleaning
df.dropna(axis=0, subset=['preprocessed'], inplace=True)

df.reset_index(drop=True, inplace=True)

print(f"length of the df = {len(df)}")



# FEW-SHOT ------------------------------------------
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch

from transformers import set_seed
from datasets import load_dataset
from sentence_transformers.losses import (
    CosineSimilarityLoss,
    ContrastiveLoss,
    BatchAllTripletLoss,
    BatchHardTripletLoss,
)
from setfit import SetFitModel, SetFitTrainer
import gc, math
from sklearn.metrics import confusion_matrix


LABEL = 4
# PREDS_FOLDER = "/home/mitrasadat.mirshafie/JC_folder/files/few-shot-final-analysis/"
PREDS_FOLDER = "/home/mitrasadat.mirshafie/Thesis/June 26th - first round/sentiment analysis/fewshot files/"

# DATASET CREATION------------------------------------------------------
dataset = load_dataset('csv', data_files={'train': TRAIN_PATH,
                                           'valid': TEST_PATH})

config = {
    'model': "sentence-transformers/all-mpnet-base-v2",
    'loss_class': "ContrastiveLoss", 
    'batch_size': 128,
    'num_iterations': 20,
    'num_epochs': 5,
    'learning_rate': 0.00005,
    'warmup_proportion': 0.7,
}

# FUNCTION FOR TRAINING
def train(config):
        
        # train dataset
        train_dataset = dataset["train"]

        # validation dataset
        valid_dataset = dataset["valid"]

        # Load a SetFit model from Hub
        model = SetFitModel.from_pretrained(config["model"])

        # Create trainer
        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            loss_class=eval(config["loss_class"]),
            metric="accuracy",
            batch_size=config["batch_size"],
            num_iterations=config["num_iterations"],
            num_epochs=config["num_epochs"],
            learning_rate=config["learning_rate"],
            warmup_proportion=config["warmup_proportion"],
            use_amp=True,
            column_mapping={"preprocessed": "text", "label": "label"}
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate()
        print(metrics)


        
        # Flush memory
        gc.collect()
        torch.cuda.empty_cache()

        # Flush memory
        del trainer

        return metrics, model

metrics, model = train(config=config)

while metrics['accuracy'] < 0.83: 

    del model
    print('training from atop.')
    metrics, model = train(config=config)



# test each model on the test set
def save_preds(model, df):
        
    # Run inference
    preds = model(df.cleaned_text.values.tolist())

    preds_df = pd.DataFrame({ 
         'preprocessed': df.loc[:, 'preprocessed'],
         'created_at': df.loc[:, 'created_at'],
         'author_id': df.loc[:, 'author_id'],
         'id': df.loc[:, 'id'],
         'label': df.loc[:, 'label'],
         'preds': preds
    })
    
    FILE_NAME = f"calgary_80_20_fewshot_preds_{DATE}.csv"
    preds_df.to_csv(PREDS_FOLDER + FILE_NAME, index=False)
        
    # Step 4: Create the confusion matrix
    cm = confusion_matrix(preds_df.loc[:, 'label'], preds_df.loc[:, 'preds'])

    # Step 5: Convert the confusion matrix to a DataFrame (optional)
    print('cm=\n')
    print(cm)

    return preds


save_preds(model, df)