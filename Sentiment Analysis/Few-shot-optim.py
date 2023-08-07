import wandb
import random
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import confusion_matrix
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



# DATASET CREATION------------------------------------------------------
BASE_FOLDER = "---"

TRAIN_PATH = BASE_FOLDER + f'training_set.csv'
TEST_PATH = BASE_FOLDER + f'testing_set.csv'

dataset = load_dataset('csv', data_files={'train': TRAIN_PATH,
                                          'valid': TEST_PATH})

PROJECT_SWEEP_NAME = "calgary-4label-sentiment-analysis"

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
    },
}
parameters = {
    # defining constant parameters
    # 'dataset': {'value': 'SetFit/sst2'},
    'model': {'value': "sentence-transformers/all-mpnet-base-v2"},
    # CHANGE TO 'sentence-transformers/paraphrase-mpnet-base-v2'
    # to see what happens

    # define different types of losses for contrastive learning
    # these losses comes from sentence_transformers library
    'loss_class': {
        'values': [
            "CosineSimilarityLoss", 
            "ContrastiveLoss", 
            "BatchAllTripletLoss", 
            "BatchHardTripletLoss"
        ]
    },

    'batch_size': {
        # integers between 4 and 64
        'distribution': 'categorical',
        'values': [4, 8, 16, 24, 32, 64, 128]
    },

    'num_iterations': {'values': [5, 10, 20, 30, 50]},
    
    'num_epochs': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 5
    },

    'learning_rate': {
        'distribution': 'categorical',
        'values': [1e-6, 5e-6, 1e-5, 5e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4],
    },

    'warmup_proportion': {'values': [0.0, 0.1, 0.2, 0.5, 0.7]},
}

# adding the hyperparameters to the parameters field in the sweep_config dictionary
sweep_config['parameters'] = parameters
sweep_config




# FUNCTION FOR TRAINING


def train(config=None):
    
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        # print(config.name)

        # Simulate the few-shot regime
        # train_dataset, eval_dataset = prepare_dataset(config["dataset"])

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

        #save_preds(model, test_df, config)

        # log metrics to wandb
        wandb.log(metrics)
        
        # Flush memory
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

# initialize a W&B sweep
sweep_id = wandb.sweep(sweep_config, project=PROJECT_SWEEP_NAME)


# run the sweep
wandb.agent(sweep_id, train, count=50)