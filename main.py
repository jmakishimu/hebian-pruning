import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainerCallback
from datasets import load_dataset
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This callback re-applies the pruning mask after each optimization step
class PruningCallback(TrainerCallback):
    def __init__(self, masks):
        self.masks = masks

    def on_step_end(self, args, state, control, model=None, **kwargs):
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    param_name = f"{name}.weight"
                    if param_name in self.masks:
                        module.weight.data.mul_(self.masks[param_name])

def plot_loss_curves(histories):
    logger.info("Plotting training loss curves...")
    plt.figure(figsize=(12, 7))

    cumulative_steps = 0
    for i, history in enumerate(histories):
        logs = [log for log in history if 'loss' in log and 'eval_loss' not in log]
        if logs:
            steps = [log['step'] + cumulative_steps for log in logs]
            loss = [log['loss'] for log in logs]
            plt.plot(steps, loss, label=f'Cycle {i+1} Training', marker='o', linestyle='-')
            # Ensure cumulative_steps is updated correctly even for single-log histories
            if steps:
                cumulative_steps = steps[-1]

    plt.title('Cyclical Training Loss Curves')
    plt.xlabel('Total Training Steps')
    plt.ylabel('Loss')
    if cumulative_steps > 0: # Only show legend if there is data
        plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_curves.png")
    logger.info("Training loss curves plot saved to training_loss_curves.png")
    plt.close()

class IterativePruner:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.metrics = evaluate.combine(["accuracy", "f1"])
        self.activations = {}

    def load_and_prepare_dataset(self, dataset_name, task, text_fields, debug_mode=False):
        logger.info(f"Loading and preparing dataset: {dataset_name} ({task})")
        dataset = load_dataset(dataset_name, task)

        def preprocess_function(examples):
            return self.tokenizer(
                examples[text_fields[0]], examples[text_fields[1]],
                truncation=True, padding="max_length", max_length=128
            )

        columns_to_remove = [name for name in dataset['train'].column_names if name != 'label']
        tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=columns_to_remove)

        if debug_mode:
            logger.info("--- RUNNING IN DEBUG MODE ---")
            self.train_dataset = tokenized_datasets['train'].select(range(160))
            self.eval_dataset = tokenized_datasets['validation'].select(range(160))
        else:
            self.train_dataset = tokenized_datasets['train']
            self.eval_dataset = tokenized_datasets['validation']
        logger.info("Dataset loaded and preprocessed.")

    def load_model(self):
        logger.info(f"Loading model from {self.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        logger.info("Model loaded.")

    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        return self.metrics.compute(predictions=preds, references=p.label_ids)

    def train_model(self, output_dir, num_epochs=1, callbacks=None):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir=f'./{output_dir}-logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True if self.device.type == 'cuda' else False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks or []
        )
        trainer.train()
        return trainer

    def compute_hebbian_importance(self):
        logger.info("Computing Hebbian importance using activation hooks...")
        self.activations = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = {'input': input[0].detach(), 'output': output.detach()}
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(get_activation(name)))

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=8, collate_fn=data_collator)
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(data_loader))
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        importance_scores = {}
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and name in self.activations:
                    # Aggregate pre-synaptic (input) and post-synaptic (output) activations
                    input_tensor = self.activations[name]['input']
                    output_tensor = self.activations[name]['output']

                    if input_tensor.ndim == 3:
                        pre_activations = input_tensor.mean(dim=[0, 1])
                    elif input_tensor.ndim == 2:
                        pre_activations = input_tensor.mean(dim=0)
                    else:
                        continue

                    if output_tensor.ndim == 3:
                        post_activations = output_tensor.mean(dim=[0, 1])
                    elif output_tensor.ndim == 2:
                        post_activations = output_tensor.mean(dim=0)
                    else:
                        continue

                    # The outer product of post- and pre-synaptic activations
                    # gives a score matrix with the same shape as the weight matrix.
                    importance = torch.outer(post_activations.abs(), pre_activations.abs())
                    importance_scores[f"{name}.weight"] = importance

        return importance_scores

    def prune_model(self, importance_scores, prune_percent):
        logger.info(f"Applying pruning with prune percent: {prune_percent}%...")
        masks = {}
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    param_name = f"{name}.weight"
                    if param_name in importance_scores:
                        scores = importance_scores[param_name]
                        if scores.sum() == 0: continue

                        threshold = torch.quantile(scores.view(-1), prune_percent / 100.0)
                        mask = (scores > threshold).float().to(self.device)
                        module.weight.mul_(mask)
                        masks[param_name] = mask
        logger.info("Pruning applied.")
        return masks

    def regrow_weights(self, regrowth_percent):
        logger.info(f"Regrowing {regrowth_percent}% of pruned weights...")
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    zero_mask = (module.weight == 0)
                    num_to_regrow = int((regrowth_percent / 100.0) * zero_mask.sum())
                    if num_to_regrow > 0:
                        zero_indices = zero_mask.nonzero(as_tuple=False)
                        perm = torch.randperm(zero_indices.size(0))
                        indices_to_regrow = zero_indices[perm[:num_to_regrow]]
                        new_weights = torch.randn(num_to_regrow, device=self.device) * 0.01
                        module.weight[indices_to_regrow[:, 0], indices_to_regrow[:, 1]] = new_weights
        logger.info("Regrowth complete.")

    def evaluate_model(self, trainer):
        logger.info("Evaluating model...")
        return trainer.evaluate(eval_dataset=self.eval_dataset)

    def count_model_parameters(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        nonzero_params = sum(p.count_nonzero().item() for p in self.model.parameters())
        return trainable_params, nonzero_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args()

    MODEL_NAME = 'bert-base-uncased'
    DATASET_NAME = 'glue'
    DATASET_TASK = 'mrpc'
    TEXT_FIELDS = ['sentence1', 'sentence2']
    DEVICE = "cuda"

    NUM_PRUNING_CYCLES = 5
    PRUNE_PERCENT_PER_CYCLE = 10.0
    REGROWTH_PERCENT_PER_CYCLE = 1.0
    initial_epochs = 1 if args.debug else 3
    finetune_epochs_per_cycle = 1

    pruner = IterativePruner(model_name=MODEL_NAME, device=DEVICE)
    pruner.load_and_prepare_dataset(DATASET_NAME, DATASET_TASK, TEXT_FIELDS, debug_mode=args.debug)
    pruner.load_model()

    logger.info("--- Step 1: Initial fine-tuning of the model ---")
    initial_trainer = pruner.train_model(output_dir='./results_initial', num_epochs=initial_epochs)
    initial_metrics = pruner.evaluate_model(initial_trainer)
    initial_params = pruner.count_model_parameters()
    logger.info(f"Initial Model Evaluation: {initial_metrics}")
    logger.info(f"Initial Parameters (Non-Zero): {initial_params[1]/1e6:.2f}M")

    all_masks = {}
    training_histories = [initial_trainer.state.log_history]

    for cycle in range(NUM_PRUNING_CYCLES):
        logger.info(f"--- Starting Pruning Cycle {cycle + 1}/{NUM_PRUNING_CYCLES} ---")
        importance = pruner.compute_hebbian_importance()
        new_masks = pruner.prune_model(importance, PRUNE_PERCENT_PER_CYCLE)

        for name, mask in new_masks.items():
            if name not in all_masks:
                all_masks[name] = mask
            else:
                all_masks[name].mul_(mask)

        pruner.regrow_weights(REGROWTH_PERCENT_PER_CYCLE)

        logger.info(f"--- Fine-tuning after cycle {cycle + 1} ---")
        pruning_callback = PruningCallback(all_masks)
        finetune_trainer = pruner.train_model(
            output_dir=f'./results_cycle_{cycle+1}',
            num_epochs=finetune_epochs_per_cycle,
            callbacks=[pruning_callback]
        )
        training_histories.append(finetune_trainer.state.log_history)

        cycle_metrics = pruner.evaluate_model(finetune_trainer)
        cycle_params = pruner.count_model_parameters()
        logger.info(f"End of Cycle {cycle+1} Evaluation: {cycle_metrics}")
        logger.info(f"End of Cycle {cycle+1} Parameters (Non-Zero): {cycle_params[1]/1e6:.2f}M")

    logger.info("--- Final Analysis ---")
    final_params = pruner.count_model_parameters()
    logger.info(f"Initial Non-Zero Params: {initial_params[1]/1e6:.2f}M")
    logger.info(f"Final Non-Zero Params:   {final_params[1]/1e6:.2f}M")
    reduction_percent = (1 - final_params[1] / initial_params[1]) * 100 if initial_params[1] > 0 else 0
    logger.info(f"Total parameter reduction: {reduction_percent:.2f}%")
    logger.info("Experiment Complete.")

    plot_loss_curves(training_histories)

