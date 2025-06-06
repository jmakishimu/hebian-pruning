import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IterativePruner:
    def __init__(self, model_name, device):
        """
        Initializes the Pruner with a model name and device.
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.teacher_model = None
        self.student_model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.metrics = evaluate.combine(["accuracy", "f1"])

    def load_and_prepare_dataset(self, dataset_name, task, text_fields):
        """
        Loads and tokenizes the dataset.
        """
        logger.info(f"Loading and preparing dataset: {dataset_name} ({task})")
        dataset = load_dataset(dataset_name, task)

        def preprocess_function(examples):
            # The tokenizer will handle padding and truncation.
            return self.tokenizer(examples[text_fields[0]], examples[text_fields[1]], truncation=True)

        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        self.train_dataset = tokenized_datasets['train']
        self.eval_dataset = tokenized_datasets['validation']
        logger.info("Dataset loaded and preprocessed.")

    def load_models(self):
        """
        Loads the teacher and student models. They start as identical copies.
        """
        logger.info(f"Loading teacher and student models from {self.model_name}")
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.student_model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        # Ensure student model is a deep copy
        self.student_model.load_state_dict(self.teacher_model.state_dict())
        logger.info("Models loaded.")

    def compute_metrics(self, p):
        """
        Computes accuracy and F1 score for evaluation.
        """
        preds = np.argmax(p.predictions, axis=1)
        return self.metrics.compute(predictions=preds, references=p.label_ids)

    def train_model(self, model, output_dir, num_epochs=3):
        """
        A generic training function for a given model.
        This uses the modern TrainingArguments for new versions of the transformers library.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./{output_dir}-logs',
            logging_steps=50,
            evaluation_strategy="epoch",  # Modern argument
            save_strategy="epoch",        # Modern argument
            load_best_model_at_end=True,
            fp16=True if self.device.type == 'cuda' else False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        return trainer

    def compute_gradient_importance(self):
        """
        Computes the importance of each weight as the absolute value of its gradient.
        This requires a forward and backward pass on a batch of data.
        """
        logger.info("Computing gradient importance for pruning...")
        # Select a subset of the training data for gradient calculation
        data_loader = torch.utils.data.DataLoader(self.train_dataset.select(range(128)), batch_size=8, collate_fn=self.teacher_model.data_collator)

        importance = {name: torch.zeros_like(param) for name, param in self.student_model.named_parameters() if 'weight' in name and param.requires_grad}

        self.student_model.train()
        for batch in data_loader:
            # Move batch to the correct device
            inputs = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.student_model(**inputs)
            loss = outputs.loss
            loss.backward()

            for name, param in self.student_model.named_parameters():
                if param.grad is not None and name in importance:
                    importance[name] += param.grad.abs() # Accumulate absolute gradients

            self.student_model.zero_grad()
        
        return importance

    def hebbian_inspired_pruning(self, importance_scores, prune_percent):
        """
        Prunes the model based on gradient importance scores.
        This is a form of magnitude pruning on the gradients, not true Hebbian learning.
        """
        logger.info(f"Applying Hebbian-inspired pruning with prune percent: {prune_percent}%...")
        for name, module in self.student_model.named_modules():
            if isinstance(module, nn.Linear):
                # We need to apply the mask to the 'weight' parameter of the module
                param_name = f"{name}.weight"
                if param_name in importance_scores:
                    prune.l1_unstructured(module, name='weight', amount=prune_percent / 100.0)
                    # Make the pruning permanent
                    prune.remove(module, 'weight')
        logger.info("Pruning applied.")


    def distill_knowledge(self, teacher_trainer, num_epochs=2):
        """
        Fine-tunes the student model using knowledge distillation.
        The loss is a combination of the standard cross-entropy loss and a distillation loss.
        """
        logger.info("Distilling knowledge from teacher to student...")

        class DistillationTrainer(Trainer):
            def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
                super().__init__(*args, **kwargs)
                self.teacher_model = teacher_model
                self.alpha = alpha
                self.temperature = temperature
                self.teacher_model.to(self.device)

            def compute_loss(self, model, inputs, return_outputs=False):
                # Student's output
                student_outputs = model(**inputs)
                student_loss = student_outputs.loss

                # Teacher's output
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)

                # Distillation loss (KL Divergence)
                loss_fct = nn.KLDivLoss(reduction="batchmean")
                distillation_loss = loss_fct(
                    nn.functional.log_softmax(student_outputs.logits / self.temperature, dim=-1),
                    nn.functional.softmax(teacher_outputs.logits / self.temperature, dim=-1)
                ) * (self.temperature ** 2)

                # Combined loss
                loss = self.alpha * student_loss + (1. - self.alpha) * distillation_loss
                return (loss, student_outputs) if return_outputs else loss

        training_args = TrainingArguments(
            output_dir='./results_student_distilled',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_steps=50,
            evaluation_strategy="epoch",  # Modern argument
            save_strategy="epoch",        # Modern argument
            load_best_model_at_end=True,
            fp16=True if self.device.type == 'cuda' else False,
        )

        distillation_trainer = DistillationTrainer(
            model=self.student_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            teacher_model=self.teacher_model,
        )

        distillation_trainer.train()
        return distillation_trainer

    def evaluate_model(self, model, trainer):
        """
        Evaluates a given model using the provided trainer's evaluation loop.
        """
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate(eval_dataset=self.eval_dataset)
        return eval_results

    def count_model_parameters(self, model):
        """
        Counts the number of trainable and non-zero parameters.
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        nonzero_params = sum(p.count_nonzero().item() for p in model.parameters())
        return trainable_params, nonzero_params

    def save_model_for_size_check(self, model, model_name):
        """
        Saves the model state dict to a temporary file to check its size on disk.
        """
        temp_path = f"{model_name}.bin"
        torch.save(model.state_dict(), temp_path)
        size = os.path.getsize(temp_path) / (1024 ** 2)
        os.remove(temp_path)
        return size

    def plot_results(self, baseline_metrics, pruned_metrics, baseline_params, pruned_params, baseline_size, pruned_size):
        """
        Generates and displays plots comparing the baseline and pruned models.
        """
        logger.info("Plotting results...")
        metrics_labels = ['Accuracy', 'F1 Score']
        baseline_scores = [baseline_metrics.get('eval_accuracy', 0), baseline_metrics.get('eval_f1', 0)]
        pruned_scores = [pruned_metrics.get('eval_accuracy', 0), pruned_metrics.get('eval_f1', 0)]

        x = np.arange(len(metrics_labels))
        width = 0.35
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # --- Metrics Plot ---
        rects1 = ax1.bar(x - width/2, baseline_scores, width, label='Baseline (Teacher)')
        rects2 = ax1.bar(x + width/2, pruned_scores, width, label='Pruned (Student)')

        ax1.set_ylabel('Scores')
        ax1.set_title('Performance: Baseline vs. Pruned Model')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_labels)
        ax1.legend()
        ax1.bar_label(rects1, padding=3, fmt='%.3f')
        ax1.bar_label(rects2, padding=3, fmt='%.3f')
        ax1.set_ylim(0, 1)

        # --- Compression Plot ---
        comp_labels = ['Parameters (Non-Zero)', 'Model Size (MB)']
        baseline_values = [baseline_params[1], baseline_size]
        pruned_values = [pruned_params[1], pruned_size]
        x_comp = np.arange(len(comp_labels))

        rects3 = ax2.bar(x_comp - width/2, baseline_values, width, label='Baseline (Teacher)')
        rects4 = ax2.bar(x_comp + width/2, pruned_values, width, label='Pruned (Student)')

        ax2.set_ylabel('Count')
        ax2.set_title('Compression: Parameters and Size')
        ax2.set_xticks(x_comp)
        ax2.set_xticklabels(comp_labels)
        ax2.legend()
        ax2.bar_label(rects3, padding=3, fmt='%.2fM', labels=[f'{v/1e6:.2f}M' for v in baseline_values])
        ax2.bar_label(rects4, padding=3, fmt='%.2fM', labels=[f'{v/1e6:.2f}M' for v in pruned_values])
        ax2.set_yscale('log')

        fig.tight_layout()
        plt.savefig("pruning_results.png")
        logger.info("Results plot saved to pruning_results.png")
        plt.show()

if __name__ == "__main__":
    # --- Configuration ---
    MODEL_NAME = 'bert-base-uncased'
    DATASET_NAME = 'glue'
    DATASET_TASK = 'mrpc' # MRPC is a small dataset, good for quick experiments
    TEXT_FIELDS = ['sentence1', 'sentence2']
    PRUNE_PERCENT = 40.0 # Prune 40% of the weights
    DEVICE = "cuda"

    # --- Initialization ---
    pruner = IterativePruner(model_name=MODEL_NAME, device=DEVICE)
    pruner.load_and_prepare_dataset(DATASET_NAME, DATASET_TASK, TEXT_FIELDS)
    pruner.load_models()

    # --- Step 1: Fine-tune the Teacher Model ---
    logger.info("--- Step 1: Training the baseline teacher model ---")
    teacher_trainer = pruner.train_model(pruner.teacher_model, output_dir='./results_teacher', num_epochs=3)
    teacher_metrics = pruner.evaluate_model(pruner.teacher_model, teacher_trainer)
    logger.info(f"Teacher Model Evaluation Results: {teacher_metrics}")

    # --- Step 2: Prune the Student Model ---
    logger.info("--- Step 2: Pruning the student model ---")
    # First, fine-tune the student model a little bit to get meaningful gradients
    student_trainer_pre_prune = pruner.train_model(pruner.student_model, output_dir='./results_student_pre_prune', num_epochs=1)
    
    # Compute importance and prune
    importance = pruner.compute_gradient_importance()
    pruner.hebbian_inspired_pruning(importance, PRUNE_PERCENT)

    # --- Step 3: Fine-tune the Pruned Student with Knowledge Distillation ---
    logger.info("--- Step 3: Fine-tuning pruned student with knowledge distillation ---")
    student_trainer_final = pruner.distill_knowledge(teacher_trainer, num_epochs=3)
    student_metrics = pruner.evaluate_model(pruner.student_model, student_trainer_final)
    logger.info(f"Final Pruned Student Model Evaluation Results: {student_metrics}")

    # --- Step 4: Analysis and Plotting ---
    logger.info("--- Step 4: Analyzing results ---")
    # Get model sizes
    teacher_size = pruner.save_model_for_size_check(pruner.teacher_model, "teacher_model")
    student_size = pruner.save_model_for_size_check(pruner.student_model, "student_model")

    # Get parameter counts
    teacher_params = pruner.count_model_parameters(pruner.teacher_model)
    student_params = pruner.count_model_parameters(pruner.student_model)

    logger.info(f"Baseline Teacher | Total Params: {teacher_params[0]/1e6:.2f}M | Non-Zero Params: {teacher_params[1]/1e6:.2f}M | Size: {teacher_size:.2f} MB")
    logger.info(f"Pruned Student   | Total Params: {student_params[0]/1e6:.2f}M | Non-Zero Params: {student_params[1]/1e6:.2f}M | Size: {student_size:.2f} MB")
    
    reduction_percent = (1 - student_params[1] / teacher_params[1]) * 100
    logger.info(f"Parameter reduction: {reduction_percent:.2f}%")

    # Plot the final comparison
    pruner.plot_results(
        teacher_metrics,
        student_metrics,
        teacher_params,
        student_params,
        teacher_size,
        student_size
    )