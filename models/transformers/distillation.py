import torch
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
# Teacher Model (BERT)
teacher = BertForSequenceClassification.from_pretrained("models/transformers/bert-base-uncased")

# Student Model (DistilBERT)
student = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Load dataset
dataset = load_from_disk("data/raw/sst2")

# Distillation training
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        student_outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher(**inputs)
        
        # KL Divergence loss
        loss = torch.nn.functional.kl_div(
            torch.log_softmax(student_outputs.logits, dim=-1),
            torch.softmax(teacher_outputs.logits, dim=-1),
            reduction="batchmean"
        )
        return (loss, student_outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="models/transformers/distilled",
    per_device_train_batch_size=32,
    num_train_epochs=3
)

trainer = DistillationTrainer(
    model=student,
    args=training_args,
    train_dataset=dataset["train"]
)
trainer.train()
student.save_pretrained("models/transformers/distilled-bert")