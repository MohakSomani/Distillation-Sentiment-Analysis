import torch
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Add evaluation and plotting imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer

# Teacher Model (BERT)
teacher = BertForSequenceClassification.from_pretrained("models/transformers/bert-base-uncased")

# Student Model (DistilBERT)
student = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Load dataset
dataset = load_from_disk("data/raw/sst2")

# Tokenize and prepare data for Trainer
from transformers import BertTokenizerFast, DataCollatorWithPadding
tokenizer = BertTokenizerFast.from_pretrained("models/transformers/bert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer)
def tokenize(batch):
    return tokenizer(batch["sentence"], padding=True, truncation=True, max_length=128)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

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

# Evaluate teacher and student on validation set
os.makedirs("logs", exist_ok=True)
val_dataset = dataset["validation"]
# Teacher evaluation
teacher_trainer = Trainer(model=teacher, args=training_args, eval_dataset=val_dataset, data_collator=data_collator)
teacher_pred = teacher_trainer.predict(val_dataset)
teacher_logits = teacher_pred.predictions
teacher_labels = teacher_pred.label_ids
teacher_pred_labels = np.argmax(teacher_logits, axis=1)
teacher_acc = accuracy_score(teacher_labels, teacher_pred_labels)
teacher_f1 = f1_score(teacher_labels, teacher_pred_labels, average='weighted')
# Student evaluation
student_trainer = Trainer(model=student, args=training_args, eval_dataset=val_dataset, data_collator=data_collator)
student_pred = student_trainer.predict(val_dataset)
student_logits = student_pred.predictions
student_pred_labels = np.argmax(student_logits, axis=1)
student_acc = accuracy_score(teacher_labels, student_pred_labels)
student_f1 = f1_score(teacher_labels, student_pred_labels, average='weighted')
# Save metrics to text file
with open("logs/distillation_metrics.txt", "w") as f:
    f.write("Model\tAccuracy\tF1\n")
    f.write(f"Teacher\t{teacher_acc}\t{teacher_f1}\n")
    f.write(f"Student\t{student_acc}\t{student_f1}\n")
# Plot comparison
metrics = ["Accuracy", "F1"]
values_teacher = [teacher_acc, teacher_f1]
values_student = [student_acc, student_f1]
x = np.arange(len(metrics))
width = 0.35
plt.figure(figsize=(8, 6))
plt.bar(x - width/2, values_teacher, width, label="Teacher")
plt.bar(x + width/2, values_student, width, label="Student")
plt.ylabel("Score")
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()
plt.title("Teacher vs Student Model Performance")
plt.savefig("logs/distillation_comparison.png")
plt.close()