
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
# import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("GPU Device Name:", tf.test.gpu_device_name())

# If GPU is available, ensure it's being used
if tf.config.list_physical_devices('GPU'):
    print("GPU is being used")
else:
    print("Warning: No GPU found, using CPU")

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_from_disk
import torch
import yaml

# Load config
with open("configs/bert_config.yaml") as f:
    config = yaml.safe_load(f)

# Load dataset
dataset = load_from_disk(f"data/raw/{config['dataset']}")
# limit dataset size for faster training
if config.get('max_train_samples'):
    dataset['train'] = dataset['train'].select(range(config['max_train_samples']))
if config.get('max_eval_samples'):
    # clamp eval count to actual validation size
    eval_limit = min(config.get('max_eval_samples'), len(dataset['validation']))
    dataset['validation'] = dataset['validation'].select(range(eval_limit))
tokenizer = BertTokenizer.from_pretrained(config["model_name"])

# Add a collator that pads inputs to the same length in each batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize(batch):
    return tokenizer(batch["sentence"], padding=True, truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# clear any unused GPU memory before training
torch.cuda.empty_cache()

# Model
model = BertForSequenceClassification.from_pretrained(config["model_name"], num_labels=2,    attn_implementation="eager" )
# enable gradient checkpointing within the model to reduce activation memory
model.gradient_checkpointing_enable()

# Training
training_args = TrainingArguments(
    output_dir="models/transformers/results",
    num_train_epochs=config["epochs"],
    per_device_train_batch_size=1,  # lower batch size to fit in memory
    gradient_accumulation_steps=2,  # accumulate gradients across steps
    gradient_checkpointing=True,  # save memory via recomputation
    fp16=True,  # use mixed precision
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator
)

trainer.train()
model.save_pretrained(f"models/transformers/{config['model_name']}")
# also save tokenizer for local use in interpretability scripts
tokenizer.save_pretrained(f"models/transformers/{config['model_name']}")