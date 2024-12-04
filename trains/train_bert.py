from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import numpy as np

# Clase personalizada para manejar el dataset en PyTorch
class QA_Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

# Preprocesar el dataset con filtrado
def preprocess_data(dataset, tokenizer, max_samples=None):
    contexts = dataset["context"][:max_samples] if max_samples else dataset["context"]
    questions = dataset["question"][:max_samples] if max_samples else dataset["question"]
    answers = dataset["answers"][:max_samples] if max_samples else dataset["answers"]

    encodings = tokenizer(
        questions,
        contexts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    start_positions = []
    end_positions = []

    for i, answer in enumerate(answers):
        if len(answer["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        offsets = encodings["offset_mapping"][i]

        try:
            start_idx = next(idx for idx, offset in enumerate(offsets) if offset[0] <= start_char < offset[1])
            end_idx = next(idx for idx, offset in enumerate(offsets) if offset[0] < end_char <= offset[1])
        except StopIteration:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_positions.append(start_idx)
        end_positions.append(end_idx)

    encodings.update({
        "start_positions": start_positions,
        "end_positions": end_positions,
    })
    encodings.pop("offset_mapping")

    return encodings

# Métricas para QA
def compute_metrics(eval_preds):
    start_preds, end_preds = eval_preds.predictions
    start_labels, end_labels = eval_preds.label_ids

    start_preds = np.argmax(start_preds, axis=1)
    end_preds = np.argmax(end_preds, axis=1)

    exact_matches = (start_preds == start_labels) & (end_preds == end_labels)
    exact_match = np.mean(exact_matches)

    f1_scores = []
    for i in range(len(start_labels)):
        pred_range = set(range(start_preds[i], end_preds[i] + 1))
        label_range = set(range(start_labels[i], end_labels[i] + 1))
        overlap = len(pred_range & label_range)
        if overlap == 0:
            f1_scores.append(0)
        else:
            precision = overlap / len(pred_range)
            recall = overlap / len(label_range)
            f1_scores.append(2 * (precision * recall) / (precision + recall))

    f1 = np.mean(f1_scores)
    return {"exact_match": exact_match, "f1": f1}

# Función principal para entrenar el modelo con mejoras
def train_bert_qa_with_improvements(max_samples=None):
    # Cargar dataset XQuAD en español
    dataset = load_dataset("xquad", "xquad.es")
    train_data = dataset["validation"].train_test_split(test_size=0.2)["train"]
    eval_data = dataset["validation"].train_test_split(test_size=0.2)["test"]

    # Cargar modelo y tokenizador
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    model = BertForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")

    # Aplicar dropout adicional
    model.config.hidden_dropout_prob = 0.3
    model.config.attention_probs_dropout_prob = 0.3

    # Preprocesar datasets
    train_encodings = preprocess_data(train_data, tokenizer, max_samples=max_samples)
    eval_encodings = preprocess_data(eval_data, tokenizer, max_samples=max_samples)

    train_dataset = QA_Dataset(train_encodings)
    eval_dataset = QA_Dataset(eval_encodings)

    # Configuración del entrenamiento
    training_args = TrainingArguments(
        output_dir="../otros/bert_qa_results",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,  
        eval_steps=500,  
        warmup_steps=500,  
        learning_rate=5e-5,  
        num_train_epochs=6,  
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,  
        save_total_limit=2, 
        load_best_model_at_end=True, 
        logging_dir="../otros/bert_qa_logs",
        fp16=True,
        weight_decay=0.01, 
        report_to=["tensorboard"],
    )

    # Configurar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Entrenar el modelo
    trainer.train()

    # Guardar modelo y tokenizador
    model.save_pretrained("../models/bert_model_improved")
    tokenizer.save_pretrained("../models/bert_model_improved")
    print("Modelo y tokenizador guardados en '../models/bert_model_improved'")

if __name__ == "__main__":
    train_bert_qa_with_improvements(max_samples=1000)  # Entrenar con más muestras para mayor robustez
