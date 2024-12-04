import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Cargar el dataset XQuAD en español
def load_xquad_spanish():
    dataset = load_dataset("xquad", "xquad.es")
    return dataset["validation"]

# Función de entrenamiento para generación de preguntas con XQuAD
def train_question_generation():
    dataset = load_xquad_spanish()

    # Dividir en entrenamiento y validación
    split_datasets = dataset.train_test_split(test_size=0.2)
    train_dataset, eval_dataset = split_datasets["train"], split_datasets["test"]

    # Cambiar a un modelo T5 más grande para mejorar la calidad
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Función de preprocesamiento
    def preprocess_examples(examples):
        inputs = []
        
        # Extraer el contexto, la pregunta y la respuesta
        for context, question, answers in zip(examples["context"], examples["question"], examples["answers"]):
            answer = answers["text"][0] if answers["text"] else "Sin respuesta"
            inputs.append({"context": context, "question": question, "answer": answer})

        # Tokenizar las entradas
        tokenized_inputs = tokenizer(
            ["context: " + ex["context"] + " answer: " + ex["answer"] for ex in inputs],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        tokenized_inputs["labels"] = tokenizer(
            [ex["question"] for ex in inputs],
            truncation=True,
            padding="max_length",
            max_length=128
        )["input_ids"]
        
        return tokenized_inputs

    # Mapear la función de tokenización en el conjunto de datos de entrenamiento
    tokenized_train_dataset = train_dataset.map(preprocess_examples, batched=True, remove_columns=["id", "context", "question", "answers"])

    # Configuración del entrenamiento
    training_args = TrainingArguments(
        output_dir="../otros/resultados_t5_pregunta",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        logging_dir="../otros/log_t5_pregunta",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,          
        weight_decay=0.01,              
        save_total_limit=1,             
        load_best_model_at_end=True,    
        metric_for_best_model="eval_loss" 
    )

    # Configurar el Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=eval_dataset.map(preprocess_examples, batched=True, remove_columns=["id", "context", "question", "answers"]),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    # Entrenar el modelo
    trainer.train()
    model.save_pretrained("../models/pregunta_model")
    tokenizer.save_pretrained("../models/pregunta_model")
    print("Entrenamiento completado y modelo guardado en '../models/pregunta_model'")

if __name__ == "__main__":
    train_question_generation()
