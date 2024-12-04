from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import random
import string
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import time
import gc

# Liberar memoria GPU
torch.cuda.empty_cache()
gc.collect()

# Definir el dispositivo globalmente
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

class TextCorrectionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        
        # Aumentado a 3000 textos
        texts = texts[:3000]
        
        print("\nPreparando textos para el dataset...")
        pbar = tqdm(total=len(texts), desc="Procesando textos", unit="textos")
        for text in texts:
            if text is not None:
                text = text[:200]
                noisy_text = create_noisy_text(text)
                self.inputs.append(noisy_text)
                self.targets.append(text)
                pbar.update(1)
        pbar.close()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_encoding = self.tokenizer(
            self.inputs[idx],
            max_length=100, 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            self.targets[idx],
            max_length=100,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

def create_noisy_text(text, error_probability=0.15):
    if not isinstance(text, str):
        return ""
    
    noisy_text = ""
    for char in text:
        if random.random() < error_probability:
            error_type = random.choice(['typo', 'number', 'space', 'case'])
            
            if error_type == 'typo':
                if char.isalpha():
                    if char.lower() in 'aeiou':
                        noisy_text += random.choice('aeiou')
                    else:
                        noisy_text += random.choice('bcdfghjklmnpqrstvwxyz')
                else:
                    noisy_text += char
            elif error_type == 'number':
                noisy_text += str(random.randint(0, 9))
            elif error_type == 'space':
                if char == ' ':
                    noisy_text += ''
                else:
                    noisy_text += ' ' + char
            elif error_type == 'case':
                noisy_text += char.swapcase()
        else:
            noisy_text += char
    return noisy_text

def print_with_timestamp(message):
    current_time = time.strftime("%H:%M:%S")
    print(f"[{current_time}] {message}")

# Cargar el dataset
print_with_timestamp("Iniciando carga del dataset DACSA...")
with tqdm(total=1, desc="Cargando dataset") as pbar:
    ds = load_dataset("ELiRF/dacsa", "spanish")
    pbar.update(1)
print_with_timestamp("Dataset cargado exitosamente!")

# Verificar la estructura del dataset
print("\nEstructura del dataset:", ds['train'].features)

# Extraer los textos de la columna 'article'
texts = ds['train']['article']
print_with_timestamp(f"Número de textos cargados: {len(texts)}")

# Inicializar el modelo y tokenizer
print_with_timestamp("\nDescargando modelo y tokenizer...")
with tqdm(total=2, desc="Inicializando modelo") as pbar:
    model_name = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pbar.update(1)
    print_with_timestamp("Tokenizer inicializado")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    pbar.update(1)
print_with_timestamp("Modelo cargado exitosamente")

# Crear el dataset personalizado
print_with_timestamp("\nPreparando dataset para entrenamiento...")
train_dataset = TextCorrectionDataset(texts, tokenizer)
print_with_timestamp(f"Dataset preparado con {len(train_dataset)} ejemplos")

# Preparar los datos para el entrenamiento
print_with_timestamp("\nConfigurando parámetros de entrenamiento...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=3e-5,
    warmup_steps=500,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=20,
    logging_first_step=True,
    report_to="none",
    fp16=False, 
    dataloader_num_workers=4,
    generation_max_length=100 
)

# Configurar el entrenador
print_with_timestamp("Configurando el entrenador...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
)

# Entrenar el modelo
print_with_timestamp("\n=== Iniciando entrenamiento ===")
trainer.train()
print_with_timestamp("¡Entrenamiento completado!")

# Guardar el modelo
print_with_timestamp("\nGuardando el modelo...")
with tqdm(total=1, desc="Guardando modelo") as pbar:
    model.save_pretrained("../models/modelo_corrector")
    tokenizer.save_pretrained("../models/modelo_corrector")
    pbar.update(1)
print_with_timestamp("Modelo guardado exitosamente!")

# Función para corregir texto
@torch.no_grad()
def correct_text(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=100)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_length=100,
        no_repeat_ngram_size=3,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ejemplo de uso
print_with_timestamp("\nProbando el modelo con ejemplos:")
ejemplos = [
    "E informe de la Agenc7ia Nacional de Océanos y Atmósfera",
    "La temperatuR a med1a del planeta ha aumenTado",
    "Los cient1ficos advi3rten sobre el camb1o climatico",
    "El niv3l del mar ha sub1do en los ultim0s años",
    "La biodivers1dad esta en pel1gro por la contam1nacion"
]

print("\nPruebas de corrección:")
for texto in ejemplos:
    print(f"\nOriginal: {texto}")
    try:
        corregido = correct_text(texto)
        print(f"Corregido: {corregido}")
    except Exception as e:
        print(f"Error al procesar: {str(e)}")
