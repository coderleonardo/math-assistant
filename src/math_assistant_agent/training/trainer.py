import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def load_training_dataset(jsonl_path):
    return load_dataset("json", data_files=jsonl_path, split="train")


def build_training_args(output_dir="./qwen-math-agent", **overrides):
    defaults = dict(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Reduzido para caber na memória (ajuste conforme sua GPU)
        gradient_accumulation_steps=4,  # Simula um batch_size maior sem usar mais VRAM
        learning_rate=2e-4,  # Taxa de aprendizado padrão para QLoRA
        logging_steps=10,
        max_steps=1,  # Apenas para o teste inicial! Depois altere para num_train_epochs
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",  # Otimizador eficiente em memória
        dataset_text_field="messages",  # O TRL reconhece a estrutura de chat automaticamente
        max_length=2048,  # Limite de tokens para o treinamento
        packing=False,
    )
    defaults.update(overrides)
    return SFTConfig(**defaults)


def build_trainer(model, training_args, dataset, tokenizer, peft_config):
    # O TRL vai envelopar o modelo com o peft_config pra você aqui
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def train_and_save_adapter(trainer, tokenizer, adapter_dir="./qwen-math-agent-adapter"):
    print("Iniciando o Fine-Tuning do seu Agente Matemático...")
    trainer.train()

    # Isso salva apenas os pesos treinados (poucos MBs), não o modelo base inteiro de 4B.
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print("Treinamento concluído e adaptadores salvos com sucesso!")
    return adapter_dir
