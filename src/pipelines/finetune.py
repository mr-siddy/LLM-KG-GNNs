import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

def run_finetuning():
    # ----- Step 1: Load and Parse SFT Data -----
    sft_file = "data/sft/SFT_data.txt"  # Adjust path as needed
    with open(sft_file, "r", encoding="utf-8") as f:
        sft_text = f.read()

    # We assume the file contains a header for the input and then a section for the output.
    # In our case, the file uses "#### Product Recommendations" as the delimiter.
    parts = sft_text.split("#### Product Recommendations")
    if len(parts) < 2:
        raise ValueError("Could not split SFT_data.txt into input and output parts using the delimiter.")
    # Everything before the delimiter is our input; everything after is the target output.
    input_part = parts[0].strip()
    output_part = parts[1].strip()

    # Create a single fine-tuning example as a dictionary.
    example = {"input": input_part, "output": output_part}
    print("Parsed Fine-Tuning Example:")
    print("Input Part:\n", input_part)
    print("\nOutput Part:\n", output_part)

    # ----- Step 2: Build a Dataset -----
    # Create a dataset (here with a single example; later you can expand to many examples)
    sft_dataset = Dataset.from_dict({
        "input": [example["input"]],
        "output": [example["output"]]
    })

    # ----- Step 3: Prepare the Model and Tokenizer for Fine-Tuning -----
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # ----- Step 4: Tokenization Function -----
    # We combine the input and output using a clear delimiter for training.
    def tokenize_record(record):
        # We use "\n\n### Response:\n" as a delimiter between input and expected output.
        full_text = record["input"] + "\n\n### Response:\n" + record["output"]
        return tokenizer(full_text, truncation=True, max_length=1024)

    tokenized_dataset = sft_dataset.map(tokenize_record, batched=False, remove_columns=["input", "output"])

    # ----- Step 5: Set Up Data Collator -----
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ----- Step 6: Define Training Arguments -----
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        overwrite_output_dir=True,
        num_train_epochs=3,                   # Adjust as needed
        per_device_train_batch_size=1,        # For a single example fine-tuning, batch size 1 is sufficient
        save_steps=10,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,                            # Enable FP16 if your GPU supports it
    )

    # ----- Step 7: Initialize the Trainer and Fine-Tune -----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\nStarting fine-tuning on the SFT data...")
    trainer.train()

    # ----- Step 8: Save the Fine-Tuned Model -----
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    print("\nFine-tuning complete. Model saved to ./finetuned_model")

    # ----- Testing the saved model -----
    from transformers import pipeline
    import torch
    # Load the fine-tuned model and tokenizer from the saved directory
    model_dir = "./finetuned_model"
    model_ft = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer_ft = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # Set device for inference (0 for CUDA if available, else -1 for CPU)
    device_ft = 0 if torch.cuda.is_available() else -1
    # Create a text-generation pipeline for the fine-tuned model
    ft_pipe = pipeline("text-generation", model=model_ft, tokenizer=tokenizer_ft, device=device_ft)
    # Construct a sample prompt in the same format as used for SFT dataset creation
    sample_prompt = (
        "Customer Profile:\nAge: 35, Membership: ACTIVE\n\n"
        "Purchase History:\n"
        "- Purchased a white vest top made of soft organic cotton with adjustable straps.\n"
        "- Bought several Cat Tee's in white, grey, and pink for everyday comfort.\n"
        "- Also acquired a green crew sweater with a relaxed, cozy fit and a grey hoodie for a sporty look.\n\n"
        "Based on the above purchase history and profile, please recommend 10 products that this customer is likely to enjoy next.\n\n"
        "### Response:"
    )
    # Generate recommendations using the fine-tuned model
    ft_output = ft_pipe(sample_prompt, max_new_tokens=150, do_sample=True, top_p=0.95, temperature=0.7)
    # Extract and print the output after the delimiter
    generated_text = ft_output[0]["generated_text"]
    if "### Response:" in generated_text:
        final_output = generated_text.split("### Response:")[-1].strip()
    else:
        final_output = generated_text.strip()

    print("Fine-Tuned Model Output:\n", final_output)

def main():
    run_finetuning()

if __name__ == "__main__":
    main()
