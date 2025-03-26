import argparse
import sys

def run_inference():
    from pipelines import inference
    inference.main()

def run_finetuning():
    from pipelines import finetune
    finetune.main()

def run_golden():
    from pipelines import golden_gen
    golden_gen.main()

def main():
    parser = argparse.ArgumentParser(
        description="Run product recommendation pipelines: inference, finetuning, or generate golden examples."
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "finetune", "golden"],
        required=True,
        help="Choose pipeline to run: inference for few-shot inference, finetune for SFT fine-tuning, golden to generate a golden example prompt."
    )
    args = parser.parse_args()

    if args.mode == "inference":
        print("Running few-shot inference pipeline...")
        run_inference()
    elif args.mode == "finetune":
        print("Running fine-tuning pipeline...")
        run_finetuning()
    elif args.mode == "golden":
        print("Running golden example generation...")
        run_golden()
    else:
        sys.exit("Invalid mode selected.")

if __name__ == "__main__":
    main()
