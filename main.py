import os
import sys
from src.control.musecoco.text2attribute_model import main
from src.control.musecoco.attribute2music_model import interactive_dict_v5_1billion

if __name__ == "__main__":
    # Step 1: Simulate terminal input by modifying sys.argv for text2attribute model
    sys.argv = [
        "main.py",
        "--do_predict",
        "--model_name_or_path=IreneXu/MuseCoco_text2attribute",
        "--test_file=data/predict.json",
        "--attributes=data/att_key.json",
        "--num_labels=num_labels.json",
        "--output_dir=./tmp",
        "--overwrite_output_dir"
    ]

    # Call the main function to process simulated inputs
    main()

    # Step 2: Set up variables as in the shell script
    start, end = 0, 100  # Example values for start and end
    model_size = "1billion"
    k = 15
    command_name = "infer_test"
    need_num = 2
    temp = 1.0
    ngram = 0
    datasets_name = "truncated_2560"
    checkpoint_name = "checkpoint_2_280000"
    BATCH_SIZE = 2
    device = "1"
    date = "0505"
    model_name = f"linear_mask-{model_size}"
    
    # Step 3: Define paths
    DATA_DIR = f"../data/{datasets_name}"
    checkpoint_path = f"../checkpoints/{model_name}/{checkpoint_name}.pt"
    ctrl_command_path = f"../data/infer_input/{command_name}.bin"
    save_root = f"../generation/{date}/{model_name}-{checkpoint_name}/{command_name}/topk{k}-t{temp}-ngram{ngram}"
    log_root = f"../log/{date}/{model_name}"

    # Step 4: Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    # Step 5: Create necessary directories
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    # Step 6: Print messages similar to echo in shell
    print(f"generating from {checkpoint_path}")
    print(f"save to {save_root}")

    # Step 7: Change the directory (similar to 'cd' in shell)
    os.chdir("linear_mask")

    # Step 8: Set up the arguments for the interactive_dict_v5_1billion script
    # Reset sys.argv for cli_main
    sys.argv = [
        "interactive_dict_v5_1billion.py",
        f"{DATA_DIR}/data-bin",
        "--task", "language_modeling_control",
        "--path", checkpoint_path,
        "--ctrl_command_path", ctrl_command_path,
        "--save_root", save_root,
        "--need_num", str(need_num),
        "--start", str(start),
        "--end", str(end),
        "--max-len-b", "2560",
        "--min-len", "512",
        "--sampling",
        "--beam", "1",
        "--sampling-topk", str(k),
        "--temperature", str(temp),
        "--no-repeat-ngram-size", str(ngram),
        "--buffer-size", str(BATCH_SIZE),
        "--batch-size", str(BATCH_SIZE)
    ]

    # Step 9: Call cli_main with modified arguments
    interactive_dict_v5_1billion.seed_everything(2024)  # Set random seed
    interactive_dict_v5_1billion.cli_main()
