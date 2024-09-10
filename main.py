import os
import sys
import shutil
from src.control.musecoco.text2attribute_model import Text2AttributePredictor, prepare_data
from src.control.musecoco.attribute2music_model import interactive_dict_v5_1billion

def init_text2attribute():
    # Step 1: Simulate terminal input by modifying sys.argv for text2attribute model
    # Define variables
    model_name_or_path = "IreneXu/MuseCoco_text2attribute"
    test_file = "storage/input/predict.json"
    attributes_file = "src/control/musecoco/text2attribute_model/data/att_key.json"
    num_labels_file = "src/control/musecoco/text2attribute_model/num_labels.json"
    output_dir = "storage/tmp"

    # Convert Python variables into sys.argv format
    sys.argv = [
        "main.py",
        "--do_predict",
        f"--model_name_or_path={model_name_or_path}",
        f"--test_file={test_file}",
        f"--attributes={attributes_file}",
        f"--num_labels={num_labels_file}",
        f"--output_dir={output_dir}",
        "--overwrite_output_dir"
    ]

    # Call the main function to process simulated inputs
    predictor = Text2AttributePredictor()
    
    return predictor

def prepare_stage2():
    # Step 2: Prepare intermediate data by executing necessary scripts
    # Move to the directory for text2attribute model processing
    # Run `stage2_pre.py` - you mentioned it's a script that can be imported
    prepare_data()

    # Move generated `infer_test.bin` to the appropriate directory
    source_path = "infer_test.bin"
    destination_path = "storage/tmp/infer_test.bin"
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(source_path, destination_path)
    
def init_attribute2midi():
    # Step 3: Set up variables for attribute2music model
    start, end = 0, 100  # Example values for start and end
    model_size = "1billion"
    k = 15
    need_num = 2
    temp = 1.0
    ngram = 0
    datasets_name = "truncated_2560"
    checkpoint_name = "checkpoint_2_280000"
    BATCH_SIZE = 2
    device = "1"
    date = "0505"
    model_name = f"linear_mask-{model_size}"

    # Step 4: Define paths
    DATA_DIR = f"src/control/musecoco/attribute2music_model/data/{datasets_name}"
    checkpoint_path = f"src/control/musecoco/attribute2music_model/checkpoints/{model_name}/{checkpoint_name}.pt"
    ctrl_command_path = f"storage/tmp/infer_test.bin"
    save_root = f"storage/generation/{date}/{model_name}-{checkpoint_name}/topk{k}-t{temp}-ngram{ngram}"
    log_root = f"storage/log/{date}/{model_name}"

    # Step 5: Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    # Step 6: Create necessary directories
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    # Simulate the command-line arguments for the `interactive_1billion.sh` script
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
    
    return interactive_dict_v5_1billion.Attribute2MusicPredictor()

if __name__ == "__main__":
    text2attribute_predictor = init_text2attribute()
    attribute2midi_predictor = init_attribute2midi()
    
    text2attribute_predictor.predict()
    prepare_stage2()
    attribute2midi_predictor.predict()
