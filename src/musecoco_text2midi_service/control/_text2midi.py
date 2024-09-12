import os
import json
from datetime import datetime
import sys

import yaml

from ..model import Config
from ._musecoco.view import init_text2attribute, prepare_stage2, init_attribute2midi

class Text2Midi:
    def __init__(self, config: Config):
        # Load the configuration using the Config data class
        self.config: Config = config

        # Access text2attribute configuration
        text2attr_config = self.config.text2attribute
        attribute2music_config = self.config.attribute2music
        paths_config = self.config.paths
        env_config = self.config.environment

        # Step 1: Simulate terminal input by modifying sys.argv for text2attribute model
        sys.argv = [
            "main.py",
            "--do_predict",
            f"--model_name_or_path={text2attr_config.model_name_or_path}",
            f"--test_file={text2attr_config.test_file}",
            f"--attributes={text2attr_config.attributes_file}",
            f"--num_labels={text2attr_config.num_labels_file}",
            f"--output_dir={text2attr_config.output_dir}",
        ]

        if text2attr_config.overwrite_output_dir:
            sys.argv.append("--overwrite_output_dir")

        self.text2attribute_predictor = init_text2attribute()

        # Set paths
        self.source_path = "infer_test.bin"
        self.destination_path = f"{text2attr_config.output_dir}/infer_test.bin"

        # Step 3: Set up variables for attribute2music model
        start = attribute2music_config.start
        end = attribute2music_config.end
        model_size = attribute2music_config.model_size
        k = attribute2music_config.k
        need_num = attribute2music_config.need_num
        temp = attribute2music_config.temp
        ngram = attribute2music_config.ngram
        datasets_name = attribute2music_config.datasets_name
        checkpoint_name = attribute2music_config.checkpoint_name
        BATCH_SIZE = attribute2music_config.batch_size
        device = attribute2music_config.device
        date = attribute2music_config.date

        # Step 4: Define paths
        DATA_DIR = paths_config.DATA_DIR.format(datasets_name=datasets_name)
        checkpoint_path = paths_config.checkpoint_path.format(model_size=model_size, checkpoint_name=checkpoint_name)
        ctrl_command_path = paths_config.ctrl_command_path
        save_root = paths_config.save_root.format(date=date, model_size=model_size, checkpoint_name=checkpoint_name, k=k, temp=temp, ngram=ngram)
        log_root = paths_config.log_root.format(date=date, model_size=model_size)

        # Step 5: Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = env_config.CUDA_VISIBLE_DEVICES.format(device=device)

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

        self.attribute2midi_predictor = init_attribute2midi()
        
        # Set input and output paths
        self.input_json_path = "storage/input/predict.json"
        self.output_bin_path = "storage/tmp/infer_test.bin"
        self.output_midi_dir = "storage/generation/0505/linear_mask-1billion-checkpoint_2_280000/topk15-t1.0-ngram0/0/midi"
        
        # # Set input and output paths # TODO need to add this to the config.
        # self.input_json_path = "modules/musecoco-text2midi-service/storage/input/predict.json"
        # self.output_bin_path = "modules/musecoco-text2midi-service/storage/tmp/infer_test.bin"
        # self.output_midi_dir = "modules/musecoco-text2midi-service/storage/generation/0505/linear_mask-1billion-checkpoint_2_280000/topk15-t1.0-ngram0/0/midi"



    def __process_input_change(self):
        """Callback for when input JSON file changes."""
        print("Input JSON file changed. Running text2attribute prediction...")
        self.text2attribute_predictor.predict()
        prepare_stage2(self.source_path, self.destination_path)  # Prepare for stage 2
        print("New bin file generated. Running attribute2midi prediction...")
        self.attribute2midi_predictor.predict()

    def text_to_midi(self, input_text, return_midi=False):
        """Function to take string input and return MIDI data with metadata."""
        # Save input text to the target directory
        with open(self.input_json_path, "w") as file:
            json.dump([{"text": input_text}], file)

        self.__process_input_change()

        # Read the MIDI data and return it with metadata
        midi_files = os.listdir(self.output_midi_dir)
        latest_midi_file = max(midi_files, key=lambda x: os.path.getctime(os.path.join(self.output_midi_dir, x)))
        midi_path = os.path.join(self.output_midi_dir, latest_midi_file)

        metadata = {
            "time_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": midi_path
        }
        
        if return_midi:
            with open(midi_path, "rb") as midi_file:
                midi_data = midi_file.read()
        else:
            midi_data = None

        return midi_data, metadata
