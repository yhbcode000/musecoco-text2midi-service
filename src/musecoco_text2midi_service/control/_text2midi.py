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
        
        argv_backup = sys.argv

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
        date = attribute2music_config.date
        max_len = attribute2music_config.max_len
        min_len = attribute2music_config.min_len
        max_positions = attribute2music_config.max_positions

        # Store configuration for dynamic save_root generation
        self.model_size = model_size
        self.checkpoint_name = checkpoint_name
        self.k = k
        self.temp = temp
        self.ngram = ngram
        self.paths_config = paths_config
        self.base_date = date

        # Step 4: Define paths
        DATA_DIR = paths_config.DATA_DIR.format(datasets_name=datasets_name)
        checkpoint_path = paths_config.checkpoint_path.format(model_size=model_size, checkpoint_name=checkpoint_name)
        ctrl_command_path = paths_config.ctrl_command_path
        save_root = paths_config.save_root.format(date=date, model_size=model_size, checkpoint_name=checkpoint_name, k=k, temp=temp, ngram=ngram)
        log_root = paths_config.log_root.format(date=date, model_size=model_size)

        # Step 5: Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = env_config.CUDA_VISIBLE_DEVICES

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
            "--max-len-b", str(max_len),
            "--min-len", str(min_len),
            "--sampling",
            "--beam", "1",
            "--sampling-topk", str(k),
            "--temperature", str(temp),
            "--no-repeat-ngram-size", str(ngram),
            "--buffer-size", str(BATCH_SIZE),
            "--batch-size", str(BATCH_SIZE),
            "--max-target-positions", str(max_positions)
        ]

        self.attribute2midi_predictor = init_attribute2midi()
        
        sys.argv = argv_backup
        self.save_root = save_root
        
        # Set input and output paths
        self.input_json_path = "storage/input/predict.json"
        self.output_bin_path = "storage/tmp/infer_test.bin"

    def __process_input_change(self):
        """Callback for when input JSON file changes."""
        print("Input JSON file changed. Running text2attribute prediction...")
        self.text2attribute_predictor.predict()
        prepare_stage2(self.source_path, self.destination_path)  # Prepare for stage 2
        print("New bin file generated. Running attribute2midi prediction...")
        self.attribute2midi_predictor.predict()

    def text_to_midi(self, input_text, return_midi=False):
        """Function to take string input and return MIDI data with metadata."""
        # Generate unique save_root with timestamp to preserve previous generations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_date = f"{self.base_date}_{timestamp}"
        unique_save_root = self.paths_config.save_root.format(
            date=unique_date,
            model_size=self.model_size,
            checkpoint_name=self.checkpoint_name,
            k=self.k,
            temp=self.temp,
            ngram=self.ngram
        )

        # Update predictor's save_root to the new unique path
        self.attribute2midi_predictor.save_root = unique_save_root
        os.makedirs(unique_save_root, exist_ok=True)

        # Save input text to the target directory
        with open(self.input_json_path, "w") as file:
            json.dump([{"text": input_text}], file)

        self.__process_input_change()

        # Read the MIDI data and return it with metadata
        midi_files = []
        if os.path.isdir(unique_save_root):
            for subdir in os.listdir(unique_save_root):
                midi_dir = os.path.join(unique_save_root, subdir, "midi")
                if not os.path.isdir(midi_dir):
                    continue
                for entry in os.listdir(midi_dir):
                    if entry.lower().endswith(".mid"):
                        midi_files.append(os.path.join(midi_dir, entry))

        if not midi_files:
            raise FileNotFoundError("No MIDI files generated in save_root directory.")

        midi_path = max(midi_files, key=os.path.getctime)

        metadata = {
            "time_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": midi_path,
            "save_root": unique_save_root
        }

        if return_midi:
            with open(midi_path, "rb") as midi_file:
                midi_data = midi_file.read()
        else:
            midi_data = None

        return midi_data, metadata

    def continue_midi_generation(self, original_save_root, return_midi=False):
        """
        Continue MIDI generation from a previous job using REMI tokens as prefix.

        Args:
            original_save_root: Path to the save_root directory of the original generation
            return_midi: Whether to return MIDI binary data

        Returns:
            Tuple of (midi_data, metadata) where midi_data is binary if return_midi=True, else None
        """
        # Find the REMI token file and infer_command.json from the original generation
        remi_file_path = None
        infer_command_path = None

        # Look for files in the save_root directory structure
        if os.path.isdir(original_save_root):
            for subdir in os.listdir(original_save_root):
                subdir_path = os.path.join(original_save_root, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                # Check for infer_command.json
                command_file = os.path.join(subdir_path, "infer_command.json")
                if os.path.exists(command_file):
                    infer_command_path = command_file

                # Check for REMI tokens
                remi_dir = os.path.join(subdir_path, "remi")
                if os.path.isdir(remi_dir):
                    for remi_file in os.listdir(remi_dir):
                        if remi_file.endswith(".txt"):
                            remi_file_path = os.path.join(remi_dir, remi_file)
                            break

                if remi_file_path and infer_command_path:
                    break

        if not remi_file_path:
            raise FileNotFoundError(f"No REMI token file found in {original_save_root}")
        if not infer_command_path:
            raise FileNotFoundError(f"No infer_command.json found in {original_save_root}")

        # Load REMI tokens
        with open(remi_file_path, "r") as f:
            remi_str = f.read().strip()

        # Extract just the REMI tokens (after <sep> token)
        tokens = remi_str.split(" ")
        try:
            sep_index = tokens.index("<sep>")
            remi_tokens = tokens[sep_index + 1:]
        except ValueError:
            # If no <sep> found, assume all tokens are REMI tokens
            remi_tokens = tokens

        # Load attribute dictionary
        with open(infer_command_path, "r") as f:
            attribute_dict = json.load(f)

        # Calculate new max_len (2x the original REMI token count)
        original_token_count = len(remi_tokens)
        new_max_len = original_token_count * 2

        print(f"Continuing generation from {original_token_count} tokens to {new_max_len} tokens")

        # Generate unique save_root for continuation with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_date = f"{self.base_date}_continued_{timestamp}"
        unique_save_root = self.paths_config.save_root.format(
            date=unique_date,
            model_size=self.model_size,
            checkpoint_name=self.checkpoint_name,
            k=self.k,
            temp=self.temp,
            ngram=self.ngram
        )

        # Update predictor's save_root
        self.attribute2midi_predictor.save_root = unique_save_root
        os.makedirs(unique_save_root, exist_ok=True)

        # Call the new predict_with_prefix method
        midi_path = self.attribute2midi_predictor.predict_with_prefix(
            attribute_dict=attribute_dict,
            remi_prefix_tokens=remi_tokens,
            custom_max_len=new_max_len
        )

        metadata = {
            "time_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": midi_path,
            "original_token_count": original_token_count,
            "new_max_len": new_max_len,
            "continuation": True
        }

        if return_midi:
            with open(midi_path, "rb") as midi_file:
                midi_data = midi_file.read()
        else:
            midi_data = None

        return midi_data, metadata

    def generate_with_context(self, original_text, new_text, original_save_root, return_midi=False):
        """
        Generate MIDI with combined text context - FRESH generation (no prefix).

        This method:
        1. Concatenates original_text + new_text
        2. Runs Text2Attribute on combined text → generates NEW attributes
        3. Generates FRESH music from combined text (no REMI prefix)

        Note: The model cannot perform modification with prefix, so this is a fresh generation.

        Args:
            original_text: Original text description from previous job
            new_text: New text to append/combine
            original_save_root: Path to the save_root directory (not used, kept for API compatibility)
            return_midi: Whether to return MIDI binary data

        Returns:
            Tuple of (midi_data, metadata) where midi_data is binary if return_midi=True, else None
        """
        # Combine texts
        combined_text = original_text + " " + new_text
        print(f"[Modification] Combined text: '{combined_text}'")
        print(f"[Modification] Generating FRESH music from combined text (no prefix)")

        # Simply call text_to_midi with the combined text for fresh generation
        midi_data, base_metadata = self.text_to_midi(combined_text, return_midi=return_midi)

        # Enhance metadata with modification-specific info
        enhanced_metadata = {
            **base_metadata,
            "original_text": original_text,
            "new_text": new_text,
            "combined_text": combined_text,
            "context_generation": True,
            "modification_type": "fresh_generation"
        }

        return midi_data, enhanced_metadata

    def context_continue_generation(self, original_text, new_text, original_save_root, return_midi=False):
        """
        Generate extended MIDI with combined text context - same as continue but with NEW attributes.

        This method:
        1. Concatenates original_text + new_text
        2. Runs Text2Attribute on combined text → generates NEW attributes
        3. Loads previous REMI tokens as FULL prefix
        4. Generates with NEW attributes + FULL prefix + DOUBLED max_len

        This is identical to continue_midi_generation, except it uses NEW attributes from
        the combined text instead of the original attributes.

        Args:
            original_text: Original text description from previous job
            new_text: New text to append/combine
            original_save_root: Path to the save_root directory of the original generation
            return_midi: Whether to return MIDI binary data

        Returns:
            Tuple of (midi_data, metadata) where midi_data is binary if return_midi=True, else None
        """
        # Combine texts
        combined_text = original_text + " " + new_text
        print(f"[Context-Continue] Combined text: '{combined_text}'")

        # Generate unique save_root for this generation with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_date = f"{self.base_date}_ctxcont_{timestamp}"
        unique_save_root = self.paths_config.save_root.format(
            date=unique_date,
            model_size=self.model_size,
            checkpoint_name=self.checkpoint_name,
            k=self.k,
            temp=self.temp,
            ngram=self.ngram
        )
        os.makedirs(unique_save_root, exist_ok=True)

        # Run Text2Attribute on combined text to get NEW attributes
        with open(self.input_json_path, "w") as file:
            json.dump([{"text": combined_text}], file)

        self.text2attribute_predictor.predict()
        from ._musecoco.view import prepare_stage2
        prepare_stage2(self.source_path, self.destination_path)

        # Load the newly generated attributes
        attribute_dict = None
        with open(self.output_bin_path, "rb") as f:
            import pickle
            test_command = pickle.load(f)
            if len(test_command) > 0:
                attribute_dict = test_command[0]

        if not attribute_dict:
            raise ValueError("Failed to generate attributes from combined text")

        print(f"[Context-Continue] Generated new attributes from combined text")

        # Load previous REMI tokens from original save_root (FULL prefix, not short)
        remi_file_path = None
        if os.path.isdir(original_save_root):
            for subdir in os.listdir(original_save_root):
                subdir_path = os.path.join(original_save_root, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                remi_dir = os.path.join(subdir_path, "remi")
                if os.path.isdir(remi_dir):
                    for remi_file in os.listdir(remi_dir):
                        if remi_file.endswith(".txt"):
                            remi_file_path = os.path.join(remi_dir, remi_file)
                            break
                if remi_file_path:
                    break

        if not remi_file_path:
            raise FileNotFoundError(f"No REMI token file found in {original_save_root}")

        # Load REMI tokens
        with open(remi_file_path, "r") as f:
            remi_str = f.read().strip()

        # Extract just the REMI tokens (after <sep> token)
        tokens = remi_str.split(" ")
        try:
            sep_index = tokens.index("<sep>")
            remi_tokens = tokens[sep_index + 1:]
        except ValueError:
            remi_tokens = tokens

        original_token_count = len(remi_tokens)
        print(f"[Context-Continue] Loaded {original_token_count} REMI tokens from previous generation")
        print(f"[Context-Continue] Using FULL prefix (all {original_token_count} tokens)")

        # DOUBLED max_len (like continue-generate)
        doubled_max_len = original_token_count * 2
        print(f"[Context-Continue] Using doubled max_len: {doubled_max_len} (2x {original_token_count})")

        # Update predictor's save_root
        self.attribute2midi_predictor.save_root = unique_save_root

        # Call predict_with_prefix with NEW attributes + FULL REMI prefix + DOUBLED max_len
        # This is identical to continue_generate except we use NEW attributes
        midi_path = self.attribute2midi_predictor.predict_with_prefix(
            attribute_dict=attribute_dict,
            remi_prefix_tokens=remi_tokens,  # FULL prefix
            custom_max_len=doubled_max_len
        )

        metadata = {
            "time_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": midi_path,
            "save_root": unique_save_root,
            "original_text": original_text,
            "new_text": new_text,
            "combined_text": combined_text,
            "prefix_token_count": original_token_count,
            "doubled_max_len": doubled_max_len,
            "context_continue_generation": True,
            "note": "Uses FULL prefix with NEW attributes from combined text"
        }

        if return_midi:
            with open(midi_path, "rb") as midi_file:
                midi_data = midi_file.read()
        else:
            midi_data = None

        return midi_data, metadata
