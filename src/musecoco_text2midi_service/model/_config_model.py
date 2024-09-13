from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Text2AttributeConfig:
    model_name_or_path: str
    test_file: str
    attributes_file: str
    num_labels_file: str
    output_dir: str
    overwrite_output_dir: bool

@dataclass
class Attribute2MusicConfig:
    start: int
    end: int
    model_size: str
    k: int
    need_num: int
    temp: float
    ngram: int
    datasets_name: str
    checkpoint_name: str
    batch_size: int
    date: str

@dataclass
class PathsConfig:
    DATA_DIR: str
    checkpoint_path: str
    ctrl_command_path: str
    save_root: str
    log_root: str

@dataclass
class EnvironmentConfig:
    CUDA_VISIBLE_DEVICES: str

@dataclass
class Config:
    text2attribute: Text2AttributeConfig
    attribute2music: Attribute2MusicConfig
    paths: PathsConfig
    environment: EnvironmentConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text2attribute': vars(self.text2attribute),
            'attribute2music': vars(self.attribute2music),
            'paths': vars(self.paths),
            'environment': vars(self.environment)
        }
