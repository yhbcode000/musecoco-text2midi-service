import yaml
from ..model import Config, Text2AttributeConfig, Attribute2MusicConfig, PathsConfig, EnvironmentConfig

def load_config_from_file(file_path: str) -> Config:
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    # Instantiate nested data classes from the parsed YAML dictionary
    text2attribute = Text2AttributeConfig(**config_dict['text2attribute'])
    attribute2music = Attribute2MusicConfig(**config_dict['attribute2music'])
    paths = PathsConfig(**config_dict['paths'])
    environment = EnvironmentConfig(**config_dict['environment'])

    # Return the main configuration class
    return Config(
        text2attribute=text2attribute,
        attribute2music=attribute2music,
        paths=paths,
        environment=environment
    )

def save_config_to_file(config: Config, file_path: str) -> None:
    with open(file_path, 'w') as file:
        yaml.dump(config.to_dict(), file)
