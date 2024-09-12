from musecoco_text2midi_service.control import Text2Midi
from musecoco_text2midi_service.dao import load_config_from_file
from musecoco_text2midi_service.view import AppView


def main():
    config = load_config_from_file("storage/config/main_config.yaml")
    text2midi = Text2Midi(config)
    while True:
        input_command = str(input("Give the text command: "))
        if input_command == "exit":
            break
        _, meta_data = text2midi.text_to_midi(input_text=input_command)
        print(meta_data)
    
def start():
    app_view = AppView("storage/config/main_config.yaml")
    app_view.run()


if __name__ == "__main__":
    main()