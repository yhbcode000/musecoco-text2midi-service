import sys
import os
import pytest
from musecoco_text2midi_service.control import Text2Midi

class TestText2Midi:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.text2midi = Text2Midi()

    def test_text_to_midi(self):
        """Test the text_to_midi function."""
        input_text = "This music embodies the essence of classic new age music with a pitch range within 3 octaves and the use of minor key, conveying a unique and resonant sound. The song is about 40 seconds long and has a brisk tempo with a tranquilizing beat. The music comes to life through the use of piano and voice and is based on a 4/4 time signature. Overall, this song showcases the quintessential features of new age music in a captivating and memorable way. Violin and ethnic instrument are not featured in this song."
        midi_data, metadata = self.text2midi.text_to_midi(input_text, return_midi=True)

        assert midi_data is not None, "MIDI data should not be None"
        assert "time_generated" in metadata, "Metadata should contain time_generated"
        assert "file_path" in metadata, "Metadata should contain file_path"
