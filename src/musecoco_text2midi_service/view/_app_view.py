from flask import Flask, request, jsonify, send_file
from musecoco_text2midi_service.control import Text2Midi
from musecoco_text2midi_service.dao import load_config_from_file
import threading
import uuid
import os

class AppView:
    def __init__(self, config_path):
        # Initialize Flask app
        self.app = Flask(__name__)

        # Load configuration for Text2Midi service
        self.config = load_config_from_file(config_path)
        self.text2midi = Text2Midi(self.config)

        # Simulated job store
        self.job_store = {}

        # Setup routes
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/submit-text', methods=['POST'])
        def submit_text():
            data = request.json
            input_text = data.get('text')
            if not input_text:
                return jsonify({'error': 'Text input is required.'}), 400

            job_id = str(uuid.uuid4())
            self.job_store[job_id] = {'status': 'submitted'}

            # Start background thread for MIDI generation
            threading.Thread(target=self.generate_midi, args=(job_id, input_text)).start()

            return jsonify({'jobId': job_id, 'status': 'submitted'}), 202

        @self.app.route('/check-status/<job_id>', methods=['GET'])
        def check_status(job_id):
            job = self.job_store.get(job_id)
            if not job:
                return jsonify({'error': 'Job ID not found.'}), 404

            return jsonify({'jobId': job_id, 'status': job['status']}), 200

        @self.app.route('/get-result/<job_id>', methods=['GET'])
        def get_result(job_id):
            job = self.job_store.get(job_id)
            if not job:
                return jsonify({'error': 'Job ID not found.'}), 404

            if job['status'] != 'completed':
                return jsonify({'error': 'MIDI generation is not complete yet.'}), 400

            return jsonify({'jobId': job_id, 'status': job['status'], 'midiFilePath': job['result']['midiFilePath'], 'metaData': job['result']['metaData']}), 200

        @self.app.route('/download-midi/<job_id>', methods=['GET'])
        def download_midi(job_id):
            job = self.job_store.get(job_id)
            if not job:
                return jsonify({'error': 'Job ID not found.'}), 404

            if job['status'] != 'completed':
                return jsonify({'error': 'MIDI generation is not complete yet.'}), 400

            midi_file_path = job['result']['midiFilePath']
            if not os.path.exists(midi_file_path):
                return jsonify({'error': 'MIDI file not found.'}), 404

            return send_file(midi_file_path, as_attachment=True)

    def generate_midi(self, job_id, input_text):
        try:
            midi_data, meta_data = self.text2midi.text_to_midi(input_text, return_midi=True)
            self.job_store[job_id]['status'] = 'completed'
            self.job_store[job_id]['result'] = {
                'midiFilePath': meta_data['file_path'],  # Use the file path from metadata
                'metaData': meta_data
            }
        except Exception as e:
            self.job_store[job_id]['status'] = 'failed'
            self.job_store[job_id]['error'] = str(e)

    def run(self, debug=True):
        self.app.run(debug=debug)


if __name__ == '__main__':
    # Initialize the AppView with the config path
    app_view = AppView("storage/config/main_config.yaml")
    app_view.run()
