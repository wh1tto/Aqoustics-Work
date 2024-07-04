from flask import Flask, jsonify, request, url_for
import os
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/api/<hopespot>', methods=['GET'])
def serve_popular_audio(hopespot):
    base_path = os.path.join('static', 'hopespots', hopespot)
    logging.debug(f"Accessing base path: {base_path}")
    
    if not os.path.exists(base_path):
        logging.error(f"Hopespot not found at path: {base_path}")
        return jsonify({'error': 'Hopespot not found'}), 404

    json_path = os.path.join(base_path, 'audio_data.json')
    logging.debug(f"Looking for JSON data at: {json_path}")
    
    if not os.path.isfile(json_path):
        logging.error(f"Audio data JSON not found at path: {json_path}")
        return jsonify({'error': 'Audio data not found'}), 404

    with open(json_path) as json_file:
        audio_data = json.load(json_file)

    most_popular_audio = max(audio_data.items(), key=lambda x: x[1].get('votes', 0))[0]
    audio_info = audio_data[most_popular_audio]

    audio_file_path = os.path.join(base_path, 'audio', most_popular_audio)
    spectrogram_path = os.path.splitext(audio_file_path)[0] + '_spectrogram.jpg'

    # Using url_for to generate URLs correctly
    response = {
        'audio': url_for('static', filename=f'hopespots/{hopespot}/audio/{most_popular_audio}', _external=True),
        'spectrogram': url_for('static', filename=f'hopespots/{hopespot}/audio/{os.path.basename(spectrogram_path)}', _external=True),
        'points_of_interest': audio_info['timestamps'],
        'votes': audio_info['votes']
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
