from flask import Flask, jsonify, request, url_for
import os
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Endpoint to get all files for a specific hopespot
@app.route('/api/hopespot/<hopespot_name>', methods=['GET'])
def get_hopespot_files(hopespot_name):
    base_path = os.path.join('static', 'hopespots', hopespot_name)
    
    if not os.path.exists(base_path):
        logging.error(f"Hopespot not found at path: {base_path}")
        return jsonify({'error': 'Hopespot not found'}), 404

    json_path = os.path.join(base_path, 'audio_data.json')
    if not os.path.isfile(json_path):
        logging.error(f"Audio data JSON not found at path: {json_path}")
        return jsonify({'error': 'Audio data not found'}), 404

    with open(json_path) as json_file:
        audio_data = json.load(json_file)

    # Construct response for all audio files
    response = []
    for audio_file, data in audio_data.items():
        audio_file_path = os.path.join(base_path, 'audio', audio_file)
        spectrogram_path = os.path.splitext(audio_file_path)[0] + '_spectrogram.jpg'
        file_info = {
            'audio': url_for('static', filename=f'hopespots/{hopespot_name}/audio/{audio_file}', _external=True),
            'spectrogram': url_for('static', filename=f'hopespots/{hopespot_name}/audio/{os.path.basename(spectrogram_path)}', _external=True),
            'points_of_interest': data['timestamps'],
            'votes': data['votes']
        }
        response.append(file_info)

    return jsonify(response)

# Endpoint to get the most popular audio file for a specific hopespot based on votes
@app.route('/api/hopespot/<hopespot_name>/popular', methods=['GET'])
def get_popular_file(hopespot_name):
    base_path = os.path.join('static', 'hopespots', hopespot_name)
    
    if not os.path.exists(base_path):
        return jsonify({'error': 'Hopespot not found'}), 404

    json_path = os.path.join(base_path, 'audio_data.json')
    if not os.path.isfile(json_path):
        return jsonify({'error': 'Audio data not found'}), 404

    with open(json_path) as json_file:
        audio_data = json.load(json_file)

    # Determine the most popular audio clip based on votes
    most_popular_audio = max(audio_data.items(), key=lambda x: x[1]['votes'])[0]
    audio_info = audio_data[most_popular_audio]
    audio_file_path = os.path.join(base_path, 'audio', most_popular_audio)
    spectrogram_path = os.path.splitext(audio_file_path)[0] + '_spectrogram.jpg'

    # Construct response for the most popular audio file
    response = {
        'audio': url_for('static', filename=f'hopespots/{hopespot_name}/audio/{most_popular_audio}', _external=True),
        'spectrogram': url_for('static', filename=f'hopespots/{hopespot_name}/audio/{os.path.basename(spectrogram_path)}', _external=True),
        'points_of_interest': audio_info['timestamps'],
        'votes': audio_info['votes']
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
