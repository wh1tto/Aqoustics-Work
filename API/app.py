from flask import Flask, jsonify, request, url_for
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

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
    
    # Check if spectogram exists, if not create it
    print("!!!!!!", os.path.splitext(os.path.basename(audio_file_path))[0])
    if not os.path.exists(os.path.splitext(audio_file_path)[0] + '_spectrogram.png'):
        logging.debug(f"Spectrogram not found for {audio_file_path}, creating it now")
        plotstft(audio_file_path, os.path.splitext(os.path.basename(audio_file_path))[0]+ '_spectrogram')
    else:
        logging.debug(f"Spectrogram found for {audio_file_path}")
        
    spectrogram_path = os.path.splitext(audio_file_path)[0] + '_spectrogram.png'

    # Using url_for to generate URLs correctly
    response = {
        'audio': url_for('static', filename=f'hopespots/{hopespot}/audio/{most_popular_audio}', _external=True),
        'spectrogram': url_for('static', filename=f'hopespots/{hopespot}/audio/{os.path.basename(spectrogram_path)}', _external=True),
        'points_of_interest': audio_info['timestamps'],
        'votes': audio_info['votes']
    }

    return jsonify(response)

# TODO: Implement the following functions
def splitAudio(audio_file_path, timestamps):
    pass

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, nameOfFile, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        # plt.show()
        # Savefig to the correct hopespot folder
        plt.savefig(f'static/hopespots/{nameOfFile}.png')

    plt.clf()

    return ims

if __name__ == '__main__':
    app.run(debug=True)
