import argparse
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import librosa
import librosa.core
import librosa.display
import numpy as np

def load_files_list(path, BASE_PATH):
    results = []
    print("Loading filenames from: " + BASE_PATH + "/" + path)
    with open(BASE_PATH + "/" + path) as inputfile:
        for line in inputfile:
            results.append(line.strip().split())
    return results

def load_file(path, BASE_PATH):
    return 

def transform_file(path):
    pass

def process_wav(path, BASE_PATH):

    data, sample_rate = librosa.load(BASE_PATH + "/" + path)
    S = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_fft=1024, n_mels=128)
    log_S = librosa.amplitude_to_db(S)
    fig = plt.figure(figsize=(3,3), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
    output_file = BASE_PATH + '/png/' + path.split('.wav')[0]    
    plt.savefig('%s.png' % output_file)
    plt.close()

    return path.split('.wav')[0],  path.split('/')[0]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to dataset folder")
    parser.add_argument("-f", "--filename", help="name of file")
    parser.add_argument("-r", "--rate", help="rate of wav files", default=44100)
    args = parser.parse_args()
    
    files = load_files_list(args.filename, args.path)
    d = {'path': [], 'label': []}
    
    for file in files:
        path, label = process_wav(file[0], args.path)
        print(label + ' ' + path)
        d['path'].append(path+'.png')
        d['label'].append(label)

    df = pd.DataFrame(data=d)
    df.to_csv('/home/adam/data-asr/png/validation.csv')

