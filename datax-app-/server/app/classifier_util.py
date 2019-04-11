import json
import os

import pickle
import pandas as pd
import numpy as np

from flask import current_app

# from nltk.stem import PorterStemmer
# from nltk.stem import LancasterStemmer

with open('app/lm.pickle', 'rb') as fi:
    lm = pickle.load(fi)

with open('app/word_vec.pickle', 'rb') as fb:
    word_vec = pickle.load(fb)

with open('app/ridge_reg.pickle', 'rb') as fc:
    # This is for sentiment analysis!!!
    ridge = pickle.load(fc)

with open('app/porter.pickle', 'rb') as fc:
    porter = pickle.load(fc)

with open('app/word_token.pickle', 'rb') as fc:
    word_token = pickle.load(fc)




def suggest_playlist_from_mood(all_tracks_with_features, mood):
    # Make song data of user fit our model
    data = pd.DataFrame.from_dict(all_tracks_with_features)
    new_labels = {'tempo': 'bpm', 'danceability': 'dnce', 'energy': 'nrgy', 'loudness': 'dB', 'liveliness': 'live',
                  'valence': 'val', 'duration_ms': 'dur', 'acousticness': 'acous'}
    data = data.rename(columns=new_labels)

    def prep_data(frame):
        frame_data = frame
        # zero_bpm = frame_data[frame_data['bpm'] == 0].index[0]
        # frame_data = frame_data.drop([zero_bpm])
        frame_data['dur'] = frame_data['dur'].astype('float')
        return frame_data

    def normalize(col):
        col_range = max(col) - min(col)
        avg = np.mean(col)
        return (col - avg) / col_range

    def prep_features(tbl):
        tbl_norm = tbl
        tbl_norm['bpm'] = normalize(tbl_norm['bpm'])
        tbl_norm['nrgy'] = normalize(tbl_norm['nrgy'] * 100)
        tbl_norm['dnce'] = normalize(tbl_norm['dnce'] * 100)
        tbl_norm['val'] = normalize(tbl_norm['val'] * 100)
        tbl_norm['acous'] = normalize(tbl_norm['acous'] * 100)
        tbl_norm['dur'] = tbl_norm['dur'] / 100000
        return tbl_norm

    data = prep_data(data)
    data = prep_features(data)

    def predict_songs(tbl):
        tbl_predicted = tbl
        predicted = lm.predict(tbl.loc[:, ['bpm', 'nrgy', 'dnce', 'dB', 'val', 'dur', 'acous']])
        tbl_predicted['mood_predicted'] = predicted
        return tbl_predicted

    predicted = predict_songs(data)

    def find_predicted_songs(tbl, score, num_songs):
        songs = num_songs
        if songs > 25:
            songs = 25
        in_range = tbl

        in_range['dists'] = abs(in_range['mood_predicted'] - score)
        sort_by_dist = in_range.sort_values('dists')

        return sort_by_dist[:num_songs]

    return find_predicted_songs(predicted, mood, 25).to_dict(orient='records')


def suggest_playlist_from_text(all_tracks_with_features, text):
    current_app.logger.warn(text)

    text = stemSentence(text, porter, word_token)

    feat = word_vec.transform([text])

    predicted_score = ridge_reg.predict(feat)

    current_app.logger.warn(predicted_score)

    return suggest_playlist_from_mood(all_tracks_with_features, predicted_score)


def stemSentence(sentence, porter, word_tokenize):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

