import json
import os

import pickle
import pandas as pd
import numpy as np
from keras.models import load_model
import math
import cv2

from flask import current_app
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

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

face_CNN = load_model('app/face_cnn_model.h5')

song_model = load_model('app/best_model.h5')

def suggest_playlist_from_mood(all_tracks_with_features, mood):
    # Make song data of user fit our model
    data = pd.DataFrame.from_dict(all_tracks_with_features)
    new_labels = {'tempo': 'bpm', 'danceability': 'dnce', 'energy': 'nrgy', 'loudness': 'dB', 'liveliness': 'live',
                  'valence': 'val', 'duration_ms': 'dur', 'acousticness': 'acous'}
    data = data.rename(columns=new_labels)
    
    # current_app.logger.warn(data['artists'].values[0][0]['name'])
    # current_app.logger.warn(data.columns)

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



def suggest_playlist_from_mood_network(all_tracks_with_features, mood):
    # Make song data of user fit our model
    data = pd.DataFrame.from_dict(all_tracks_with_features)
    new_labels = {'tempo': 'bpm', 'danceability': 'dnce', 'energy': 'nrgy', 'loudness': 'dB', 'liveliness': 'live',
                  'valence': 'val', 'duration_ms': 'dur', 'acousticness': 'acous'}
    data = data.rename(columns=new_labels)
    
    # current_app.logger.warn(data['name'].values)

    copy_data = data.copy()

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

    def get_artist(dictionary):
        return dictionary[0]['name']

    def prep_title(tbl):
        le = LabelEncoder()
        tbl_copy = tbl
        tbl_copy.loc[:, 'name'] = le.fit_transform(tbl_copy.loc[:, 'name'])
        return tbl_copy

    def prep_artist(tbl):
        le = LabelEncoder()
        tbl_copy = tbl
        tbl_copy.loc[:, 'artist'] = le.fit_transform(tbl_copy.loc[:, 'artist'])
        return tbl_copy

    # def get_features(tbl):
    #     tbl_copy = tbl
    #     scaler = MinMaxScaler()
    #     tbl_copy = scaler.fit_transform(tbl_copy)
    #     return tbl_copy

    data['artist'] = data['artists'].apply(get_artist)
    data = prep_data(data)
    
    data = prep_features(data)
    data = prep_title(data)
    data = prep_artist(data)

    data = data.loc[:, ['name', 'artist', 'val','dB','bpm', 'nrgy', 'dnce']]
    # data = get_features(data)

    def predict_songs(tbl, copy):
        tbl_predicted = tbl
        predicted = song_model.predict(tbl)
        predicted = np.argmax(predicted, axis=1)
        predicted = face_to_moodlyr(predicted)
        current_app.logger.warn(predicted)
        copy['mood_predicted'] = predicted
        return copy

    predicted = predict_songs(data, copy_data)

    def get_score(scores, prev_ind):
        if prev_ind != -1:
            max_so_far = -math.inf
        else:
            max_so_far = scores[prev_ind]
        max_index = 0
        for ind in range(len(scores)):
            if scores[ind] > max_so_far:
                max_index = ind
        return max_index
        

    def find_predicted_songs(tbl, score, num_songs):
        songs = num_songs
        
        if songs > 25:
            songs = 25
        in_range = tbl
        current_app.logger.warn("All Scores")
        current_app.logger.warn(score)
        score_now = get_score(score, -1)
        previous_ind = score_now
        score_now = face_to_moodlyr([score_now])[0]
        
        in_range = in_range[in_range['mood_predicted'] == score_now]

        # in_range['dists'] = abs(in_range['mood_predicted'] - score)
        # sort_by_dist = in_range.sort_values('dists')

        current_app.logger.warn("First Score " + str(score_now))

        length_of_score = len(score)
        counter = 1


        while(len(in_range) == 0 and length_of_score != counter):
            score_now = get_score(score, previous_ind)
            score_now = face_to_moodlyr([score_now])[0]
            current_app.logger.warn("Score Now " + str(score_now))
            in_range = in_range[in_range['mood_predicted'] == score_now]
            counter += 1

        current_app.logger.warn(score_now)

        if (len(in_range) < num_songs):
            return in_range[:len(in_range)]
        else:
            return in_range[:num_songs]

    return find_predicted_songs(predicted, mood, 25).to_dict(orient='records')


def suggest_playlist_from_text(all_tracks_with_features, text):
    current_app.logger.warn(text)

    text = stemSentence(text, porter, word_token)

    feat = word_vec.transform([text])

    predicted_score = ridge.predict(feat)

    current_app.logger.warn(predicted_score)

    return suggest_playlist_from_mood(all_tracks_with_features, predicted_score)


def suggest_playlist_from_image(all_tracks_with_features, image):
    image.save('temp.jpg')
    image = img_to_matrix('temp.jpg')

    predicted_score = face_CNN.predict(image.reshape(1, 48, 48, 1))

    # current_app.logger.warn(predicted_score.reshape(7))

    index_of = np.argmax(predicted_score)

    mood = face_to_moodlyr([index_of])

    current_app.logger.warn(mood)

    return suggest_playlist_from_mood_network(all_tracks_with_features, predicted_score[0])

def face_to_moodlyr(y_pred):
    new_pred = []
    for pred in y_pred:
        if pred == 3:
            new_pred.append(2)
        elif pred == 0 or pred == 1:
            new_pred.append(1)
        elif pred == 2 or pred == 4:
            new_pred.append(3)
        else:
            new_pred.append(0)
    return np.array(new_pred)


def img_to_matrix(imagePath):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (48,48))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def stemSentence(sentence, porter, word_tokenize):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

