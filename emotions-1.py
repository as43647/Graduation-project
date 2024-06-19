#!/usr/bin/env python
# coding: utf-8

# In[1]:


import threading
import multiprocessing as mp
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import speech_recognition as sr
import pyaudio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = 'display'

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

#########music setting
import spotipy
import spotipy.util as util

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random
import os

client_id = '6112e58c17e24f43b807dfad4eb50343'
client_secret = 'f3a619c16cbe443bb17532185e1f664b'
username = 'as43647'
redirect_uri = 'https://developer.spotify.com/dashboard/applications/6112e58c17e24f43b807dfad4eb50343'
AUTH_SCOPE = 'user-library-read user-read-currently-playing user-modify-playback-state user-read-playback-state streaming'

token = util.prompt_for_user_token(username, AUTH_SCOPE, client_id, client_secret, redirect_uri)
sp = spotipy.Spotify(auth=token)


def isplaying():
    return sp.current_playback()["is_playing"]
    # print(isplaying())


def get_track_ids(playlist_id):
    music_id_list = []
    playlist = sp.playlist(playlist_id)

    for item in playlist['tracks']['items']:
        music_track = item['track']
        music_id_list.append(music_track['id'])

        return music_id_list


def get_track_data(track_id):
    meta = sp.track(track_id)

    track_details = {'name': meta['name'], 'album': meta['album']['name'],
                     'artist': meta['album']['artists'][0]['name'], 'release_date': meta['album']['release_date'],
                     'duration_in_mins': round((meta['duration_ms'] * 0.001) / 60, 2)}
    return track_details


devices = sp.devices()
#print(json.dumps(devices, sort_keys=True, indent=4))
deviceID = devices['devices'][0]['id']
#print(deviceID)
# 這個值是影像辨識輸入的值


# In[2]:


# emotions will be displayed on your face from the webcam feed
if mode == "display":
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# In[4]:


def magic(artist, ids, emos, num):
    ja = []
    nums = int(num)
    for i in range(nums):
        recs = sp.recommendations(seed_artists=[artist], seed_tracks=[ids],limit = nums)["tracks"][i]['external_urls']['spotify']
        ja.append(recs)
        #print(recs)
    #print(ja)
    new_playlist = sp.user_playlist_create(username, name = emos)
    sp.user_playlist_add_tracks(username, new_playlist['id'], ja)
    return [emos, new_playlist['id']]


# In[5]:


def interr(emos):
    print("Looks like you're missing a " + emos + " playlist...")
    #print(emotion)
    num = input("How many songs do you want: ")
    track = input("Please tell me what song you like to listen to when you're " + emos.lower() + ": ")
    #print(track)
    artist = input("Please tell me the artist of this song: ")
    print("Ok, now we got " + track + " from " + artist + " when you feel " + emos + ".")
    print("Generating a playlist, please wait...")
    track_infoo = sp.search(track, 1, 0, "track")
    track_urio = track_infoo['tracks']['items'][0]['external_urls']['spotify']
    track_id = track_infoo['tracks']['items'][0]['id']
    track_artist = track_infoo['tracks']['items'][0]['artists'][0]['id']
    #sp.start_playback(device_id = deviceID, uris = [track_urio])
    #print(track_artist)   #歌手id
    #print(track_id)
    return magic(track_artist, track_id, emos, num)


# In[7]:


happyar = []
sadar = []
angryar = []
neutralar = []
fearfular = []
surprisedar = []
disgustedar = []

playlists = input("How many playlists do you want to search? [plz inter 1~50]: ")
num = int(playlists)
for i in range(num):
    if 'happy' in sp.user_playlists('as43647')['items'][i]['name'].lower():
        happy = sp.user_playlists('as43647')['items'][i]['name']
        happyid = sp.user_playlists('as43647')['items'][i]['id']
        happyar.append(happyid)  
    elif 'angry' in sp.user_playlists('as43647')['items'][i]['name'].lower():
        angry = sp.user_playlists('as43647')['items'][i]['name']
        angryid = sp.user_playlists('as43647')['items'][i]['id']
        angryar.append(angryid)      
    elif 'sad' in sp.user_playlists('as43647')['items'][i]['name'].lower():
        sad = sp.user_playlists('as43647')['items'][i]['name']
        sadid = sp.user_playlists('as43647')['items'][i]['id']
        sadar.append(sadid)
    elif 'neutral' in sp.user_playlists('as43647')['items'][i]['name'].lower():
        neutral = sp.user_playlists('as43647')['items'][i]['name']
        neutralid = sp.user_playlists('as43647')['items'][i]['id']
        neutralar.append(neutralid)
    elif 'fearful' in sp.user_playlists('as43647')['items'][i]['name'].lower():
        fearful = sp.user_playlists('as43647')['items'][i]['name']
        fearfulid = sp.user_playlists('as43647')['items'][i]['id']
        fearfular.append(fearfulid)        
    elif 'surprised' in sp.user_playlists('as43647')['items'][i]['name'].lower():
        surprised = sp.user_playlists('as43647')['items'][i]['name']
        surprisedid = sp.user_playlists('as43647')['items'][i]['id']
        surprisedar.append(surprisedid)        
    elif 'disgusted' in sp.user_playlists('as43647')['items'][i]['name'].lower():
        disgusted = sp.user_playlists('as43647')['items'][i]['name']
        disgustedid = sp.user_playlists('as43647')['items'][i]['id']
        disgustedar.append(disgustedid)
        
if not angryar:   #空矩陣
    a = interr('angry')
    angry = a[0]
    angryid = a[1]
if not happyar:
    b = interr('happy')
    happy = b[0]
    happyid = b[1]
if not sadar:
    b = interr('sad')
    sad = b[0]
    sadid = b[1]
if not neutralar:
    b = interr('neutral')
    neutral = b[0]
    neutralid = b[1]
if not fearfular:
    b = interr('fearful')
    fearful = b[0]
    fearfulid = b[1]
if not surprisedar:
    b = interr('surprised')
    surprised = b[0]
    surprisedid = b[1]
if not disgustedar:
    b = interr('disgusted')
    disgusted = b[0]
    disgustedid = b[1]

print("Now we have all the mood playlists!")


# In[8]:


def IDdataN(emos):
    emo = emos.lower()
    if emo == 'happy':
        haid = happyid
        return haid
    elif emo == 'sad':
        said = sadid
        return said
    elif emo == 'fearful':
        feid = fearfulid
        return feid
    elif emo == 'angry':
        anid = angryid
        return anid
    elif emo == 'disgusted':
        diid = disgustedid
        return diid
    elif emo == 'neutral':
        neid = neutralid
        return neid
    elif emo == 'surprised':
        suid = surprisedid
        return suid


# In[9]:


def random(emo):   
    import random
    playlist_uri =  "spotify:playlist:" + IDdataN(emo)
    track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_uri)["items"]]
    r_uris = random.sample(track_uris, len(track_uris))
    sp.start_playback(device_id = deviceID, uris = r_uris)


# In[11]:


def playing(emos):   
    if sp.current_playback()["is_playing"]:
        if int(sp.current_playback()["progress_ms"]/1000) > (int(sp.current_user_playing_track()['item']["duration_ms"]/1000)-3):
            random(emos)
    else:
        if int(sp.current_playback()["progress_ms"]/1000) == 0:
            random(emos)


# In[ ]:


j = []
cap = cv2.VideoCapture(0)
predata = '0'
while True:
    with open('writeSomething.txt', 'r') as f:
        data = f.read()
        if data == '1' and predata != data:
            sp.pause_playback()
            predata = data
        if data == '2' and predata != data:
            sp.start_playback()
            predata = data
        if data == '3' and predata != data:
            sp.next_track(deviceID)
            predata = data
        if data == '4' and predata != data:
            sp.previous_track(deviceID)
            predata = data
        #print(data)
        f.close()
        #print('------------------')
    ret, frame = cap.read()
    #if not ret:
        #break
    start = time.time()
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        j.append(maxindex)
        
    if len(j) == 5:
        k = max(j, key=j.count)
        emo = emotion_dict[k]
        playing(emo)
        j.clear()
    end = time.time()
    seconds = end - start
    fps = 1 / seconds
    cv2.putText(frame, "FPS: %.1f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
    cv2.imshow('Video', cv2.resize(frame, (800, 480), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




