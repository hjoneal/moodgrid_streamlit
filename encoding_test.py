### import libraries
import streamlit as st
import joblib

st.title("MoodGrid Playlist Subsetter")


import csv
import os
import re
import pandas as pd
import numpy as np
import re
from collections import Counter
import joblib
import time
import sys


import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# update default encoding so streamlit accepts unconventional characters
sys.stdout.reconfigure(encoding='utf-8')

sleep_time=0.1

# load credentials from .env file
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
OUTPUT_FILE_NAME = "../data/testing/22mar_1pl_test.csv"

print(CLIENT_ID)

# # read a csv containing a list of playlist URLs and their associated mood
# print("Welcome to the MoodGrid Playlist Subsetter")
# print("")



# B. Set up input field
# with st.form("playlist_form"):
PLAYLIST_LINK = 'https://open.spotify.com/playlist/3Q72icWYTFw9MTnEp0vGiw?si=3e7885ed80a64c98'
if PLAYLIST_LINK:

    #     playlist_submit = st.form_submit_button("Submit")

    # if playlist_submit:


    # authenticate
    client_credentials_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET
    )

    # create spotify session object
    session = spotipy.Spotify(client_credentials_manager=client_credentials_manager, status_retries=0)




    # get uri from https link
    if match := re.match(r"https://open.spotify.com/playlist/(.*)\?", PLAYLIST_LINK):
        playlist_uri = match.groups()[0]
    else:
        raise ValueError("Expected format: https://open.spotify.com/playlist/...")
    

    pname = session.user_playlist(user=None, playlist_id=playlist_uri, fields="name")
    time.sleep(sleep_time)
    playlist_input_name = pname["name"]

    results = session.playlist_tracks(playlist_uri)
    time.sleep(sleep_time)
    tracks = results["items"]
    while results['next']:
        results = session.next(results)
        time.sleep(sleep_time)
        tracks.extend(results['items'])

    pl_length = len(tracks)
    print(f"Playlist {playlist_input_name} contains {pl_length} tracks")

    # create contatiners for updating text in place
    status_container = st.empty()
    printing_container = st.empty()

    track_counter=0
    # extract name and artist
    if tracks is not None:
        for track in tracks:
            if track["track"] is not None:
                track_id = track["track"]["id"]
                track_name = track["track"]["name"]
                artists = ", ".join(
                    [artist["name"] for artist in track["track"]["artists"]]
                )
                popularity = track["track"]["popularity"]
                explicit = track["track"]["explicit"]

                # #section to grab genre of each artist from separate API call

                # artist_url = session.artist(track["track"]["artists"][0]["external_urls"]["spotify"])
                # time.sleep(sleep_time)
                # artist_genre = artist_url["genres"] 

                # get a dictionary for the audio features to add these to our csv output

                features = session.audio_features(track["track"]["id"])
                time.sleep(sleep_time)
                                
                if features is not None:
                    for feature in features:
                        if feature is not None:

                            danceability = feature.get("danceability", "NaN")
                            energy = feature.get("energy", "NaN")
                            key = feature.get("key", "NaN")
                            loudness = feature.get("loudness", "NaN")
                            mode = feature.get("mode", "NaN")
                            speechiness = feature.get("speechiness", "NaN")
                            acousticness = feature.get("acousticness", "NaN")
                            instrumentalness = feature.get("instrumentalness", "NaN")
                            liveness = feature.get("liveness", "NaN")
                            valence = feature.get("valence", "NaN")
                            tempo = feature.get("tempo", "NaN")
                            duration_ms = feature.get("duration_ms", "NaN")
                            time_signature = feature.get("time_signature", "NaN")





                    track_counter +=1
                    encoded_track_name = track_name.encode('utf-8')
                    print(f"Successfully read '{playlist_input_name}' - Track {track_counter}/{pl_length}: {track_name} by {artists}")
                    status_container.text(f"Loading playlist: '{playlist_input_name}' ---> {round(((track_counter/pl_length)*100),1)}% complete")
                    printing_container.text(f"Track {track_counter}/{pl_length}: {track_name} by {artists}")

        

                else:
                    print(f"No features found for the playlist track IDs.")

            else:
                print(f"Track not found")

    else:
        print(f"Playlist not found")

    print("Playlist read complete")