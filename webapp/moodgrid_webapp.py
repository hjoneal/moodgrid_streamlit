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

sleep_time=0.075

# update default encoding so streamlit accepts unconventional characters
sys.stdout.reconfigure(encoding='utf-8')

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
PLAYLIST_LINK = st.text_input("Enter the Spotify URL of a playlist you want to subset (Spotify -> Share -> Copy link to playlist): ")

if PLAYLIST_LINK:

    #     playlist_submit = st.form_submit_button("Submit")

    # if playlist_submit:


    # authenticate
    client_credentials_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET
    )

    # create spotify session object
    session = spotipy.Spotify(client_credentials_manager=client_credentials_manager, status_retries=0)

    # create csv file
    with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # write header column names
        writer.writerow(["track_id", "track_name", "artists", "popularity", 'danceability', 'energy', 'key',\
                    'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',\
                    'valence', 'tempo', 'duration_ms', 'time_signature', 'explicit'])


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



                                    # write to csv
                        writer.writerow([track_id, track_name, artists, popularity, danceability, energy, \
                                        key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, \
                                        valence, tempo, duration_ms, time_signature, explicit])

                        track_counter +=1
                        #print(f"Successfully read '{playlist_input_name}' - Track {track_counter}/{pl_length}: {track_name} by {artists}")
                        status_container.text(f"Loading playlist: '{playlist_input_name}' ---> {round(((track_counter/pl_length)*100),1)}% complete")
                        printing_container.text(f"Track {track_counter}/{pl_length}: {track_name} by {artists}")

            

                    else:
                        print(f"No features found for the playlist track IDs.")

                else:
                    print(f"Track not found")

        else:
            print(f"Playlist not found")

        print("Playlist read complete")


    # read in csv file
    df = pd.read_csv("../data/testing/22mar_1pl_test.csv")

    # drop nulls
    df.dropna(inplace = True)

    # drop songs that are duplicated within the same playlist mood, we can keep duplicates that are in different mood categories
    df.drop_duplicates(inplace =True)


    # # genre is currently a long string of different genres
    # # get genre as a series
    # genre = df['artist_genre']

    # # split into a list using the comma
    # genre_split = genre.str.split(", ")

    # #strip out unwanted characters
    # new_genre_list = []

    # for artist in genre_split:
    #     strippedgenre = []
    #     for genre in artist:
    #         strippedgenre.append((re.sub(r'[^\w]', ' ', genre)).strip())
    #     new_genre_list.append(strippedgenre)


    # ### count up the instances of each specific genre ###

    # genre_counts = {}  # dictionary to keep track of genre counters

    # for genres in new_genre_list:
    #     # create a Counter for the current list of genres
    #     current_genre_count = Counter(genres)

    #     for genre, count in current_genre_count.items():
    #         # update the corresponding counter for the genre in the dictionary
    #         if genre in genre_counts:
    #             genre_counts[genre] += count
    #         else:
    #             genre_counts[genre] = count
                

    # # create a dictionary to store the genre counts, categorised by certain keywords.
    # # the list of keywords can be added to iteratively as more category options are found that end up in 'other'

    # rocklist = ['rock', 'metal', 'grunge', 'alt z', 'punk', 'indie', 'country', 'folk', 'guitar', 'emo']
    # poplist = ['pop', 'songwriter', 'stomp and holler', 'neo mellow']
    # electroniclist = ['edm', 'house', 'dance', 'trance', 'tech', 'bass', 'electro', 'rave', 'tronica', 'new french touch', 'beats', 'neo mellow', 'psych']
    # soullist = ['soul', 'funk', 'r b', 'motown', 'jazz', 'quiet storm', 'reggae', 'disco', 'r&b']
    # raplist = ['rap', 'hop', 'trap', 'drill']

    # #instantiate new list and genre dictionary with counts
    # grouped_genre_list = []
    # for artist in new_genre_list:
    #     genre_counts = {
    #     'rock': 0,
    #     'pop': 0,
    #     'electronic': 0,
    #     'soul': 0,
    #     'rap': 0,
    #     'other': 0
    #     }

    # # count the genres for each artist and update the dictionary with a counter
    #     for genre in artist:

    #         if any(keyword in genre for keyword in rocklist):
    #             genre_counts['rock'] += 1
    #         elif any(keyword in genre for keyword in raplist):
    #             genre_counts['rap'] += 1
    #         elif any(keyword in genre for keyword in electroniclist):
    #             genre_counts['electronic'] += 1
    #         elif any(keyword in genre for keyword in soullist):
    #             genre_counts['soul'] += 1
    #         elif any(keyword in genre for keyword in poplist):
    #             genre_counts['pop'] += 1
    #         else:
    #             genre_counts['other'] += 0.1  # give 'unknown' a lower weighting

    #     # sort the dictionary by values in descending order
    #     sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)


    #     # top genre is the first key in the sorted dict 
    #     max_genre = sorted_genres[0][0]

    #     # add genre to the list for that artist
    #     grouped_genre_list.append(max_genre)

    # # add as new df column and drop the original
    # df['grouped_genres'] = grouped_genre_list
    # df.drop(columns='artist_genre', inplace=True)


    # map to common or not common time and drop the original
    df["common_time"] = df["time_signature"].map(
        {
            4: 1,
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            5: 0,
            6: 0, 
            7: 0
            
        }
    )
    df.drop(columns='time_signature', inplace=True)

    # # convert playlist mood into dummies
    # mood_dummies = pd.get_dummies(df['playlist_mood'], prefix="mood")
    # df = pd.concat([df, mood_dummies], axis=1)

    # # map to playlist_mood to numerical for modelling purposes and drop the original
    # df["mood_num"] = df["playlist_mood"].map(
    #     {
    #         'Chill': 0,
    #         'Energetic': 1,
    #         'Sad': 2,
    #         'Happy': 3
            
    #     }
    # )
    # df.drop(columns='playlist_mood', inplace=True)

    # convert explicit column to binary
    df["explicit"] = df["explicit"].astype(int)


    # convert key column to actual key so it is more interpretable before creating dummies
    df["key_sig"] = df["key"].map(
        {
            0: 'C',
            1: 'Db',
            2: 'D',
            3: 'Eb',
            4: 'E',
            5: 'F', 
            6: 'Gb',
            7: 'G',
            8: 'Ab',
            9: 'A',
            10: 'Bb',
            11: 'B'
            
        
        }
    )
    # convert key into dummies and drop originals
    key_dummies = pd.get_dummies(df['key_sig'], prefix="key")
    df = pd.concat([df, key_dummies], axis=1)

    df.drop(columns=['key', 'key_sig'], inplace=True)



    # # convert genres into dummies and drop original
    # genre_dummies = pd.get_dummies(df['grouped_genres'], prefix="genre")
    # df = pd.concat([df, genre_dummies], axis=1)

    # df.drop(columns='grouped_genres', inplace=True)


    #df.to_csv("../data/preprocessed/19_03_2023_1pl_preprocessed_from_script.csv")

    EC_HS_features = joblib.load("../data/pkl/models/1744pl_common_features.pkl")

    for col in EC_HS_features:
        if col not in df:
            df[col] = 0

    # keep the common EC, HS features for the modelling
    df_sel = df[EC_HS_features]

    # load the scalers and models
    EC_scaler = joblib.load("../data/pkl/models/1744pl_EC_scaler.pkl")
    EC_logreg_model = joblib.load("../data/pkl/models/1744pl_EC_logreg_model.pkl")
    HS_scaler = joblib.load("../data/pkl/models/1744pl_HS_scaler.pkl")
    HS_logreg_model = joblib.load("../data/pkl/models/1744pl_HS_logreg_model.pkl")

    # scale the data
    EC_scaled = EC_scaler.transform(df_sel)
    HS_scaled = HS_scaler.transform(df_sel)


    #get the energy score by predicting probabilities and taking the second column
    X_energy_score = (EC_logreg_model.predict_proba(EC_scaled))[:, 1]
    #get the energy score for the HS dataset by applying the same model predictions
    X_happy_score = (HS_logreg_model.predict_proba(HS_scaled))[:, 1]


    # add scores to respective df
    df['energy_score'] = X_energy_score
    df['happy_score'] = X_happy_score

    # scale happy and energy scores away from being skewed to v high or v low values
    df = df.assign(happy_coeff = lambda x: np.log(x['happy_score'] / (1 - x['happy_score'])))
    df = df.assign(energy_coeff = lambda x: np.log(x['energy_score'] / (1 - x['energy_score'])))


    # drop rows with duplicate track ids from different playlist moods as we no longer want them
    df.drop(df[df["track_id"].duplicated()].index, inplace=True)
    X_coeffs = df[["energy_coeff", "happy_coeff"]]
    X_array = np.array(X_coeffs)

    # get min and max energy and happy score
    energy_min = round(df["energy_coeff"].min(), 1)
    energy_max = round(df["energy_coeff"].max(), 1)
    happy_min = round(df["happy_coeff"].min(), 1)
    happy_max = round(df["happy_coeff"].max(), 1)

    # update min and max to be at least 1 for plotting purposes, add 0.1 so that extreme values aren't on the very edge
    if energy_min > -1:
         plotting_energy_min = -1
    else:
        plotting_energy_min = energy_min-0.1

    if energy_max < 1:
         plotting_energy_max = 1
    else:
        plotting_energy_max = energy_max+0.1

    if happy_min > -1:
         plotting_happy_min = -1
    else:
        plotting_happy_min = happy_min-0.1

    if happy_max < 1:
         plotting_happy_max = 1
    else:
        plotting_happy_max = happy_max+0.1


    #update min and maxes for plotting
    left_quad = plotting_energy_min/2
    right_quad = plotting_energy_max/2
    top_quad = plotting_happy_max/2
    bottom_quad = plotting_happy_min/2


    # create a new column and set it to 0 for all rows
    df['selected'] = 0
    # map selected numerical to a string for plotting
    df["selected_str"] = df["selected"].map(
        {
            0: 'No',
            1: 'Yes',
            
        }
    )
    fig1 = px.scatter(df, x="energy_coeff", y="happy_coeff", hover_data=["artists", "track_name"], height=600,width=900,opacity=0.7,
                    color=df["selected_str"].map({"No": playlist_input_name}),
                    labels={"happy_coeff": "Happy Score",
                            "energy_coeff": "Energy Score",
                            "artists": "Artist",
                            "track_name": "Track",
                            "color": "Playlist"})
    fig1.add_vline(x=0, line_width=2, line_color="black")
    fig1.add_hline(y=0, line_width=2, line_color="black")
    # Add text annotations
    fig1.update_layout(annotations=[
            dict(x=left_quad,y=top_quad,xref="x",yref="y",text="Happy & Chilled",showarrow=False,font=dict(size=15)),
            dict(x=right_quad,y=top_quad,xref="x",yref="y",text="Happy & Energetic",showarrow=False,font=dict(size=15)),
            dict(x=left_quad,y=bottom_quad,xref="x",yref="y",text="Sad & Chilled",showarrow=False,font=dict(size=15)),
            dict(x=right_quad,y=bottom_quad,xref="x",yref="y",text="Sad & Energetic",showarrow=False,font=dict(size=15))])
    fig1.update_yaxes(
    range=(plotting_happy_min, plotting_happy_max),
    constrain='domain')
    fig1.update_xaxes(
    range=(plotting_energy_min, plotting_energy_max),
    constrain='domain')
    #fig1.update_layout(title={'font': {'size': 15}})
    #fig1.show()

    st.plotly_chart(fig1, theme=None, use_container_width=True)

    # create a default value for number of tracks to subset
    kneigh_default = int(round((track_counter/4),0))

    # get user inputs for number of songs and happy/energy score
    with st.form("user_form"):
        # def update_slider():
        #     st.session_state.slider = st.session_state.numeric
        # def update_numin():
        #     st.session_state.numeric = st.session_state.slider
        #find_happy = st.number_input(label=(f"The input playlist has a happy score ranging from {happy_min} to {happy_max}, how happy do you want your new playlist to be?: "), value=0.0, step=1.)
        find_happy = st.slider(label=(f"The input playlist has a happy score ranging from {happy_min} to {happy_max}, how happy do you want your new playlist to be?: "), value=0.0, step=0.1, min_value=happy_min, max_value=happy_max)
        find_energy = st.slider(label=(f"The input playlist has an energy score ranging from {energy_min} to {energy_max}, how energetic do you want your new playlist to be?: "), value=0.0, step=0.1, min_value=energy_min, max_value=energy_max)
        #find_energy = st.number_input(label=(f"The input playlist has an energy score ranging from {energy_min} to {energy_max}, how energetic do you want your new playlist to be?: "), value=0.0, step=1.)
        new_playlist_name = st.text_input("Give your new playlist a name: ")
        # val = st.number_input("How many tracks out of the original {track_counter} do you want in your playlist?: ",
        #                       min_value=1,
        #                       max_value=track_counter,
        #                       value=kneigh_default,
        #                       step=1, key = 'numeric',
        #                       on_change = update_slider)
        # kneigh = st.slider(label=(f"How many tracks out of the original {track_counter} do you want in your playlist?: "),
        #                    min_value=1,
        #                    max_value=track_counter,
        #                    value=val,
        #                    step=1, key='slider', on_change=update_numin)
        #user = st.text_input("Spotify username:")
        kneigh = st.number_input(label=(f"How many tracks out of the original {track_counter} do you want in your playlist?: "), min_value=1, max_value=track_counter, value=5, step=1)

        user_submit = st.form_submit_button("Submit")

    if user_submit:
        st.write("Submitted")
        
        # find_happy = float(input(f"The input playlist has a happy score ranging from {happy_min} to {happy_max}, how happy do you want your new playlist to be?: "))
        # find_energy = float(input(f"The input playlist has an energy score ranging from {energy_min} to {energy_max}, how energetic do you want your new playlist to be?: "))
        # new_playlist_name = input("Give your new playlist a name: ")
        # kneigh = int(input(f"How many tracks out of the original {track_counter} do you want in your playlist?: "))

        # lambda function to create new column - euclidean distance away from selected happy, energy score
        X_coeffs = X_coeffs.assign(dist = lambda x: np.sqrt(((find_energy - x["energy_coeff"])**2) + ((find_happy - x["happy_coeff"])**2)))
        # find indices of top "Kneigh" results
        Kneighbours_index = list(X_coeffs.sort_values(by="dist", ascending=True).head(kneigh).index)


        # set the new column to 1 at the specified indexes
        for i in Kneighbours_index:
            df.loc[i, 'selected'] = 1

        # initialise the track, artist, track id lists
        track_list = []
        artist_list = []
        track_id_list = []

        # populate the lists with the nearest neighbours
        for i in Kneighbours_index:
            track_name = df.iloc[i]['track_name']
            track_list.append(track_name)
            artist_name = df.iloc[i]['artists']
            artist_list.append(artist_name)
            track_id = df.iloc[i]['track_id']
            track_id_list.append(track_id)

        # # print the list of songs
        # print(f"The closest {kneigh} songs to your chosen energy score of {find_energy} and happy score of {find_happy} are...")
        # for track, artist in zip(track_list, artist_list):
        #     print(f"{track} by {artist}")


        # re-map selected numerical to a string for plotting
        df["selected_str"] = df["selected"].map(
            {
                0: 'No',
                1: 'Yes',
                
            }
        )


        # add new playlist
        fig2 = px.scatter(df, x="energy_coeff", y="happy_coeff", color=df["selected_str"].map({"No": playlist_input_name, "Yes": new_playlist_name}),
                        hover_data=["artists", "track_name"],height=600,width=900,opacity=0.7,
                        color_discrete_map={playlist_input_name: 'light blue',new_playlist_name: 'magenta'},
                        labels={"happy_coeff": "Happy Score",
                                "energy_coeff": "Energy Score",
                                "artists": "Artist",
                                "track_name": "Track",
                                "selected_str": "Track selected?",
                                "color": "Playlist"},
                        )
        fig2.add_vline(x=0, line_width=2, line_color="black")
        fig2.add_hline(y=0, line_width=2, line_color="black")

        # Add text annotations
        fig2.update_layout(annotations=[
                dict(x=left_quad,y=top_quad,xref="x",yref="y",text="Happy & Chilled",showarrow=False,font=dict(size=15)),
                dict(x=right_quad,y=top_quad,xref="x",yref="y",text="Happy & Energetic",showarrow=False,font=dict(size=15)),
                dict(x=left_quad,y=bottom_quad,xref="x",yref="y",text="Sad & Chilled",showarrow=False,font=dict(size=15)),
                dict(x=right_quad,y=bottom_quad,xref="x",yref="y",text="Sad & Energetic",showarrow=False,font=dict(size=15))])
        #fig2.update_layout(title={'font': {'size': 15}})
        fig2.update_yaxes(
        range=(plotting_happy_min, plotting_happy_max),
        constrain='domain')
        fig2.update_xaxes(
        range=(plotting_energy_min, plotting_energy_max),
        constrain='domain')
        #fig.show()

        st.plotly_chart(fig2, theme=None, use_container_width=True)


        # load credentials from .env file
        load_dotenv()

        CLIENT_ID = os.getenv("CLIENT_ID", "")
        CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")

        # authenticate
        client_credentials_manager = SpotifyClientCredentials(
            client_id=CLIENT_ID, client_secret=CLIENT_SECRET
        )

        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                                    client_secret=CLIENT_SECRET,
                                                    redirect_uri="http://localhost:8888/callback",
                                                    scope="playlist-modify-private"))

        
        
        user = sp.current_user()["id"]

        playlist_description = f"A playlist created using MoodGrid from {playlist_input_name} with a Happy rating of {find_happy} and an Energy rating of {find_energy}"

        playlist = sp.user_playlist_create(user, new_playlist_name, public=False, description=playlist_description)

        sp.playlist_add_items(playlist["id"], track_id_list)

        new_playlist_URL = playlist['external_urls']['spotify']
        st.write(f"Playlist {new_playlist_name} added to Spotify")
        st.write(f"Click the link the check it out! {new_playlist_URL}")


