### import libraries

import joblib

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
import random


import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import streamlit as st
from streamlit_extras.stateful_button import button
import timeit

import warnings
warnings.filterwarnings('ignore')

# load full preprocessed dataset so we can skip API calls
full_df = joblib.load("./pickles/compressed_song_dataset.pkl")

sleep_time=0.075
st.set_page_config(page_title="MoodGrid Playlist Subsetter", layout="wide")

st.title("Spotify MoodGrid Playlister")
st.text("A Harry Neal innovation")
st.subheader("Step 1: Add one or more Spotify playlists to see where the songs sit on the Happiness & Energy MoodGrid")
st.text("")
st.subheader("Step 2: Choose a Happy & Energy score and create a new playlist centred on your chosen mood")
st.text("")
st.subheader("Step 3: Listen to your new playlist!")
st.text("")

# update default encoding so streamlit accepts unconventional characters
sys.stdout.reconfigure(encoding='utf-8')

# # load credentials from .env file
# load_dotenv()

CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]

# CLIENT_ID = os.getenv("CLIENT_ID", "")
# CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")

PLAYLIST_LINK = -1
PLAYLIST_LINK2 = -1
PLAYLIST_LINK3 = -1
PLAYLIST_LINK4 = -1
PLAYLIST_LINK5 = -1

# add button to unveil the form once user has read instructions
if button("Get started", key="button1"):
    st.text("")

    # B. Set up input fields
    PLAYLIST_LINK = st.text_input("Enter the Spotify URL of a playlist you want to subset (Spotify Playlist -> o o o -> Share -> Copy link to playlist): ", key="PLAYLIST_LINK")
    PLAYLIST_LINK2 = st.text_input("Enter the URLs of any additional Spotify playlists you want to explore", key="PLAYLIST_LINK2")
    PLAYLIST_LINK3 = st.text_input("", key="PLAYLIST_LINK3")
    PLAYLIST_LINK4 = st.text_input("", key="PLAYLIST_LINK4")
    PLAYLIST_LINK5 = st.text_input("", key="PLAYLIST_LINK5")

    if button("Submit", key="button2"):

        # authenticate
        client_credentials_manager = SpotifyClientCredentials(
            client_id=CLIENT_ID, client_secret=CLIENT_SECRET
        )

        # create spotify session object
        session = spotipy.Spotify(client_credentials_manager=client_credentials_manager, status_retries=0)

        # create a list of playlists that are either a valid link or -1
        pl_list = [PLAYLIST_LINK,PLAYLIST_LINK2,PLAYLIST_LINK3,PLAYLIST_LINK4,PLAYLIST_LINK5]
        # for loop to add valid links to a new list
        pl_to_run = []
        for pl in pl_list:
            if len(pl) > 10:
                pl_to_run.append(pl)

        # create an empty list to store the data
        data = []
        # create an empty list to store the playlist names
        plname_list = []

        #create an empty dataframe to add pre-existing tracks
        trax_df = pd.DataFrame(columns=full_df.columns)

        # instantiate an index to count and label the distinct playlist links
        playlist_link_index = 0

        for playlist_link in pl_to_run:
            playlist_link_index += 1
            # get uri from https link
            if match := re.match(r"https://open.spotify.com/playlist/(.*)\?", playlist_link):
                playlist_uri = match.groups()[0]
            else:
                raise ValueError("Expected format: https://open.spotify.com/playlist/...")
            
            playlist_number = playlist_link_index
            pname = session.user_playlist(user=None, playlist_id=playlist_uri, fields="name")
            time.sleep(sleep_time)
            playlist_input_name = pname["name"]
            plname_list.append(playlist_input_name)

            results = session.playlist_tracks(playlist_uri)
            time.sleep(sleep_time)
            tracks = results["items"]
            while results['next']:
                results = session.next(results)
                time.sleep(sleep_time)
                tracks.extend(results['items'])

            pl_length = len(tracks)
            print(f"Playlist {playlist_input_name} contains {pl_length} tracks")

            # create containers for updating text in place
            status_container = st.empty()
            printing_container = st.empty()

            track_counter=0
            # extract name and artist
            if tracks is not None:
                for track in tracks:
                    if track["track"] is not None:
                        track_id = track["track"]["id"]
                        # check if track id already exists in the cleaned dataset
                        if (full_df['track_id'].eq(track_id)).any():
                            # select the row containing the track id, use iloc to only select the first in case of duplicates
                            track_series = full_df.loc[full_df['track_id'] == track_id].iloc[0]
                            # update playlist number and name to be new input instead of orig
                            track_series['playlist_number'] = playlist_number
                            track_series['playlist_name'] = playlist_input_name
                            #append pre-existing track info to df
                            trax_df = trax_df.append(track_series, ignore_index=True)
                            #update track and artist variables for progress printing
                            track_name = track_series['track_name']
                            artists = track_series['artists']

                        # if track not in original dataset, get features from spotify API
                        else:
                            track_name = track["track"]["name"]
                            artists = ", ".join(
                                [artist["name"] for artist in track["track"]["artists"]]
                            )
                            popularity = track["track"]["popularity"]
                            explicit = track["track"]["explicit"]

                            #section to grab genre of each artist from separate API call
                            try:
                                artist_url = session.artist(track["track"]["artists"][0]["external_urls"]["spotify"])
                                time.sleep(sleep_time*2)
                                artist_genre = artist_url["genres"]
                            except:
                                artist_genre = "" 

                            features = session.audio_features(track["track"]["id"])
                            time.sleep(sleep_time)
                                            
                            if features is not None:
                                feature_data = {}
                                feature_data['playlist_number'] = playlist_number
                                feature_data['playlist_name'] = playlist_input_name
                                feature_data['track_id'] = track_id
                                feature_data['track_name'] = track_name
                                feature_data['artists'] = artists
                                feature_data['artist_genre'] = artist_genre
                                feature_data['popularity'] = popularity
                                feature_data['explicit'] = explicit

                                for feature in features:
                                    if feature is not None:
                                        feature_data['danceability'] = feature.get("danceability", "NaN")
                                        feature_data['energy'] = feature.get("energy", "NaN")
                                        feature_data['key'] = feature.get("key", "NaN")
                                        feature_data['loudness'] = feature.get("loudness", "NaN")
                                        feature_data['mode'] = feature.get("mode", "NaN")
                                        feature_data['speechiness'] = feature.get("speechiness", "NaN")
                                        feature_data['acousticness'] = feature.get("acousticness", "NaN")
                                        feature_data['instrumentalness'] = feature.get("instrumentalness", "NaN")
                                        feature_data['liveness'] = feature.get("liveness", "NaN")
                                        feature_data['valence'] = feature.get("valence", "NaN")
                                        feature_data['tempo'] = feature.get("tempo", "NaN")
                                        feature_data['duration_ms'] = feature.get("duration_ms", "NaN")
                                        feature_data['time_signature'] = feature.get("time_signature", "NaN")

                                data.append(feature_data)

                            else:
                                print(f"No features found for the playlist track IDs.")

                        track_counter +=1
                        #print(f"Successfully read '{playlist_input_name}' - Track {track_counter}/{pl_length}: {track_name} by {artists}")
                        status_container.text(f"Loading playlist: '{playlist_input_name}' ---> {round(((track_counter/pl_length)*100),1)}% complete")
                        printing_container.text(f"Track {track_counter}/{pl_length}: {track_name} by {artists}")
                    
                    else:
                        print(f"Track not found")
                        track_counter +=1

            else:
                print(f"Playlist not found")

            print("Playlist read complete")

        # if there is data create a dataframe from it
        if len(data) > 0:
            # convert to a dataframe
            df = pd.DataFrame(data)
        # otherwise create an empty dataframe so the rest of the code runs
        else:
            api_cols = joblib.load("./pickles/api_cols.pkl")
            df = pd.DataFrame(columns=api_cols)

        df = df[
            df['artists'].notna()
        ]

        #drop nan tracks
        df = df[
            df['track_name'].notna()
        ]


        df = df.fillna("[]")

        df['artist_genre'] = df['artist_genre'].astype(str)

        # get genre as a series
        genre = df['artist_genre']

        # split into a list using the comma
        genre_split = genre.str.split(", ")

        #strip out unwanted characters
        new_genre_list = []

        for artist in genre_split:
            strippedgenre = []
            for genre in artist:
                strippedgenre.append((re.sub(r'[^\w]', ' ', genre)).strip())
            new_genre_list.append(strippedgenre)

        genre_counts = {}  # dictionary to keep track of genre counters

        for genres in new_genre_list:
            # create a Counter for the current list of genres
            current_genre_count = Counter(genres)

            for genre, count in current_genre_count.items():
                # update the corresponding counter for the genre in the dictionary
                if genre in genre_counts:
                    genre_counts[genre] += count
                else:
                    genre_counts[genre] = count

        sorted_by_genre = sorted(genre_counts.items(), key=lambda x:x[1], reverse=True)

        # the list of keywords was initially chosen with a smaller set, and was added to iteratively \
        # as more category options were found that end up in 'other'

        rocklist = ['rock','alt z', 'punk', 'grunge', 'indie', 'guitar', 'weirdcore']
        metallist = ['metal', 'emo', 'hardcore', 'screamo', 'death', 'slayer']
        folklist = ['country', 'folk', 'llywood', 'ccm', 'filmi', 'kirtan', 'sad sierreno',
                    'mexicana', 'worship', 'afrofuturism', 'americana', 'fingerstyle', 'salsa',
                    'musica']
        poplist = ['pop', 'songwriter', 'stomp and holler', 'neo mellow', 'sped up', 'a cappella']
        electroniclist = ['edm', 'house', 'dance', 'trance', 'tech','bass','electro', 
                        'rave', 'tronica', 'new french touch', 'beats', 'neo mellow',
                        'psych', 'hardstyle', 'rawstyle', 'dnb', 'frenchcore', 'tekk',
                        'phonk', 'breakcore', 'otacore', 'glitch', 'dubstep', 'chillwave']
        soullist = ['soul', 'funk', 'r b', 'motown', 'jazz', 'quiet storm', 'reggae', 'disco',
                    'r&b', 'adult standards', 'gospel', 'high vibe', 'lo fi']
        raplist = ['rap', 'hop', 'trap', 'drill', 'pluggnb', 'gymcore']
        instrumentallist = ['classical', 'meditation', 'background', 'orchestral', 'soundtrack', 
                            'ambient', 'sleep', 'instrumental', 'pet calming','vgm', 'pixel', 
                            'piano cover', 'video game music', 'classify', 'pink noise', 
                            'movie tunes', 'piano', 'ukulele', 'orchestra']

        #instantiate new list and genre dictionary with counts
        grouped_genre_list = []
        for artist in new_genre_list:
            genre_counts = {
            'rock': 0,
            'metal': 0,
            'folk': 0,
            'pop': 0,
            'electronic': 0,
            'soul': 0,
            'rap': 0,
            'instrumental': 0,
            'other': 0
            }

        # count the genres for each artist and update the dictionary with a counter
            for genre in artist:

                if any(keyword in genre for keyword in rocklist):
                    genre_counts['rock'] += 1
                if any(keyword in genre for keyword in metallist):
                    genre_counts['metal'] += 1
                if any(keyword in genre for keyword in folklist):
                    genre_counts['folk'] += 1
                elif any(keyword in genre for keyword in raplist):
                    genre_counts['rap'] += 1
                elif any(keyword in genre for keyword in electroniclist):
                    genre_counts['electronic'] += 1
                elif any(keyword in genre for keyword in soullist):
                    genre_counts['soul'] += 1
                elif any(keyword in genre for keyword in poplist):
                    genre_counts['pop'] += 1
                elif any(keyword in genre for keyword in instrumentallist):
                    genre_counts['instrumental'] += 1
                else:
                    genre_counts['other'] += 0.1  # give 'unknown' a lower weighting

            # sort the dictionary by values in descending order
            sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

            # top genre is the first key in the sorted dict, except in the case of a tie,\
            #  in which case the winner is chosen at random from the top scorers
            matching_indices = [0]

            for i in range(1, len(sorted_genres)):
                if sorted_genres[i][1] == sorted_genres[0][1]:
                    matching_indices.append(i)
            random.seed(666)
            chosen_index = random.choice(matching_indices)

            max_genre = sorted_genres[chosen_index][0]

            # add genre to the list for that artist
            grouped_genre_list.append(max_genre)

        # add as new df column and drop the original
        df['grouped_genres'] = grouped_genre_list

        # convert genres into dummies and drop original
        genre_dummies = pd.get_dummies(df['grouped_genres'], prefix="genre")
        df = pd.concat([df, genre_dummies], axis=1)

        df.drop(columns='grouped_genres', inplace=True)
        df.drop(columns="artist_genre", inplace=True)

        # calculate median of numeric duration_ms values
        numeric_durations = pd.to_numeric(df['duration_ms'], errors='coerce')
        median_duration = np.median(numeric_durations[~np.isnan(numeric_durations)])

        # apply lambda function to duration_ms column
        df['duration_ms'] = df['duration_ms'].apply(lambda x: int(x) if pd.to_numeric(x, errors='coerce')==x else median_duration)
        df['duration_ms'] = df['duration_ms'].apply(lambda x: min(int(x), 707600))
        
        # winsorize for modelling
        df['duration_ms'] = df['duration_ms'].apply(lambda x: min(int(x), 707600))


        # map to common or not common time
        df["common_time"] = df["time_signature"].map(
            {
                4: 1,
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                5: 0,
                
            }
        )


        df.drop(columns='time_signature', inplace=True)

        print("Pre-process 'key' column and create dummies")
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

        print("Binarize 'explicit' column")
        # set the explicit tag true or false to a 1 or 0
        df["explicit"] = df["explicit"].astype(int)

        EC_HS_features = joblib.load("./pickles/ec_hs_features.pkl")

        #load full column list
        fullcols = joblib.load("./pickles/full_col_list.pkl")

        # add column as 0 if not in dataframe (e.g. genre or key not present in playlist)
        for col in fullcols:
            if col not in df:
                df[col] = 0


        # if track not in full dataset, append the preproc track to it for future use
        ids_to_add = df['track_id']
        for id in ids_to_add:
            series_add = df.loc[df['track_id'] == id].iloc[0]
            full_df = full_df.append(series_add, ignore_index=True)

        joblib.dump(full_df, "./pickles/compressed_song_dataset.pkl", compress=9)


        # combine tracks that have been loaded and tracks that have been loaded from dataset
        df = pd.concat([df, trax_df], axis=0)

        # reset index
        df.reset_index(inplace=True, drop=True)


        # keep the common EC, HS features for the modelling
        df_sel = df[EC_HS_features]

        # EC fitted pipeline
        ec_pipe = joblib.load("./pickles/ec_fitted_pipeline.pkl")

        # HS fitted pipeline
        hs_pipe = joblib.load("./pickles/hs_fitted_pipeline.pkl")

        # there are some '[]' and nans where a track has been skipped
        # fill these with median values, can deal with them properly
        for col in df_sel.columns:
            if df_sel[col].dtype == 'object' and '[]' in df_sel[col].unique():
                # If the column is of object type and contains '[]', replace it with NaN
                df_sel[col] = df_sel[col].replace('[]', np.nan)
        
            # Calculate median for the column
            median = np.nanmedian(df_sel[col].astype(float))
            
            # Fill NaN values with the median of the column
            df_sel[col] = df_sel[col].fillna(median).astype(float)

        
        #get the energy score
        X_energy_score = (ec_pipe.predict_proba(df_sel))[:,1]
        #get the happy score
        X_happy_score = hs_pipe.predict_proba(df_sel)[:,0]


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

        number_of_tracks = df.shape[0]


        # create a new column and set it to 0 for all rows
        df['selected'] = 0
        # map selected numerical to a string for plotting
        df["selected_str"] = df["selected"].map(
            {
                0: 'No',
                1: 'Yes',
                
            }
        )
        fig1 = px.scatter(df, x="energy_coeff", y="happy_coeff", hover_data=["artists", "track_name"], height=900,width=900,opacity=0.7,
                        color=df["playlist_name"],
                        labels={"happy_coeff": "Happy Score",
                                "energy_coeff": "Energy Score",
                                "artists": "Artist",
                                "track_name": "Track",
                                "playlist_name": "Playlist"})
        fig1.add_vline(x=0, line_width=2, line_color="grey")
        fig1.add_hline(y=0, line_width=2, line_color="grey")
        # Add text annotations
        fig1.update_layout(annotations=[
                dict(x=left_quad,y=top_quad,xref="x",yref="y",text="Happy & Chilled",showarrow=False,font=dict(size=20, color='lightgray')),
                dict(x=right_quad,y=top_quad,xref="x",yref="y",text="Happy & Energetic",showarrow=False,font=dict(size=20, color='lightgray')),
                dict(x=left_quad,y=bottom_quad,xref="x",yref="y",text="Sad & Chilled",showarrow=False,font=dict(size=20, color='lightgray')),
                dict(x=right_quad,y=bottom_quad,xref="x",yref="y",text="Sad & Energetic",showarrow=False,font=dict(size=20, color='lightgray'))])
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
            
            find_happy = st.slider(label=(f"The input playlist has a happy score ranging from {happy_min} to {happy_max}, how happy do you want your new playlist to be?: "), value=0.0, step=0.1, min_value=happy_min, max_value=happy_max)
            find_energy = st.slider(label=(f"The input playlist has an energy score ranging from {energy_min} to {energy_max}, how energetic do you want your new playlist to be?: "), value=0.0, step=0.1, min_value=energy_min, max_value=energy_max)
            new_playlist_name = st.text_input("Give your new playlist a name: ")

            kneigh = st.number_input(label=(f"How many tracks out of the original {number_of_tracks} do you want in your playlist?: "), min_value=1, max_value=number_of_tracks, value=5, step=1)

            user_submit = st.form_submit_button(label="Submit")


        if user_submit:
            
            # lambda function to create new column - euclidean distance away from selected happy, energy score
            X_coeffs['dist'] = np.sqrt((np.square(find_energy - X_coeffs['energy_coeff']) + np.square(find_happy - X_coeffs['happy_coeff'])))
            # find indices of top "Kneigh" results
            Kneighbours_index = list(X_coeffs.sort_values(by="dist", ascending=True).head(kneigh).index)


            # set the new column to 1 at the specified indexes
            for i in Kneighbours_index:
                df.at[i, 'selected'] = 1

            # initialise the track, artist, track id lists
            track_list = []
            artist_list = []
            track_id_list = []

            # populate the lists with the nearest neighbours
            for i in Kneighbours_index:
                track_name = df.loc[i]['track_name']
                track_list.append(track_name)
                artist_name = df.loc[i]['artists']
                artist_list.append(artist_name)
                track_id = df.loc[i]['track_id']
                track_id_list.append(track_id)


            # re-map selected numerical to a string for plotting
            df["selected_str"] = df["selected"].map(
                {
                    0: 'No',
                    1: 'Yes',
                    
                }
            )

            # add new playlist
            fig2 = px.scatter(df, x="energy_coeff", y="happy_coeff", color=df["selected_str"].map({"No": "Original", "Yes": new_playlist_name}),
                            hover_data=["artists", "track_name"],height=900,width=900,opacity=0.7,
                            color_discrete_map={playlist_input_name: 'lightslategray',new_playlist_name: 'magenta'},
                            labels={"happy_coeff": "Happy Score",
                                    "energy_coeff": "Energy Score",
                                    "artists": "Artist",
                                    "track_name": "Track",
                                    "selected_str": "Track selected?",
                                    "color": "Playlist"},
                            )
            fig2.add_vline(x=0, line_width=2, line_color="grey")
            fig2.add_hline(y=0, line_width=2, line_color="grey")

            # Add text annotations
            fig2.update_layout(annotations=[
                    dict(x=left_quad,y=top_quad,xref="x",yref="y",text="Happy & Chilled",showarrow=False,font=dict(size=20, color='lightgray')),
                    dict(x=right_quad,y=top_quad,xref="x",yref="y",text="Happy & Energetic",showarrow=False,font=dict(size=20, color='lightgray')),
                    dict(x=left_quad,y=bottom_quad,xref="x",yref="y",text="Sad & Chilled",showarrow=False,font=dict(size=20, color='lightgray')),
                    dict(x=right_quad,y=bottom_quad,xref="x",yref="y",text="Sad & Energetic",showarrow=False,font=dict(size=20, color='lightgray'))])
            fig2.update_yaxes(
            range=(plotting_happy_min, plotting_happy_max),
            constrain='domain')
            fig2.update_xaxes(
            range=(plotting_energy_min, plotting_energy_max),
            constrain='domain')
            #fig.show()

            st.plotly_chart(fig2, theme=None, use_container_width=True)


            # # load credentials from .env file
            # load_dotenv()

            # CLIENT_ID = os.getenv("CLIENT_ID", "")
            # CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")

            def get_token(oauth, code):

                token = oauth.get_access_token(code, as_dict=False, check_cache=False)
                # remove cached token saved in directory
                os.remove(".cache")
    
                # return the token
                return token
            
            def sign_in(token):
                sp = spotipy.Spotify(auth=token)
                return sp
            
            def app_get_token():
                try:
                    token = get_token(st.session_state["oauth"], st.session_state["code"])
                except Exception as e:
                    st.error("An error occurred during token retrieval!")
                    st.write("The error is as follows:")
                    st.write(e)
                else:
                    st.session_state["cached_token"] = token

            def app_sign_in():
                try:
                    sp = sign_in(st.session_state["cached_token"])
                except Exception as e:
                    st.error("An error occurred during sign-in!")
                    st.write("The error is as follows:")
                    st.write(e)
                else:
                    st.session_state["signed_in"] = True
                    app_display_welcome()
                    st.success("Sign in success!")
                    
                return sp
            
            def app_display_welcome():
    
                # import secrets from streamlit deployment
                cid = st.secrets["CLIENT_ID"]
                csecret = st.secrets["CLIENT_SECRET"]
                uri = "http://localhost:8888/callback"

                # set scope and establish connection
                scopes = " ".join(["playlist-modify-private",
                                "playlist-modify-public"])

                # create oauth object
                oauth = SpotifyOAuth(scope=scopes,
                                    redirect_uri=uri,
                                    client_id=cid,
                                    client_secret=csecret)
                # store oauth in session
                st.session_state["oauth"] = oauth

                # retrieve auth url
                auth_url = oauth.get_authorize_url()

                # this SHOULD open the link in the same tab when Streamlit Cloud is updated
                # via the "_self" target
                link_html = " <a target=\"_self\" href=\"{url}\" >{msg}</a> ".format(
                    url=auth_url,
                    msg="Click me to authenticate!"
                )
                                
                # define welcome
                welcome_msg = """
                Welcome! :wave: This app uses the Spotify API to interact with general 
                music info and your playlists! In order to view and modify information 
                associated with your account, you must log in. You only need to do this 
                once.
                """
                if not st.session_state["signed_in"]:
                    st.markdown(welcome_msg)
                    st.write(" ".join(["No tokens found for this session. Please log in by",
                                    "clicking the link below."]))
                    st.markdown(link_html, unsafe_allow_html=True)

        if "signed_in" not in st.session_state:
            st.session_state["signed_in"] = False
        if "cached_token" not in st.session_state:
            st.session_state["cached_token"] = ""
        if "code" not in st.session_state:
            st.session_state["code"] = ""
        if "oauth" not in st.session_state:
            st.session_state["oauth"] = None


        # get current url (stored as dict)
        url_params = st.experimental_get_query_params()

        # attempt sign in with cached token
        if st.session_state["cached_token"] != "":
            sp = app_sign_in()
        # if no token, but code in url, get code, parse token, and sign in
        elif "code" in url_params:
            # all params stored as lists, see doc for explanation
            st.session_state["code"] = url_params["code"][0]
            app_get_token()
            sp = app_sign_in()
        # otherwise, prompt for redirect
        else:
            app_display_welcome()

        # only display the following after login
        ### is there another way to do this? clunky to have everything in an if:
        if st.session_state["signed_in"]:
            user = sp.current_user()
            name = user["display_name"]
            username = user["id"]


            # # authenticate
            # client_credentials_manager = SpotifyClientCredentials(
            #     client_id=CLIENT_ID, client_secret=CLIENT_SECRET
            # )

            # sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
            #                                             client_secret=CLIENT_SECRET,
            #                                             redirect_uri="http://localhost:8888/callback",
            #                                             scope="playlist-modify-private"))

            
            
            # user = sp.current_user()["id"]

            
            playlist_description = f"A playlist created using MoodGrid from {plname_list[:]} with a Happy rating of {find_happy} and an Energy rating of {find_energy}"

            playlist = sp.user_playlist_create(user=username, name=new_playlist_name, public=False, description=playlist_description)

            
            # split track_id_list into batches of 100, as Spotify only lets you add 100 tracks at a time
            batches = [track_id_list[i:i+100] for i in range(0, len(track_id_list), 100)]

            # add each batch to the playlist
            for batch in batches:
                sp.playlist_add_items(playlist["id"], batch)
                time.sleep(sleep_time)

            
            new_playlist_URL = playlist['external_urls']['spotify']
            st.write(f"Playlist '{new_playlist_name}' has been added to Spotify")
            st.write(f"Click the link the check it out! {new_playlist_URL}")