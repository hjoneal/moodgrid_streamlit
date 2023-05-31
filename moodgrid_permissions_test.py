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
    This app uses the Spotify API to interact with general 
    music info and your playlists! In order to view and modify information 
    associated with your account, you must log in. You only need to do this 
    once.
    """
    
    # # define temporary note
    # note_temp = """
    # _Note: Unfortunately, the current version of Streamlit will not allow for
    # staying on the same page, so the authorization and redirection will open in a 
    # new tab. This has already been addressed in a development release, so it should
    # be implemented in Streamlit Cloud soon!_
    # """

    if not st.session_state["signed_in"]:
        st.markdown(welcome_msg)
        st.write(" ".join(["No tokens found for this session. Please log in by",
                        "clicking the link below."]))
        st.markdown(link_html, unsafe_allow_html=True)
        # st.markdown(note_temp)

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

st.write("URL params:", url_params)

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


    
    # playlist_description = f"A playlist created using MoodGrid from {plname_list[:]} with a Happy rating of {find_happy} and an Energy rating of {find_energy}"

    # playlist = sp.user_playlist_create(user=username, name=new_playlist_name, public=False, description=playlist_description)

    
    # # split track_id_list into batches of 100, as Spotify only lets you add 100 tracks at a time
    # batches = [track_id_list[i:i+100] for i in range(0, len(track_id_list), 100)]

    # # add each batch to the playlist
    # for batch in batches:
    #     sp.playlist_add_items(playlist["id"], batch)
    #     time.sleep(sleep_time)

    
    # new_playlist_URL = playlist['external_urls']['spotify']
    # st.write(f"Playlist '{new_playlist_name}' has been added to Spotify")
    # st.write(f"Click the link the check it out! {new_playlist_URL}")