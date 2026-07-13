import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="TuneMatch AI",
    page_icon="🎵",
    layout="wide"
)

# ---------------------------------
# Custom CSS
# ---------------------------------

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

h1{
    color:#1DB954;
    text-align:center;
}

.stButton>button{
    background-color:#1DB954;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
    font-weight:bold;
}

.song-card{
    background:#262730;
    padding:15px;
    border-radius:10px;
    margin-bottom:12px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Dataset
# ---------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("clean_song_data.csv")

    df = df[['track_name', 'artist_name', 'tags']]

    df.dropna(inplace=True)

    df.drop_duplicates(subset="track_name", inplace=True)

    df.reset_index(drop=True, inplace=True)

    df["track_name"] = df["track_name"].str.lower()

    df["artist_name"] = df["artist_name"].astype(str)

    df["tags"] = df["tags"].astype(str)

    return df


songs = load_data()

# ---------------------------------
# Create Similarity Matrix
# ---------------------------------

@st.cache_resource
def create_similarity(df):

    tfidf = TfidfVectorizer(stop_words="english")

    matrix = tfidf.fit_transform(df["tags"])

    similarity = cosine_similarity(matrix)

    return similarity


similarity = create_similarity(songs)

# ---------------------------------
# Recommendation Function
# ---------------------------------

def recommend(song):

    song = song.lower()

    if song not in songs["track_name"].values:
        return []

    index = songs[songs["track_name"] == song].index[0]

    distances = list(enumerate(similarity[index]))

    recommended = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    output = []

    for i in recommended:

        output.append({

            "song": songs.iloc[i[0]]["track_name"].title(),

            "artist": songs.iloc[i[0]]["artist_name"]

        })

    return output

# ---------------------------------
# Sidebar
# ---------------------------------

st.sidebar.title("🎵 TuneMatch AI")

st.sidebar.write("AI Powered Song Recommendation")

st.sidebar.markdown("---")

st.sidebar.metric("Songs", len(songs))

st.sidebar.metric("Artists", songs["artist_name"].nunique())

st.sidebar.markdown("---")

st.sidebar.success("Content Based Recommendation")

# ---------------------------------
# Main Page
# ---------------------------------

st.title("🎵 TuneMatch AI")

st.write("Find songs similar to your favourite song.")

selected_song = st.selectbox(

    "Choose a Song",

    songs["track_name"].str.title()

)

if st.button("🎧 Recommend Songs"):

    recommendations = recommend(selected_song)

    if len(recommendations) == 0:

        st.error("Song not found!")

    else:

        st.success("Top 5 Recommended Songs")

        for item in recommendations:

            st.markdown(f"""
            <div class="song-card">

            <h3>🎵 {item['song']}</h3>

            <p>🎤 Artist : {item['artist']}</p>

            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

st.caption("Made with ❤️ using Python, Streamlit and Machine Learning")