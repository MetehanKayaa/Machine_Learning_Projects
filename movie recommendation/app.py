import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    # API'den veri çekme
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    try:
        data = requests.get(url)
        data = data.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster" # Poster yoksa boş resim
    except:
        return "https://via.placeholder.com/500x750?text=Error"



def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_movies_poster = []
    
    for i in movie_list:
        # i[0] satır indeksidir. Gerçek movie_id'yi DataFrame'den çekiyoruz:
        actual_movie_id = movies.iloc[i[0]].movie_id 
        
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_poster.append(fetch_poster(actual_movie_id)) # Gerçek ID'yi gönderiyoruz
        
    return recommended_movies, recommended_movies_poster

# Verileri yükleme (Dosya yollarına dikkat!)
movies_dict = pickle.load(open('./movie/movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open(r'C:\Users\computeer\Desktop\end_to_end\movie\similarity.pkl','rb'))

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Hangi filme benzer öneriler istersin?',
    movies['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    
    # st.beta_columns yerine st.columns kullanıyoruz
    cols = st.columns(5) 
    
    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])














