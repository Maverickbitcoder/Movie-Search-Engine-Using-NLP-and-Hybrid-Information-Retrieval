"""
Streamlit Web Application: MovieLens Media Retrieval Search Engine
Run with: streamlit run streamlit_app.py
"""

import os
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Page Config & Styling
st.set_page_config(
    page_title="MovieLens IR Search Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .movie-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .movie-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0f172a;
    }
    .movie-genres {
        font-size: 0.85rem;
        color: #2563eb;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .movie-tags {
        font-size: 0.9rem;
        color: #334155;
    }
    .badge-score {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_corpus_and_models():
    """Loads corpus and initializes TF-IDF, BM25, and Dense Sentence Transformers."""
    movies_df = pd.read_csv('movies.csv')
    tags_df = pd.read_csv('tags.csv')
    tags_df['tag'] = tags_df['tag'].fillna('').astype(str).str.strip()
    tags_clean = tags_df[tags_df['tag'] != ''].drop_duplicates(subset=['userId', 'movieId', 'tag'])

    movie_tags_agg = tags_clean.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index(name='all_tags')
    df = movies_df.merge(movie_tags_agg, on='movieId', how='left')
    df['all_tags'] = df['all_tags'].fillna('')
    df['genres_clean'] = df['genres'].str.replace('|', ' ', regex=False).str.replace('(no genres listed)', '', regex=False).str.strip()
    df['search_document'] = (df['title'] + ' ' + df['genres_clean'] + ' ' + df['all_tags']).str.strip()
    
    # Active 2,500 movies with richest metadata
    active_df = df[df['search_document'].str.len() > 5].head(2500).copy().reset_index(drop=True)

    # 1. TF-IDF
    tfidf_vec = TfidfVectorizer(max_features=15000, stop_words='english')
    tfidf_mat = tfidf_vec.fit_transform(active_df['search_document'])

    # 2. BM25
    tokenized_corpus = [doc.lower().split() for doc in active_df['search_document']]
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

    # 3. Dense Sentence Transformers
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    dense_embeddings = st_model.encode(active_df['search_document'].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    return active_df, tfidf_vec, tfidf_mat, bm25_model, st_model, dense_embeddings

with st.spinner("Initializing IR Search Engines & Preloading Dense Vectors..."):
    active_df, tfidf_vec, tfidf_mat, bm25_model, st_model, dense_embeddings = load_corpus_and_models()

# App Header
st.markdown("<div class='main-title'>🎬 MovieLens Media Retrieval Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Explore Classical Lexical (BM25, TF-IDF), Neural Dense (Sentence Transformers), and Hybrid Reciprocal Rank Fusion (RRF).</div>", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Search Engine Settings")
model_choice = st.sidebar.selectbox(
    "Ranking Architecture",
    options=["Hybrid (RRF)", "BM25 (Probabilistic Lexical)", "Dense Semantic (Sentence Transformers)", "TF-IDF (Cosine Similarity)"],
    index=0
)
top_k = st.sidebar.slider("Number of Results (Top K)", min_value=1, max_value=20, value=6)
rrf_k = st.sidebar.slider("RRF Smoothing Parameter (k)", min_value=10, max_value=100, value=60) if "Hybrid" in model_choice else 60

st.sidebar.markdown("---")
st.sidebar.markdown("**Example Queries to Try:**")
if st.sidebar.button("🌌 Sci-Fi Space War"):
    st.session_state['query_input'] = "star wars galactic empire space adventure"
if st.sidebar.button("💭 Mind-Bending Thriller"):
    st.session_state['query_input'] = "christopher nolan dream mind bending reality thriller"
if st.sidebar.button("🧸 Talking Toys Cartoon"):
    st.session_state['query_input'] = "hilarious talking animal animated cartoon"
if st.sidebar.button("🕶️ Tarantino Mob Crime"):
    st.session_state['query_input'] = "quentin tarantino pulp gangster violent crime"

default_query = st.session_state.get('query_input', "futuristic cyberpunk detective dystopia matrix")
query = st.text_input("Enter Movie Search Query:", value=default_query)

if query and query.strip():
    t_start = time.time()
    query_str = query.strip()

    if "Hybrid" in model_choice:
        b_scores = bm25_model.get_scores(query_str.lower().split())
        b_top = np.argsort(b_scores)[::-1][:40]
        
        q_vec = st_model.encode([query_str], normalize_embeddings=True)[0]
        d_scores = np.dot(dense_embeddings, q_vec)
        d_top = np.argsort(d_scores)[::-1][:40]
        
        rrf_dict = {}
        for r, idx in enumerate(b_top, 1):
            rrf_dict[idx] = rrf_dict.get(idx, 0.0) + (1.0 / (rrf_k + r))
        for r, idx in enumerate(d_top, 1):
            rrf_dict[idx] = rrf_dict.get(idx, 0.0) + (1.0 / (rrf_k + r))
            
        ranked_indices = [idx for idx, _ in sorted(rrf_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        scores = [round(rrf_dict[idx], 5) for idx in ranked_indices]
        score_name = "RRF Fusion Score"
        
    elif "BM25" in model_choice:
        b_scores = bm25_model.get_scores(query_str.lower().split())
        ranked_indices = np.argsort(b_scores)[::-1][:top_k]
        scores = [round(float(b_scores[idx]), 3) for idx in ranked_indices]
        score_name = "BM25 Score"
        
    elif "Dense" in model_choice:
        q_vec = st_model.encode([query_str], normalize_embeddings=True)[0]
        d_scores = np.dot(dense_embeddings, q_vec)
        ranked_indices = np.argsort(d_scores)[::-1][:top_k]
        scores = [round(float(d_scores[idx]), 4) for idx in ranked_indices]
        score_name = "Cosine Similarity"
        
    else:  # TF-IDF
        q_vec = tfidf_vec.transform([query_str])
        sims = cosine_similarity(q_vec, tfidf_mat).flatten()
        ranked_indices = np.argsort(sims)[::-1][:top_k]
        scores = [round(float(sims[idx]), 4) for idx in ranked_indices]
        score_name = "Cosine Similarity"

    latency_ms = (time.time() - t_start) * 1000

    st.markdown(f"**Retrieved {len(ranked_indices)} movies using {model_choice} in `{latency_ms:.2f} ms`**")
    
    # Display Movie Cards
    for rank, (idx, score) in enumerate(zip(ranked_indices, scores), 1):
        row = active_df.iloc[idx]
        tags = row['all_tags'][:140] + "..." if len(row['all_tags']) > 140 else (row['all_tags'] or "No user tags listed.")
        
        st.markdown(f"""
        <div class='movie-card'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div class='movie-title'>#{rank}. {row['title']}</div>
                <div class='badge-score'>{score_name}: {score}</div>
            </div>
            <div class='movie-genres'>🎭 {row['genres']}</div>
            <div class='movie-tags'>🏷️ <b>Tags & Themes:</b> {tags}</div>
        </div>
        """, unsafe_allow_html=True)
