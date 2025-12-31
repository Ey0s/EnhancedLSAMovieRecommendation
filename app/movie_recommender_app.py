"""
Enhanced Streamlit Web Application for LSA-based Movie Recommendation System
============================================================================
Advanced content-based movie recommender using Enhanced Latent Semantic Analysis (LSA)
with hybrid features, weighted text processing, and intelligent recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import requests
import joblib
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# TMDB CONFIGURATION
# =========================
TMDB_API_KEY = "f5a9653ca297a51265b464d01e1484ea"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🎬 LSA Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF6B6B;
    text-align: center;
    margin-bottom: 1.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.movie-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.similarity-score {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    color: white;
    padding: 0.5rem;
    border-radius: 5px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# ENHANCED RECOMMENDER CLASS
# =========================
class EnhancedMovieRecommender:
    def __init__(self):
        self.df = None
        self.similarity_matrix = None
        self.movie_to_idx = None
        self.tfidf_model = None
        self.lsa_model = None
        self.hybrid_features = None
    
    def load_models(self):
        """Load all enhanced models and data"""
        try:
            # Determine the correct path based on current working directory
            if os.path.exists('data/processed/movies_enhanced_df.csv'):
                data_path = 'data/processed/movies_enhanced_df.csv'
                models_path = 'models'
            else:
                data_path = '../data/processed/movies_enhanced_df.csv'
                models_path = '../models'
            
            # Load dataframe
            self.df = pd.read_csv(data_path)
            
            # Load models
            self.tfidf_model = joblib.load(f'{models_path}/tfidf_enhanced.pkl')
            self.lsa_model = joblib.load(f'{models_path}/lsa_enhanced.pkl')
            self.hybrid_features = joblib.load(f'{models_path}/hybrid_features_enhanced.pkl')
            self.similarity_matrix = joblib.load(f'{models_path}/similarity_enhanced.pkl')
            
            # Create movie index mapping
            self.movie_to_idx = {title: idx for idx, title in enumerate(self.df['original_title'])}
            
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def convert_standardized_rating(self, standardized_rating):
        """Convert standardized rating back to original 0-10 scale"""
        # Based on typical movie rating distribution, approximate conversion
        # Standardized ratings are z-scores, so we need to reverse the standardization
        # Using approximate values: mean ≈ 6.2, std ≈ 1.2 for movie ratings
        original_rating = (standardized_rating * 1.2) + 6.2
        return max(0, min(10, original_rating))  # Clamp to 0-10 range
    
    def convert_standardized_rating_filter(self, original_rating_filter):
        """Convert original rating filter to standardized scale for filtering"""
        # Convert user's 0-10 rating to standardized scale for filtering
        return (original_rating_filter - 6.2) / 1.2
    
    def convert_standardized_numeric(self, standardized_value, feature_type):
        """Convert standardized numeric values back to original scale"""
        # Approximate conversion based on typical movie data distributions
        if feature_type == 'vote_count':
            # vote_count: log-transformed then standardized
            # Reverse: (z * std) + mean, then exp - 1
            log_value = (standardized_value * 2.5) + 6.5  # Approximate log scale
            return max(0, int(np.exp(log_value) - 1))
        
        elif feature_type == 'budget':
            # budget: log-transformed then standardized
            log_value = (standardized_value * 1.8) + 17.5  # Approximate log scale
            return max(0, int(np.exp(log_value) - 1))
        
        elif feature_type == 'revenue':
            # revenue: log-transformed then standardized
            log_value = (standardized_value * 2.0) + 18.0  # Approximate log scale
            return max(0, int(np.exp(log_value) - 1))
        
        elif feature_type == 'runtime':
            # runtime: log-transformed then standardized
            log_value = (standardized_value * 0.3) + 4.7  # Approximate log scale
            return max(0, int(np.exp(log_value) - 1))
        
        return standardized_value
    
    def get_recommendations(self, movie_title, n_recommendations=10, min_rating=6.0):
        """Get movie recommendations using enhanced LSA"""
        if movie_title not in self.movie_to_idx:
            return {'error': f"Movie '{movie_title}' not found in dataset."}
        
        movie_idx = self.movie_to_idx[movie_title]
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Convert min_rating to standardized scale for filtering
        min_rating_standardized = self.convert_standardized_rating_filter(min_rating)
        
        recommendations = []
        for idx, score in sim_scores[1:]:  # Skip the movie itself
            movie_data = self.df.iloc[idx]
            
            # Apply rating filter using standardized values
            if movie_data['vote_average'] < min_rating_standardized:
                continue
            
            # Parse genres safely
            try:
                genres = eval(movie_data['genres_list']) if isinstance(movie_data['genres_list'], str) else []
            except:
                genres = []
            
            # Parse directors safely
            try:
                directors = eval(movie_data['director_list']) if isinstance(movie_data['director_list'], str) else []
            except:
                directors = []
            
            # Parse cast safely
            try:
                cast = eval(movie_data['cast_list']) if isinstance(movie_data['cast_list'], str) else []
            except:
                cast = []
            
            # Convert standardized values back to original scale for display
            original_rating = self.convert_standardized_rating(movie_data['vote_average'])
            original_vote_count = self.convert_standardized_numeric(movie_data['vote_count'], 'vote_count')
            
            recommendations.append({
                'title': movie_data['original_title'],
                'year': int(movie_data['release_year']) if pd.notna(movie_data['release_year']) else 'Unknown',
                'genres': ', '.join(genres[:3]) if genres else 'N/A',
                'director': ', '.join(directors[:2]) if directors else 'N/A',
                'cast': ', '.join(cast[:3]) if cast else 'N/A',
                'rating': round(original_rating, 1),
                'vote_count': original_vote_count,
                'overview': movie_data['overview'] if pd.notna(movie_data['overview']) else 'N/A',
                'similarity_score': score
            })
            
            if len(recommendations) >= n_recommendations:
                break
        
        return {'recommendations': recommendations}
    
    def get_movie_info(self, movie_title):
        """Get detailed information about a specific movie"""
        if movie_title not in self.movie_to_idx:
            return {'error': f"Movie '{movie_title}' not found in dataset."}
        
        movie_idx = self.movie_to_idx[movie_title]
        movie = self.df.iloc[movie_idx]
        
        # Parse lists safely
        try:
            genres = eval(movie['genres_list']) if isinstance(movie['genres_list'], str) else []
            directors = eval(movie['director_list']) if isinstance(movie['director_list'], str) else []
            cast = eval(movie['cast_list']) if isinstance(movie['cast_list'], str) else []
        except:
            genres = directors = cast = []
        
        # Convert standardized values back to original scale
        original_rating = self.convert_standardized_rating(movie['vote_average'])
        original_vote_count = self.convert_standardized_numeric(movie['vote_count'], 'vote_count')
        original_budget = self.convert_standardized_numeric(movie['budget'], 'budget')
        original_revenue = self.convert_standardized_numeric(movie['revenue'], 'revenue')
        original_runtime = self.convert_standardized_numeric(movie['runtime'], 'runtime')
        
        return {
            'title': movie['original_title'],
            'year': int(movie['release_year']) if pd.notna(movie['release_year']) else 'Unknown',
            'genres': ', '.join(genres) if genres else 'N/A',
            'director': ', '.join(directors) if directors else 'N/A',
            'cast': ', '.join(cast[:5]) if cast else 'N/A',
            'rating': round(original_rating, 1),
            'vote_count': original_vote_count,
            'overview': movie['overview'] if pd.notna(movie['overview']) else 'N/A',
            'budget': original_budget,
            'revenue': original_revenue,
            'runtime': original_runtime
        }

# =========================
# LOAD ENHANCED MODEL
# =========================
@st.cache_resource(show_spinner=False)
def load_enhanced_recommender():
    """Load the enhanced movie recommender"""
    recommender = EnhancedMovieRecommender()
    if recommender.load_models():
        return recommender
    return None
# =========================
# AUTOCOMPLETE FUNCTION
# =========================
def get_movie_autocomplete(recommender, query, limit=10):
    if not query or len(query) < 2:
        return []

    titles = recommender.df['original_title'].dropna().unique()
    query = query.lower()

    matches = [t for t in titles if t.lower().startswith(query)]
    return sorted(matches)[:limit]

# =========================
# TMDB POSTER FETCH
# =========================
def fetch_movie_poster(movie_title):
    try:
        params = {"api_key": TMDB_API_KEY, "query": movie_title}
        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
    except Exception:
        pass
    return None

# =========================
# ENHANCED MOVIE CARD
# =========================
def display_movie_card(movie_info, rank=None):
    """Display an enhanced movie card with better styling"""
    poster_url = fetch_movie_poster(movie_info['title'])

    with st.container():
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        
        col0, col1, col2 = st.columns([1, 3, 1])

        with col0:
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                st.markdown("🎬")
                st.caption("No poster")

        with col1:
            title = f"{rank}. {movie_info['title']}" if rank else movie_info['title']
            st.markdown(f"### {title}")
            
            if movie_info.get('year', 'Unknown') != 'Unknown':
                st.markdown(f"**Year:** {movie_info['year']}")
            
            if movie_info.get('director', 'N/A') != 'N/A':
                st.markdown(f"**Director:** {movie_info['director']}")
            
            if movie_info.get('genres', 'N/A') != 'N/A':
                st.markdown(f"**Genres:** {movie_info['genres']}")
            
            if movie_info.get('cast', 'N/A') != 'N/A':
                st.markdown(f"**Cast:** {movie_info['cast']}")
            
            if movie_info.get("overview") not in [None, "N/A", ""]:
                overview = movie_info['overview']
                if len(overview) > 200:
                    overview = overview[:200] + "..."
                st.markdown(f"**Plot:** {overview}")

        with col2:
            if 'similarity_score' in movie_info:
                st.markdown(
                    f'<div class="similarity-score">{movie_info["similarity_score"]:.1%}</div>',
                    unsafe_allow_html=True
                )
                st.caption("Similarity")
            
            if movie_info.get("rating") not in [None, "N/A", 0]:
                st.metric("⭐ Rating", f"{movie_info['rating']}/10")
            
            if movie_info.get("vote_count") not in [None, "N/A", 0]:
                votes = movie_info['vote_count']
                if votes >= 1000:
                    votes_str = f"{votes/1000:.1f}K"
                else:
                    votes_str = str(votes)
                st.metric("🗳️ Votes", votes_str)
            
            # Additional metrics for detailed view
            if 'budget' in movie_info and movie_info['budget'] > 0:
                budget = movie_info['budget']
                if budget >= 1000000000:
                    budget_str = f"${budget/1000000000:.1f}B"
                elif budget >= 1000000:
                    budget_str = f"${budget/1000000:.1f}M"
                elif budget >= 1000:
                    budget_str = f"${budget/1000:.1f}K"
                else:
                    budget_str = f"${budget:,.0f}"
                st.metric("💰 Budget", budget_str)
            
            # Add revenue if available
            if 'revenue' in movie_info and movie_info['revenue'] > 0:
                revenue = movie_info['revenue']
                if revenue >= 1000000000:
                    revenue_str = f"${revenue/1000000000:.1f}B"
                elif revenue >= 1000000:
                    revenue_str = f"${revenue/1000000:.1f}M"
                elif revenue >= 1000:
                    revenue_str = f"${revenue/1000:.1f}K"
                else:
                    revenue_str = f"${revenue:,.0f}"
                st.metric("💵 Revenue", revenue_str)
            
            # Add runtime if available
            if 'runtime' in movie_info and movie_info['runtime'] > 0:
                runtime = movie_info['runtime']
                hours = runtime // 60
                minutes = runtime % 60
                if hours > 0:
                    runtime_str = f"{hours}h {minutes}m"
                else:
                    runtime_str = f"{minutes}m"
                st.metric("⏱️ Runtime", runtime_str)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

# =========================
# ENHANCED VISUALIZATION
# =========================
def create_enhanced_similarity_chart(recommendations):
    """Create an enhanced similarity chart with better styling"""
    movies = [m['title'][:30] + '...' if len(m['title']) > 30 else m['title'] 
              for m in recommendations['recommendations']]
    scores = [m['similarity_score'] for m in recommendations['recommendations']]
    ratings = [m['rating'] for m in recommendations['recommendations']]

    fig = go.Figure()
    
    # Add similarity bars
    fig.add_trace(go.Bar(
        y=movies,
        x=scores,
        orientation='h',
        name='Similarity Score',
        marker=dict(
            color=scores,
            colorscale='Viridis',
            colorbar=dict(title="Similarity Score")
        ),
        text=[f"{s:.3f}" for s in scores],
        textposition='inside'
    ))

    fig.update_layout(
        title="Movie Similarity Scores (Enhanced LSA)",
        xaxis_title="Similarity Score",
        yaxis_title="Movies",
        height=max(400, len(movies) * 25),
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        margin=dict(l=200)
    )
    
    return fig

def create_rating_distribution(recommendations):
    """Create a rating distribution chart"""
    ratings = [m['rating'] for m in recommendations['recommendations']]
    
    fig = px.histogram(
        x=ratings,
        nbins=10,
        title="Rating Distribution of Recommendations",
        labels={'x': 'Rating', 'y': 'Count'},
        color_discrete_sequence=['#FF6B6B']
    )
    
    fig.update_layout(height=300)
    return fig

# =========================
# DATA EXPLORER FUNCTIONS
# =========================
def create_data_explorer_tab(recommender):
    """Create the data explorer tab with various visualizations and statistics"""
    st.header("📊 Movie Dataset Explorer")
    
    # Convert standardized data for display
    df_display = recommender.df.copy()
    
    # Convert key metrics back to original scale
    df_display['rating_original'] = df_display['vote_average'].apply(
        lambda x: recommender.convert_standardized_rating(x)
    )
    df_display['vote_count_original'] = df_display['vote_count'].apply(
        lambda x: recommender.convert_standardized_numeric(x, 'vote_count')
    )
    df_display['budget_original'] = df_display['budget'].apply(
        lambda x: recommender.convert_standardized_numeric(x, 'budget')
    )
    df_display['revenue_original'] = df_display['revenue'].apply(
        lambda x: recommender.convert_standardized_numeric(x, 'revenue')
    )
    df_display['runtime_original'] = df_display['runtime'].apply(
        lambda x: recommender.convert_standardized_numeric(x, 'runtime')
    )
    
    # Dataset Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📽️ Total Movies", len(df_display))
    
    with col2:
        avg_rating = df_display['rating_original'].mean()
        st.metric("⭐ Average Rating", f"{avg_rating:.1f}/10")
    
    with col3:
        year_range = f"{df_display['release_year'].min():.0f} - {df_display['release_year'].max():.0f}"
        st.metric("📅 Year Range", year_range)
    
    with col4:
        total_genres = len(set([g for sublist in df_display['genres_list'].apply(
            lambda x: eval(x) if isinstance(x, str) else []
        ) for g in sublist]))
        st.metric("🎭 Unique Genres", total_genres)
    
    st.markdown("---")
    
    # Interactive Filters
    st.subheader("🔍 Interactive Filters")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        year_range = st.slider(
            "Release Year Range",
            int(df_display['release_year'].min()),
            int(df_display['release_year'].max()),
            (1990, 2020)
        )
    
    with filter_col2:
        rating_range = st.slider(
            "Rating Range",
            0.0, 10.0,
            (6.0, 10.0),
            0.1
        )
    
    with filter_col3:
        min_votes = st.slider(
            "Minimum Vote Count",
            0, 5000,
            100
        )
    
    # Apply filters
    filtered_df = df_display[
        (df_display['release_year'] >= year_range[0]) &
        (df_display['release_year'] <= year_range[1]) &
        (df_display['rating_original'] >= rating_range[0]) &
        (df_display['rating_original'] <= rating_range[1]) &
        (df_display['vote_count_original'] >= min_votes)
    ]
    
    st.info(f"📊 Showing {len(filtered_df)} movies after filtering")
    
    # Visualizations
    st.subheader("📈 Data Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Rating distribution
        fig_rating = px.histogram(
            filtered_df,
            x='rating_original',
            nbins=20,
            title="Rating Distribution",
            labels={'rating_original': 'Rating (0-10)', 'count': 'Number of Movies'},
            color_discrete_sequence=['#FF6B6B']
        )
        fig_rating.update_layout(height=400)
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with viz_col2:
        # Movies per year
        movies_per_year = filtered_df.groupby('release_year').size().reset_index(name='count')
        fig_year = px.line(
            movies_per_year,
            x='release_year',
            y='count',
            title="Movies Released Per Year",
            labels={'release_year': 'Year', 'count': 'Number of Movies'},
            color_discrete_sequence=['#4ECDC4']
        )
        fig_year.update_layout(height=400)
        st.plotly_chart(fig_year, use_container_width=True)
    
    # Genre analysis
    st.subheader("🎭 Genre Analysis")
    
    # Extract all genres
    all_genres = []
    for genres_str in filtered_df['genres_list']:
        try:
            genres = eval(genres_str) if isinstance(genres_str, str) else []
            all_genres.extend(genres)
        except:
            pass
    
    genre_counts = pd.Series(all_genres).value_counts().head(15)
    
    fig_genres = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        title="Most Popular Genres",
        labels={'x': 'Number of Movies', 'y': 'Genre'},
        color_discrete_sequence=['#9B59B6']
    )
    fig_genres.update_layout(height=500)
    st.plotly_chart(fig_genres, use_container_width=True)
    
    # Budget vs Revenue analysis
    st.subheader("💰 Budget vs Revenue Analysis")
    
    # Filter out zero values for better visualization
    budget_revenue_df = filtered_df[
        (filtered_df['budget_original'] > 0) & 
        (filtered_df['revenue_original'] > 0)
    ].copy()
    
    if len(budget_revenue_df) > 0:
        fig_budget = px.scatter(
            budget_revenue_df,
            x='budget_original',
            y='revenue_original',
            color='rating_original',
            size='vote_count_original',
            hover_data=['original_title', 'release_year'],
            title="Budget vs Revenue (colored by rating, sized by vote count)",
            labels={
                'budget_original': 'Budget ($)',
                'revenue_original': 'Revenue ($)',
                'rating_original': 'Rating'
            },
            color_continuous_scale='Viridis'
        )
        fig_budget.update_layout(height=500)
        st.plotly_chart(fig_budget, use_container_width=True)
    else:
        st.info("No movies with both budget and revenue data in the filtered selection.")
    
    # Top movies table
    st.subheader("🏆 Top Movies (Filtered)")
    
    display_columns = [
        'original_title', 'release_year', 'rating_original', 
        'vote_count_original', 'budget_original', 'revenue_original'
    ]
    
    top_movies = filtered_df.nlargest(20, 'rating_original')[display_columns].copy()
    top_movies.columns = ['Title', 'Year', 'Rating', 'Votes', 'Budget ($)', 'Revenue ($)']
    
    # Format the display
    top_movies['Rating'] = top_movies['Rating'].round(1)
    top_movies['Budget ($)'] = top_movies['Budget ($)'].apply(
        lambda x: f"${x:,.0f}" if x > 0 else "N/A"
    )
    top_movies['Revenue ($)'] = top_movies['Revenue ($)'].apply(
        lambda x: f"${x:,.0f}" if x > 0 else "N/A"
    )
    top_movies['Votes'] = top_movies['Votes'].apply(
        lambda x: f"{x:,}" if x > 0 else "N/A"
    )
    
    st.dataframe(top_movies, use_container_width=True, hide_index=True)

# =========================
# MAIN APP WITH TABS
# =========================
def main():
    st.markdown('<h1 class="main-header">🎬 Enhanced LSA Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Content-Based Filtering with Enhanced Latent Semantic Analysis")
    st.markdown("*Using Weighted TF-IDF + Hybrid Features + TruncatedSVD for superior semantic understanding*")

    recommender = load_enhanced_recommender()
    if recommender is None:
        st.error("❌ Enhanced models not found. Please run the enhanced modeling notebook first.")
        st.info("Run: `notebooks/03_modeling_enhanced.ipynb`")
        st.stop()

    # Enhanced Sidebar
    st.sidebar.header("🚀 Enhanced LSA Model Info")
    st.sidebar.metric("Total Movies", len(recommender.df))
    st.sidebar.metric("TF-IDF Features", recommender.tfidf_model.max_features)
    st.sidebar.metric("LSA Dimensions", recommender.lsa_model.n_components)
    st.sidebar.metric("Hybrid Features", recommender.hybrid_features.shape[1])
    
    # Calculate explained variance
    if hasattr(recommender.lsa_model, 'explained_variance_ratio_'):
        total_variance = recommender.lsa_model.explained_variance_ratio_.sum()
        st.sidebar.metric("Explained Variance", f"{total_variance:.1%}")
    
    st.sidebar.markdown("""
    **🔬 Enhanced LSA Pipeline**
    1. **Weighted Features**: Genres 4x, Keywords 3x, Directors 3x
    2. **Advanced TF-IDF**: N-grams + optimized parameters
    3. **Enhanced LSA**: 150 components for better semantics
    4. **Hybrid Similarity**: LSA (80%) + Numerical (20%)
    5. **Quality Filtering**: Intelligent recommendations
    
    **✨ Enhanced Benefits**
    - Superior semantic understanding
    - Weighted feature importance
    - Hybrid content + metadata
    - Quality-based filtering
    - Advanced similarity metrics
    """)

    # Main Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔍 Search Movie")

        typed_text = st.text_input(
            "Start typing a movie name",
            placeholder="e.g. Avatar, The Dark Knight, Inception"
        )

        selected_movie = None
        if typed_text:
            suggestions = get_movie_autocomplete(recommender, typed_text)
            if suggestions:
                selected_movie = st.selectbox("Select a movie", suggestions)
            else:
                st.info("No matching movies found")

    with col2:
        st.subheader("⚙️ Recommendation Settings")
        num_recommendations = st.slider("Number of recommendations", 1, 20, 10)
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 6.0, 0.5)
        show_charts = st.checkbox("Show visualization charts", True)
        show_details = st.checkbox("Show detailed movie info", True)

    # Recommendation Logic
    if selected_movie:
        with st.spinner("🔍 Finding similar movies using Enhanced LSA..."):
            recommendations = recommender.get_recommendations(
                selected_movie,
                n_recommendations=num_recommendations,
                min_rating=min_rating
            )

        if 'error' in recommendations:
            st.error(recommendations['error'])
            return

        st.success(f"✅ Found {len(recommendations['recommendations'])} recommendations for **{selected_movie}**")

        # Query movie info
        query_info = recommender.get_movie_info(selected_movie)
        if 'error' not in query_info:
            with st.expander("🎥 About the selected movie", expanded=True):
                display_movie_card(query_info)

        # Charts
        if show_charts and recommendations['recommendations']:
            st.header("📊 Recommendation Analytics")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.plotly_chart(
                    create_enhanced_similarity_chart(recommendations), 
                    use_container_width=True
                )
            
            with chart_col2:
                st.plotly_chart(
                    create_rating_distribution(recommendations),
                    use_container_width=True
                )

        # Recommendations
        st.header("🎯 Enhanced LSA Recommendations")
        
        if not recommendations['recommendations']:
            st.warning(f"No movies found with rating >= {min_rating}. Try lowering the minimum rating.")
        else:
            for i, movie in enumerate(recommendations['recommendations'], 1):
                if show_details:
                    display_movie_card(movie, rank=i)
                else:
                    # Compact view
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{i}. {movie['title']}** ({movie['year']})")
                        st.caption(f"Genres: {movie['genres']}")
                    with col2:
                        st.metric("Rating", f"{movie['rating']}/10")
                    with col3:
                        st.metric("Similarity", f"{movie['similarity_score']:.1%}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#777'>"
        "🚀 Enhanced LSA Movie Recommendation System | "
        "Streamlit · Scikit-learn · Enhanced TruncatedSVD · Hybrid Features · TMDB API"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
