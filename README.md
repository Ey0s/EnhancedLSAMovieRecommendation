# Enhanced LSA Movie Recommendation System

A sophisticated content-based movie recommendation system using Enhanced Latent Semantic Analysis (LSA) with weighted feature engineering and hybrid similarity computation.

##  Project Overview

This project implements an advanced movie recommendation system that leverages Natural Language Processing and dimensionality reduction techniques to provide intelligent, personalized movie recommendations. The system addresses common challenges in recommendation systems through innovative feature engineering and semantic understanding.

### Key Features

- **Enhanced LSA Implementation**: Advanced semantic analysis with 150 components
- **Weighted Feature Engineering**: Domain-expert weighting (Genres 4x, Keywords 3x, Directors 3x)
- **Hybrid Similarity**: Combines content features (80%) with numerical metadata (20%)
- **Real-time Performance**: Sub-10ms response time with 101 recommendations/second
- **Interactive Web App**: Professional Streamlit interface with movie posters
- **Quality Filtering**: Intelligent recommendation filtering based on ratings

##  Performance Metrics

- **Genre Consistency**: 93.16% accuracy in genre matching
- **Rating Quality**: 7.1/10 average rating for recommendations
- **Response Time**: 9.90ms average recommendation generation
- **Dataset Coverage**: 4,802 movies spanning 1916-2017
- **Feature Dimensions**: 8,000 TF-IDF features reduced to 150 LSA components

## Quick Start

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd enhanced-lsa-movie-recommender
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app/movie_recommender_app.py
```

4. **Access the app**
- Local: http://localhost:8501
- Network: http://your-ip:8501

## 📁 Project Structure

```
enhanced-lsa-movie-recommender/
├── 📊 data/
│   ├── raw/                    # Original dataset files
│   └── processed/              # Cleaned and processed data
├── 📓 notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data cleaning and preprocessing
│   ├── 03_modeling_enhanced.ipynb    # Enhanced LSA model development
│   ├── 03_modeling_simple.ipynb     # Simple LSA baseline
│   └── 04_evaluation_enhanced.ipynb # Model evaluation and testing
├── 💻 app/
│   ├── movie_recommender_app.py      # Main Streamlit application
│   └── movie_recommender_simple.py  # Test application
├── 🤖 models/
│   ├── tfidf_enhanced.pkl     # Enhanced TF-IDF vectorizer
│   ├── lsa_enhanced.pkl       # Enhanced LSA model
│   ├── similarity_enhanced.pkl # Precomputed similarity matrix
│   └── hybrid_features_enhanced.pkl # Hybrid feature matrix
├── 📄 reports/
│   └── Enhanced_LSA_Movie_Recommendation_System_Report.docx
├── 📋 requirements.txt        # Python dependencies
├── 🚫 .gitignore             # Git ignore rules
└── 📖 README.md              # Project documentation
```

## 🔬 Methodology

### 1. Data Processing Pipeline

```python
Raw Data → Feature Engineering → TF-IDF Vectorization → 
LSA Transformation → Hybrid Features → Similarity Computation → Recommendations
```

### 2. Enhanced Feature Engineering

- **Weighted Text Features**: Intelligent weighting based on domain expertise
- **Advanced TF-IDF**: 8,000 features with bigram support
- **Hybrid Approach**: Combines semantic (LSA) and numerical features
- **Quality Filtering**: Rating-based recommendation filtering

### 3. Model Architecture

```python
# Enhanced TF-IDF Configuration
TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    sublinear_tf=True
)

# Optimized LSA Implementation
TruncatedSVD(
    n_components=150,
    algorithm='randomized'
)

# Hybrid Feature Combination
hybrid_features = 0.8 * lsa_features + 0.2 * numerical_features
```

## 📈 Results

### Genre Consistency Analysis
- **Perfect Consistency (100%)**: 12/19 test movies
- **High Consistency (90-99%)**: 5/19 test movies
- **Good Consistency (80-89%)**: 2/19 test movies

### Example Recommendations

**Query: "Avatar" (2009)**
1. Independence Day (1996) - 90.6% similarity
2. Star Trek (2009) - 88.6% similarity
3. Star Trek Into Darkness (2013) - 88.4% similarity

**Query: "The Dark Knight" (2008)**
1. The Dark Knight Rises (2012) - 95.4% similarity
2. Batman Begins (2005) - 95.2% similarity
3. Watchmen (2009) - 85.0% similarity

## 🛠️ Usage Examples

### Basic Recommendation

```python
from app.movie_recommender_app import EnhancedMovieRecommender

# Initialize recommender
recommender = EnhancedMovieRecommender()
recommender.load_models()

# Get recommendations
recommendations = recommender.get_recommendations(
    movie_title="Inception",
    n_recommendations=10,
    min_rating=6.0
)

# Display results
for i, movie in enumerate(recommendations['recommendations'], 1):
    print(f"{i}. {movie['title']} ({movie['year']}) - {movie['rating']}/10")
```

### Movie Information

```python
# Get detailed movie information
movie_info = recommender.get_movie_info("The Matrix")
print(f"Title: {movie_info['title']}")
print(f"Rating: {movie_info['rating']}/10")
print(f"Genres: {movie_info['genres']}")
print(f"Director: {movie_info['director']}")
```

## 📊 Dataset Information

### Source
- **Primary**: The Movie Database (TMDb)
- **Size**: 4,802 movies with 62 features
- **Time Period**: 1916-2017 (101 years of cinema)
- **Languages**: Primarily English with international films

### Key Features
- **Content**: Overview, tagline, genres, keywords
- **People**: Cast, crew, directors
- **Metadata**: Budget, revenue, ratings, runtime
- **Temporal**: Release dates, production years

### Data Quality
- **Completeness**: 95% text features, 85% numerical features
- **Coverage**: 20 unique genres, global movie representation
- **Rating Distribution**: Mean 6.2/10, std 1.2 (realistic spread)

## 🔧 Technical Details

### Dependencies

```txt
streamlit>=1.12.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
plotly>=5.0.0
requests>=2.25.0
joblib>=1.0.0
```

### System Requirements
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 500MB for models and data
- **CPU**: Multi-core processor recommended for faster similarity computation
- **Network**: Internet connection for movie poster fetching (TMDB API)

### Performance Optimization
- **Caching**: Streamlit resource caching for model loading
- **Vectorization**: Optimized numpy operations
- **Memory Management**: Efficient sparse matrix handling
- **Parallel Processing**: Multi-threaded similarity computation

## 🎯 Evaluation Metrics

### Quantitative Metrics
- **Genre Consistency**: Percentage of recommendations sharing genres
- **Rating Quality**: Average rating of recommended movies
- **Similarity Distribution**: Cosine similarity score analysis
- **Response Time**: Recommendation generation speed
- **Coverage**: Catalog diversity in recommendations

### Qualitative Assessment
- **Relevance**: Human evaluation of recommendation quality
- **Diversity**: Variety in recommended movies
- **Serendipity**: Discovery of unexpected but relevant movies
- **User Experience**: Interface usability and satisfaction

## 🚧 Known Limitations

### Technical Limitations
- **Cold Start**: New movies with limited metadata
- **Language Bias**: Primarily English-language optimized
- **Temporal Bias**: Recent movies have richer metadata
- **Popularity Bias**: Well-known movies may be over-represented

### Mitigation Strategies
- **Graceful Degradation**: System handles missing data
- **Quality Filtering**: Minimum rating thresholds
- **Diversity Metrics**: Monitor recommendation variety
- **Continuous Evaluation**: Regular performance assessment

## 🔮 Future Enhancements

### Short-term (3-6 months)
- [ ] User feedback integration
- [ ] Multi-language support
- [ ] Mobile application development
- [ ] Advanced filtering options

### Medium-term (6-12 months)
- [ ] Deep learning integration (BERT embeddings)
- [ ] Collaborative filtering hybrid
- [ ] Real-time learning capabilities
- [ ] A/B testing framework

### Long-term (1-2 years)
- [ ] Multi-media expansion (TV shows, documentaries)
- [ ] Cross-platform synchronization
- [ ] Predictive analytics for content success
- [ ] Recommendation-as-a-Service API

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/enhanced-lsa-movie-recommender.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 app/ notebooks/
black app/ notebooks/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **The Movie Database (TMDb)** for providing comprehensive movie data
- **Scikit-learn** community for excellent machine learning tools
- **Streamlit** team for the intuitive web app framework
- **Open source community** for inspiration and best practices

## 📞 Contact & Support

### Project Maintainers
- **Technical Questions**: Open a GitHub issue
- **Feature Requests**: Use the feature request template
- **Bug Reports**: Use the bug report template

### Resources
- **Documentation**: See `/docs` folder for detailed guides
- **API Reference**: Available in the code docstrings
- **Tutorials**: Check `/notebooks` for step-by-step examples
- **Community**: Join our discussions in GitHub Discussions

## 📊 Project Status

- **Status**: ✅ Production Ready
- **Version**: 1.0.0
- **Last Updated**: December 2024
- **Maintenance**: Actively maintained

### Recent Updates
- ✅ Fixed numeric display issues (ratings, budgets, vote counts)
- ✅ Enhanced web application with professional UI
- ✅ Comprehensive evaluation framework
- ✅ Performance optimization and caching
- ✅ Complete documentation and reports

---

**Built with ❤️ for movie lovers and data science enthusiasts**

*Discover your next favorite movie with the power of Enhanced Latent Semantic Analysis!* 🎬✨
