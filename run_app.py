import subprocess
import sys
import os

def run_streamlit_app():
    """Run the Streamlit app"""
    print("Starting Enhanced Movie Recommender App...")
    print("App will open in your default browser")
    print("URL: http://localhost:8501")
    print("\n" + "="*50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/movie_recommender_app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\nApp stopped by user")
    except Exception as e:
        print(f"Error running app: {e}")

if __name__ == "__main__":
    run_streamlit_app()