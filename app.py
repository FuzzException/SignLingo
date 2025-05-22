import datetime
from flask import Flask, render_template, request, session, jsonify
from utils.isl_converter import convert_to_isl
from utils.quiz_generator import generate_quiz, evaluate_quiz
from utils.user_progress import update_user_progress, get_user_stats
from sentence_transformers import SentenceTransformer
from recommender import main
from recommender2 import main
from recommender2 import calculate_similarities, recommend_words, auto_categorize_words
import pandas as pd
import sqlite3
import os
os.environ['MPLBACKEND'] = 'Agg'  # Force matplotlib to use non-interactive backend
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from datetime import date, timedelta
from flask import send_from_directory
from flask_cors import CORS
import nltk
import numpy as np
nltk.download('wordnet')
nltk.download('punkt')

# Your app initialization
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.urandom(24)  # For session management
CORS(app)
if 'FLASK_APP' in os.environ:
    try:
        # Only import Tkinter if needed and capture/discard any errors
        import tkinter as tk
        # Create a root window that will manage Tkinter's internal state
        root = tk.Tk()
        root.withdraw()  # Hide the window
        # NOTE: This root should persist for the lifetime of the application
    except ImportError:
        print("Tkinter not available, using alternative methods")
    except Exception as e:
        print(f"Could not initialize Tkinter: {e}")

# Then modify your model loading code to handle potential Tkinter issues:

# Wrap your SentenceTransformer initialization in a try/except
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
except RuntimeError as e:
    if "main thread is not in main loop" in str(e):
        print("Warning: Tkinter threading issue detected. Consider running in a separate process.")
        # You might need a fallback or alternative approach here
    raise
# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = "supersecretkey"  # Change this for security
Session(app)
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    gender = db.Column(db.String(50), nullable=False)
    #date = db.Column(db.String(20), nullable=False,default=lambda: datetime.now().strftime("%Y-%m-%d"))
    progress_level = db.Column(db.Integer, default=1,nullable=False)
    noquiz=db.Column(db.Integer, default=1,nullable=False)
    alphaquiz=db.Column(db.Integer, default=1,nullable=False)
    twoquiz=db.Column(db.Integer, default=1,nullable=False)
    threequiz=db.Column(db.Integer, default=1,nullable=False)
    date = db.Column(db.Date,default=None)  # Track last activity
    quiz=db.Column(db.Integer, default=0,nullable=False)
    current_streak = db.Column(db.Integer, default=0)

# Create database tables (Run once)
with app.app_context():
    db.create_all()  # Recreate tables with the new structure
    print("Database tables recreated!")

df = pd.read_csv(r"/Users/sarahgteerthan/Desktop/Finalproject-main/Final_dataset.csv")
# Initialize sample data and recommender
sample_data=pd.read_csv("filtered_categories1.csv")
data=pd.read_csv("learning.csv")
#data=pd.read_csv("alphabet.csv")
recommender = main(sample_data)
recommender2 = main(data)

# Logout
@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    return jsonify({"message": "Logged out successfully"})

# Get current user
@app.route("/get_current_user", methods=["GET"])
def get_current_user():
    if "user_id" in session:
        user = User.query.get(session["user_id"])
        if user:
            return jsonify({"user_id": user.id,"username": user.username, "gender":user.gender,"date":user.date,"progress_level": user.progress_level,"noquiz":user.noquiz,"alphaquiz":user.alphaquiz})
    return jsonify({"message": "Not logged in"}), 401


@app.route('/update_progress', methods=['POST'])
def update_progress():
    data = request.json
    print("Received update data:", data)
    user_id = session.get("user_id")
    print("User ID from session:", user_id)
    
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401
    
    user = User.query.filter_by(id=user_id).first()
    print("Found user:", user is not None)
    
    if user:
        # Convert string date to Date object if needed
        if isinstance(data.get("date"), str):
            try:
                user.date = datetime.datetime.strptime(data["date"], "%Y-%m-%d").date()
            except:
                user.date = datetime.datetime.now().date()
        else:
            user.date = data.get("date") or datetime.datetime.now().date()
            
        user.progress_level = data.get("progress_level", user.progress_level)
        user.alphaquiz = data.get("alphaquiz", user.alphaquiz)
        user.noquiz = data.get("noquiz", user.noquiz)
        user.twoquiz = data.get("twoquiz", user.twoquiz)
        user.threequiz = data.get("threequiz", user.threequiz)
        user.quiz=data.get("quiz",user.quiz)
        
        try:
            db.session.commit()
            print("Database updated successfully")
            return jsonify({"message": "Progress updated successfully"})
        except Exception as e:
            print("Database error:", str(e))
            db.session.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
    
    return jsonify({"error": "User not found"}), 404

# Get user progress

@app.route('/progress', methods=['GET'])
def get_progress():
    user_id = session.get("user_id")
    
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401
    
    user = User.query.filter_by(id=user_id).first()
    
    if user:
        return jsonify({
            "date": user.date,
            "progress_level": user.progress_level,
            "quiz":user.quiz,
            "noquiz": user.noquiz,
            "alphaquiz":user.alphaquiz,
            "twoquiz":user.twoquiz,
            "threequiz":user.threequiz,
            "quizzes_taken": user.noquiz,
            "words_learned": user.progress_level * 20,  # Example calculation
            "learning_streak": user.current_streak,  # Replace with real streak logic
            "current_level": "Beginner" if user.progress_level == 1 else "Intermediate" if user.progress_level == 2 else "Advanced",
            "progress_percentage": user.progress_level * 10,  # Example calculation
        })
    
    return jsonify({"error": "User not found"}), 404

# Basic routes
@app.route('/')
def web():
    return render_template('web.html')

@app.route('/dashboard.html')
def dashboard():
    return render_template('dashboard.html')

# Update the signup function to return a redirect instead of a JSON response
@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.json
        print("Received signup data:", data)  # Debug print
        
        username = data.get("username")
        gender = data.get("gender")

        if not username or not gender:
            return jsonify({"message": "Username and gender are required", "redirect": False}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({"message": "Username already taken", "redirect": False}), 400

        new_user = User(username=username, gender=gender)
        db.session.add(new_user)
        db.session.commit()
        
        # Get the new user's ID
        user_id = new_user.id
        session["user_id"] = user_id  # Store in session
        
        # Return both success message and redirect URL
        return jsonify({
            "message": "Signup successful!", 
            "user_id": user_id,
            "redirect": True,
            "redirect_url": "/"  # Redirect to route for web.html
        })
    except Exception as e:
        print("Signup error:", str(e))
        db.session.rollback()
        return jsonify({"message": "Error during signup: " + str(e), "redirect": False}), 500

# Update the login function similarly
@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.json
        username = data.get("username")
        gender = data.get("gender")

        user = User.query.filter_by(username=username, gender=gender).first()
        if user:
            session["user_id"] = user.id
            return jsonify({
                "message": "Login successful", 
                "user_id": user.id,
                "redirect": True,
                "redirect_url": "/"  # Redirect to route for web.html
            })
        return jsonify({"message": "Invalid credentials", "redirect": False}), 401
    except Exception as e:
        print("Login error:", str(e))
        return jsonify({"message": "Error during login: " + str(e), "redirect": False}), 500

# Function to connect to the database


def convert_to_isl_structure(paragraph):
    # Split the paragraph into sentences
    import re
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag

    # Ensure necessary NLTK data is downloaded
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    isl_sentences = []
    
    for sentence in sentences:
        
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        
        
        verb_index = next((i for i, word_pos in enumerate(pos_tags) if word_pos[1].startswith('VB')), None)
        
        if verb_index is not None:
            # Identify subject, verb, and object
            subject = " ".join(words[:verb_index])  
            verb = words[verb_index]  
            object_words = words[verb_index + 1:]
            
            
            object_phrase = " ".join(object_words)
            isl_sentence = f"{subject.strip()} {object_phrase.strip()} {verb.strip()}".strip()
            
            
            isl_sentence = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', isl_sentence)
            isl_sentences.append(isl_sentence)
        else:
            
            isl_sentences.append(sentence.strip())
    
    # Join the ISL sentences back into a paragraph
    return " ".join(isl_sentences)

def preprocess_text(text):
    import string
    from nltk.tokenize import word_tokenize
    from nltk.stem.wordnet import WordNetLemmatizer
    lemma = WordNetLemmatizer()
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text)
    lemmatized_words = [lemma.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)

def serve_static(filename):
    return send_from_directory('static', filename)

# Update the route for the dashboard
@app.route('/roadmap.html')
def roadmap():
    return render_template('roadmap.html')

@app.route('/temp.html')
def temp():
    return render_template('temp.html')

@app.route('/temp2.html')
def temp2():
    return render_template('temp2.html')

@app.route('/temp3.html')
def temp3():
    return render_template('temp3.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/index2.html')
def index2():
    return render_template('index2.html')

@app.route('/sign.html')
def sign():
    return render_template('sign.html')

@app.route('/game.html')
def game():
    return render_template('game.html')

@app.route('/quiz1.html')
def quiz1():
    return render_template('quiz1.html')

@app.route('/word1.html')
def word():
    return render_template('word1.html')

@app.route('/number.html')
def number():
    return render_template('number.html')


@app.route('/static/videos/<filename>')
def get_video(filename):
    return send_from_directory('static/videos', filename, as_attachment=False)
VIDEO_FOLDER = "static/videos"

@app.route('/static/assets1/<filename>')
def get_word(filename):
    return send_from_directory('static/assets1', filename, as_attachment=False)
WORD_FOLDER = "static/assets1"

@app.route('/get_video_list')
def get_video_list():
    try:
        video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
        return jsonify(video_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/get_word_list')
def get_word_list():
    try:
        video_files = [f for f in os.listdir(WORD_FOLDER) if f.endswith('.jpg')]
        return jsonify(video_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/progress.html')
def progress():
    if 'user_id' not in session:
        # For demo, create a temporary user ID
        session['user_id'] = 'temp_user_' + os.urandom(8).hex()
        
    user_stats = get_user_stats(session['user_id'])
    return render_template("progress.html", stats=user_stats)

def update_user_streak(user):
    today = date.today()

    if user.date:
        # Check if the last activity was yesterday
        if user.date == today - timedelta(days=1):
            user.current_streak += 1  # Continue the streak
        # Check if the last activity was today
        elif user.date == today:
            pass  # Do nothing; streak remains unchanged
        else:
            user.current_streak = 1  # Reset the streak (new streak starts today)
    else:
        user.current_streak = 1  # First activity; initialize the streak

    # Update the last activity date
    user.date = today
    db.session.commit()

import traceback  # Import traceback to capture detailed errors
import pandas as pd
model = SentenceTransformer('all-MiniLM-L6-v2')
data['word'] = data['word'].str.lower().str.strip()
word_embeddings = {word: model.encode([word])[0] for word in data['word'].unique()}
@app.route('/learn', methods=['POST'])
def learn():
    word = request.form.get('word', '').lower().strip()

    if not word:
        return jsonify({'error': 'Please enter a word'})

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        word_embeddings = {word: model.encode([word])[0] for word in data['word'].unique()}
        similarity_df = calculate_similarities(word_embeddings)
        df_categorized, word_to_cluster = auto_categorize_words(data, word_embeddings)
        
        recommendations = recommend_words(word, data, similarity_df, model, word_embeddings, top_n=5)

        # Debugging: Print type and content
        print("Type of recommendations:", type(recommendations))
        print("Recommendations content:", recommendations)

        # Ensure recommendations is a DataFrame
        if isinstance(recommendations, list):
            recommendations = pd.DataFrame(recommendations)

        # Convert NumPy types to standard Python types
        results = recommendations.map(lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x)
        print("Final JSON Response:", results.to_dict(orient='records'))
        return jsonify({'recommendations': results.to_dict('records')})

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {error_trace}")
        return jsonify({'error': f'Error getting recommendations: {str(e)}', 'trace': error_trace})

sample_data['word'] = sample_data['word'].str.lower().str.strip()
word_embedding = {word: model.encode([word])[0] for word in sample_data['word'].unique()}
@app.route('/search', methods=['POST'])
def search():
    word = request.form.get('word', '').lower().strip()

    if not word:
        return jsonify({'error': 'Please enter a word'})

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        word_embeddings = {word: model.encode([word])[0] for word in sample_data['word'].unique()}
        similarity_df = calculate_similarities(word_embedding)
        df_categorized, word_to_cluster = auto_categorize_words(sample_data, word_embedding)
        
        recommendations = recommend_words(word, sample_data, similarity_df, model, word_embedding, top_n=5)

        # Debugging: Print type and content
        print("Type of recommendations:", type(recommendations))
        print("Recommendations content:", recommendations)

        # Ensure recommendations is a DataFrame
        if isinstance(recommendations, list):
            recommendations = pd.DataFrame(recommendations)

        # Convert NumPy types to standard Python types
        results = recommendations.map(lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x)
        print("Final JSON Response:", results.to_dict(orient='records'))
        return jsonify({'recommendations': results.to_dict('records')})

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {error_trace}")
        return jsonify({'error': f'Error getting recommendations: {str(e)}', 'trace': error_trace})

@app.route('/get_videos', methods=['POST'])
def get_videos():
    # Use predefined paragraph
    data = request.get_json()
    user_paragraph = data.get('paragraph', '')

    isl_paragraph = convert_to_isl_structure(user_paragraph)
    cleaned_summarized_text = preprocess_text(isl_paragraph)
    # Find matching words and their corresponding URLs
    words = cleaned_summarized_text.split()
    video_urls = []
    for word in words:
        match = df[df['word'].str.lower() == word.lower()]
        if not match.empty:
            video_urls.append(match.iloc[0]['url'])
        else:
            continue    

    return jsonify(video_urls=video_urls)

@app.route('/index3.html')
def index3():
    return render_template('index3.html')
@app.route("/quiz3.html")
def quiz3():
    if 'current_quiz' not in session:
        session['current_quiz'] = generate_quiz(difficulty='easy')

    quiz = session.get('current_quiz', {})
    return render_template("quiz3.html", quiz=quiz)

@app.route('/convert', methods=['POST'])
def convert():
    try:
        data = request.get_json()
        sentence = data.get("sentence", "").strip()

        if not sentence:
            return jsonify({"error": "No sentence provided"}), 400

        # Get ISL translation
        isl_sentence, words = convert_to_isl(sentence)

        # Retrieve video URLs for words
        words_with_videos = []
        for word in words:
            video_url = df.loc[df['word'].str.lower() == word.lower(), 'url'].values
            if len(video_url) > 0:
                words_with_videos.append({"word": word, "hasVideo": True, "url": video_url[0]})
            else:
                words_with_videos.append({"word": word, "hasVideo": False})

        response = {
            "isl_sentence": isl_sentence,
            "words_with_videos": words_with_videos
        }

        print("Backend response:", response)  # Debugging
        return jsonify(response)

    except Exception as e:
        print("Error in /convert:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/quiz")
def quiz():
    if 'current_quiz' not in session:
        session['current_quiz'] = generate_quiz(difficulty='easy')

    # Ensure current_quiz is correctly structured
    current_quiz = session.get('current_quiz')
    if not current_quiz:
        current_quiz = generate_quiz(difficulty='easy')
        session['current_quiz'] = current_quiz

    # Debugging information
    print("Current Quiz:", current_quiz)

    return render_template("quiz3.html", quiz=current_quiz)

@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    data = request.get_json()
    user_answers = data.get("answers", {})

    try:
        # Debug: Check session data
        print("Current Quiz in session:", session.get('current_quiz'), type(session.get('current_quiz')))

        # Evaluate quiz
        results = evaluate_quiz(session['current_quiz'], user_answers)

        # Debug: Results check
        print("Evaluation Results:", results)

        # Update user progress
        if 'user_id' in session:
            update_user_progress(session['user_id'], results)

        # Determine focus areas for the next quiz
        focus_areas = None
        if results['weak_areas']:
            focus_areas = results['weak_areas']

        # Generate a new quiz
        session['current_quiz'] = generate_quiz(
            difficulty=results['next_difficulty'],
            focus_areas=focus_areas
        )

        # Debug: Check new quiz
        print("New Generated Quiz:", session['current_quiz'], type(session['current_quiz']))

        return jsonify(results)

    except Exception as e:
        print("Error in submit_quiz:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/play_video/<word>")
def get_video_url(word):
    word = word.strip().lower()
    # Convert word column to lowercase for case-insensitive matching
    df['word'] = df['word'].str.strip().str.lower()
    # Search for the word in the DataFrame
    match = df[df['word'] == word]
    
    if not match.empty:
        video_url = match.iloc[0]['url']  # Get the first matched URL
        return jsonify({"word": word, "url": video_url})
    else:
        return jsonify({"error": "Video not found for this word."}), 404

def get_available_videos():
    # Get the list of available video files (without extension)
    video_dir = os.path.join(app.static_folder, "assets", "videos")
    if os.path.exists(video_dir):
        return [os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith('.mp4')]
    return []
@app.route("/clear_session")
def clear_session():
    session.clear()
    return "Session cleared!"

# Add this at the end of your file
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8008, debug=True)