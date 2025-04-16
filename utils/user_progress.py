# utils/user_progress.py

import json
import os
import datetime
from collections import defaultdict

# Store user data in a JSON file for simplicity
# In a production app, this would use a database
USER_DATA_FILE = 'user_data.json'

def _load_user_data():
    """Load all user data from storage"""
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def _save_user_data(data):
    """Save user data to storage"""
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def update_user_progress(user_id, quiz_results):
    """
    Update user progress based on quiz results
    
    Parameters:
    user_id (str): User identifier
    quiz_results (dict): Results from quiz evaluation
    """
    user_data = _load_user_data()
    
    # Initialize user data if not exists
    if user_id not in user_data:
        user_data[user_id] = {
            'quizzes_taken': 0,
            'words_learned': 0,
            'current_level': 'easy',
            'quiz_history': [],
            'learning_sessions': [],
            'weak_areas': defaultdict(int),
            'words_practiced': set(),
            'last_active': None
        }
    
    # Update user stats
    user_data[user_id]['quizzes_taken'] += 1
    user_data[user_id]['current_level'] = quiz_results['next_difficulty']
    user_data[user_id]['last_active'] = datetime.datetime.now().isoformat()
    
    # Add quiz result to history
    quiz_history_entry = {
        'date': datetime.datetime.now().isoformat(),
        'score_percentage': quiz_results['score_percentage'],
        'difficulty': quiz_results.get('difficulty', 'easy'),
        'correct_answers': quiz_results['correct_answers'],
        'total_questions': quiz_results['total_questions']
    }
    user_data[user_id]['quiz_history'].append(quiz_history_entry)
    
    # Update weak areas
    for area in quiz_results.get('weak_areas', []):
        user_data[user_id]['weak_areas'][area] += 1
    
    # Convert defaultdict to dict for JSON serialization
    user_data[user_id]['weak_areas'] = dict(user_data[user_id]['weak_areas'])
    
    # Convert set to list for JSON serialization
    user_data[user_id]['words_practiced'] = list(user_data[user_id]['words_practiced'])
    
    _save_user_data(user_data)
    return True

def log_learning_session(user_id, words_practiced, duration_seconds):
    """
    Log a learning session
    
    Parameters:
    user_id (str): User identifier
    words_practiced (list): Words practiced in this session
    duration_seconds (int): Duration of session in seconds
    """
    user_data = _load_user_data()
    
    # Initialize user data if not exists
    if user_id not in user_data:
        user_data[user_id] = {
            'quizzes_taken': 0,
            'words_learned': 0,
            'current_level': 'easy',
            'quiz_history': [],
            'learning_sessions': [],
            'weak_areas': {},
            'words_practiced': [],
            'last_active': None
        }
    
    # Add learning session
    session = {
        'date': datetime.datetime.now().isoformat(),
        'words_practiced': words_practiced,
        'duration_seconds': duration_seconds
    }
    user_data[user_id]['learning_sessions'].append(session)
    
    # Update words practiced
    practiced_words_set = set(user_data[user_id]['words_practiced'])
    practiced_words_set.update(words_practiced)
    user_data[user_id]['words_practiced'] = list(practiced_words_set)
    
    # Update words learned count
    user_data[user_id]['words_learned'] = len(practiced_words_set)
    
    # Update last active timestamp
    user_data[user_id]['last_active'] = datetime.datetime.now().isoformat()
    
    _save_user_data(user_data)
    return True

def get_user_stats(user_id):
    """
    Get user statistics
    
    Parameters:
    user_id (str): User identifier
    
    Returns:
    dict: User statistics
    """
    user_data = _load_user_data()
    
    # Return empty stats if user doesn't exist
    if user_id not in user_data:
        return {
            'quizzes_taken': 0,
            'words_learned': 0,
            'current_level': 'easy',
            'quiz_history': [],
            'learning_streak': 0,
            'weak_areas': [],
            'progress_percentage': 0,
            'recent_activity': []
        }
    
    # Calculate learning streak
    streak = 0
    if user_data[user_id]['quiz_history']:
        # Sort quiz history by date
        sorted_history = sorted(
            user_data[user_id]['quiz_history'], 
            key=lambda x: x['date'], 
            reverse=True
        )
        
        current_date = datetime.datetime.now().date()
        last_quiz_date = datetime.datetime.fromisoformat(sorted_history[0]['date']).date()
        
        # Check if user took quiz today
        if last_quiz_date == current_date:
            streak = 1
            # Check consecutive days backwards
            prev_date = current_date - datetime.timedelta(days=1)
            for quiz in sorted_history[1:]:
                quiz_date = datetime.datetime.fromisoformat(quiz['date']).date()
                if quiz_date == prev_date:
                    streak += 1
                    prev_date -= datetime.timedelta(days=1)
                elif quiz_date < prev_date:
                    # Skip ahead if we missed this date
                    prev_date = quiz_date - datetime.timedelta(days=1)
                else:
                    # Not consecutive
                    break
    
    # Calculate progress percentage (simple version based on level and quiz history)
    progress_percentage = 0
    if user_data[user_id]['current_level'] == 'easy':
        # Base progress on quiz scores at easy level
        easy_quizzes = [q for q in user_data[user_id]['quiz_history'] 
                      if q.get('difficulty') == 'easy']
        if easy_quizzes:
            avg_score = sum(q['score_percentage'] for q in easy_quizzes) / len(easy_quizzes)
            progress_percentage = min(33, (avg_score / 100) * 33)
    elif user_data[user_id]['current_level'] == 'medium':
        # Base progress on quiz scores at medium level plus 33%
        medium_quizzes = [q for q in user_data[user_id]['quiz_history'] 
                        if q.get('difficulty') == 'medium']
        if medium_quizzes:
            avg_score = sum(q['score_percentage'] for q in medium_quizzes) / len(medium_quizzes)
            progress_percentage = 33 + min(33, (avg_score / 100) * 33)
        else:
            progress_percentage = 33
    elif user_data[user_id]['current_level'] == 'hard':
        # Base progress on quiz scores at hard level plus 66%
        hard_quizzes = [q for q in user_data[user_id]['quiz_history'] 
                       if q.get('difficulty') == 'hard']
        if hard_quizzes:
            avg_score = sum(q['score_percentage'] for q in hard_quizzes) / len(hard_quizzes)
            progress_percentage = 66 + min(34, (avg_score / 100) * 34)
        else:
            progress_percentage = 66
    
    # Extract weak areas
    weak_areas = sorted(
        user_data[user_id]['weak_areas'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:3]
    
    # Get recent activity (combine quiz history and learning sessions)
    recent_activity = []
    
    for quiz in user_data[user_id]['quiz_history'][-5:]:  # Last 5 quizzes
        recent_activity.append({
            'type': 'quiz',
            'date': quiz['date'],
            'details': f"Score: {quiz['score_percentage']:.1f}%, Difficulty: {quiz.get('difficulty', 'easy').title()}"
        })
    
    for session in user_data[user_id]['learning_sessions'][-5:]:  # Last 5 learning sessions
        recent_activity.append({
            'type': 'learning',
            'date': session['date'],
            'details': f"Practiced {len(session['words_practiced'])} words for {session['duration_seconds'] // 60} minutes"
        })
    
    # Sort combined activity by date
    recent_activity.sort(key=lambda x: x['date'], reverse=True)
    recent_activity = recent_activity[:5]  # Keep only 5 most recent
    
    return {
        'quizzes_taken': user_data[user_id]['quizzes_taken'],
        'words_learned': user_data[user_id]['words_learned'],
        'current_level': user_data[user_id]['current_level'],
        'quiz_history': user_data[user_id]['quiz_history'][-10:],  # Last 10 quizzes
        'learning_streak': streak,
        'weak_areas': [{'area': area, 'count': count} for area, count in weak_areas],
        'progress_percentage': round(progress_percentage, 1),
        'recent_activity': recent_activity
    }