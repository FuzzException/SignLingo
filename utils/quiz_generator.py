import random
from utils.isl_converter import convert_to_isl

# Sample English sentences for quizzes
SAMPLE_SENTENCES = {
    'easy': [
        "My name is John",
        "I am happy",
        "She is a teacher",
        "He went to school",
        "Keep a safe distance",
        "I will call you tomorrow",
        "They work from home",
        "Can you help me",
        "Do not touch this",
        "The class is over"
    ],
    'medium': [
        "Yesterday I watched a beautiful movie",
        "My brother will graduate from college next month",
        "Can you tell me where the bathroom is",
        "I have been studying sign language for two years",
        "She doesn't want to go to the party tonight",
        "The red car belongs to my father",
        "We should finish our homework before playing",
        "Please sign your name on the dotted line",
        "How many languages do you speak fluently",
        "The children are playing in the garden"
    ],
    'hard': [
        "If it rains tomorrow, the outdoor event will be canceled",
        "After finishing her degree, she plans to work abroad for experience",
        "Can you explain why this algorithm is more efficient than the previous one",
        "Despite the challenges, they managed to complete the project on time",
        "The professor who taught us last semester has published a new book",
        "You should submit your application before the deadline expires next week",
        "When was the last time you visited your grandparents in their village",
        "The company announced that they will be hiring twenty new employees soon",
        "Please make sure that all electronic devices are turned off during the exam",
        "How many hours do you typically spend on learning new skills every day"
    ]
}

def generate_quiz(difficulty='easy', num_questions=5, focus_areas=None):
    """
    Generate a quiz with questions, optionally focusing on specific areas.
    """
    if difficulty not in SAMPLE_SENTENCES:
        difficulty = 'easy'

    selected_sentences = random.sample(SAMPLE_SENTENCES[difficulty], min(num_questions, len(SAMPLE_SENTENCES[difficulty])))

    quiz = {
        'difficulty': difficulty,
        'questions': []
    }
    
    question_types = ['reordering', 'multiple_choice', 'fill_blanks']
    if focus_areas:
        question_types = [q_type for q_type in question_types if q_type in focus_areas]

    for i, sentence in enumerate(selected_sentences):
        isl_sentence, isl_words = convert_to_isl(sentence)
        question_type = question_types[i % len(question_types)]
        
        if question_type == 'reordering':
            shuffled_words = random.sample(isl_words, len(isl_words))
            question = {
                'id': f'q{i+1}',
                'type': 'reordering',
                'english': sentence,
                'shuffled_words': shuffled_words,
                'correct_order': isl_words
            }
        elif question_type == 'multiple_choice':
            # Same logic for multiple-choice question
            options = []
            options.append(isl_sentence)
            for _ in range(3):
                wrong_order = isl_words.copy()
                random.shuffle(wrong_order)
                options.append(" ".join(wrong_order))
            random.shuffle(options)
            question = {
                'id': f'q{i+1}',
                'type': 'multiple_choice',
                'english': sentence,
                'question': 'Which is the correct ISL structure for this sentence?',
                'options': options,
                'correct_answer': options.index(isl_sentence)
            }
        elif question_type == 'fill_blanks':
            blanked_words = isl_words.copy()
            num_blanks = 2 if len(blanked_words) >= 4 else 1
            blank_positions = random.sample(range(len(blanked_words)), num_blanks)
            blanks = {}
            for pos in blank_positions:
                blanks[pos] = str(blanked_words[pos])
                blanked_words[pos] = '_____'
            question = {
                'id': f'q{i+1}',
                'type': 'fill_blanks',
                'english': sentence,
                'sentence_with_blanks': " ".join(blanked_words),
                'blanks': blanks,
                'options': list(blanks.values())
            }
        quiz['questions'].append(question)

    return quiz


def evaluate_quiz(quiz, user_answers):
    """
    Evaluate user answers for a quiz.
    """
    results = {
        'total_questions': len(quiz['questions']),
        'correct_answers': 0,
        'score_percentage': 0,
        'feedback': [],
        'question_results': {},
        'weak_areas': [],
        'next_difficulty': quiz['difficulty']
    }
    
    for question in quiz['questions']:
        q_id = question['id']

        if question['type'] == 'reordering':
            # Check if the user's answer matches the correct order
            correct_order = question.get('correct_order', [])
            user_order = user_answers.get(q_id, [])
            if user_order == correct_order:
                results['correct_answers'] += 1
                results['feedback'].append({'question_id': q_id, 'correct': True})
            else:
                results['feedback'].append({'question_id': q_id, 'correct': False, 'correct_answer': correct_order})

        elif question['type'] == 'multiple_choice':
            # Check if the user's selected option matches the correct answer index
            correct_answer = question.get('correct_answer')
            user_answer = user_answers.get(q_id)
            if user_answer == correct_answer:
                results['correct_answers'] += 1
                results['feedback'].append({'question_id': q_id, 'correct': True})
            else:
                results['feedback'].append({
                    'question_id': q_id,
                    'correct': False,
                    'correct_answer': correct_answer
                })

        elif question['type'] == 'fill_blanks':
            # Check if the user's filled blanks match the correct blanks
            correct_blanks = question.get('blanks', {})
            user_blanks = user_answers.get(q_id, {})
            if user_blanks == correct_blanks:
                results['correct_answers'] += 1
                results['feedback'].append({'question_id': q_id, 'correct': True})
            else:
                results['feedback'].append({
                    'question_id': q_id,
                    'correct': False,
                    'correct_answer': correct_blanks
                })
        else:
            print(f"Unknown question type: {question['type']}")

    # Calculate the percentage score
    results['score_percentage'] = (results['correct_answers'] / results['total_questions']) * 100

    if results['score_percentage'] >= 80:
        if quiz['difficulty'] == 'easy':
            results['next_difficulty'] = 'medium'
        elif quiz['difficulty'] == 'medium':
            results['next_difficulty'] = 'hard'
    elif results['score_percentage'] < 50:
        if quiz['difficulty'] == 'hard':
            results['next_difficulty'] = 'medium'
        elif quiz['difficulty'] == 'medium':
            results['next_difficulty'] = 'easy'
    
    # Count occurrences of each weak area
    from collections import Counter
    weak_area_counts = Counter(results['weak_areas'])
    
    # Only keep the most frequent weak areas
    results['weak_areas'] = [area for area, count in weak_area_counts.most_common(2)]
    return results

