// static/js/quiz.js
document.addEventListener('DOMContentLoaded', function() {
    initializeQuizUI();
});

function initializeQuizUI() {
    const quizForm = document.getElementById('quizForm');
    const submitQuizButton = document.getElementById('submitQuiz');
    const resultsContainer = document.getElementById('quizResults');
    
    if (!quizForm || !submitQuizButton) return;
    
    submitQuizButton.addEventListener('click', function(event) {
        event.preventDefault();
        
        // Collect all answers
        const answers = {};
        
        // Process different question types
        document.querySelectorAll('.quiz-question').forEach(questionDiv => {
            const questionId = questionDiv.dataset.questionId;
            const questionType = questionDiv.dataset.questionType;
            
            if (questionType === 'reordering') {
                // Get the current order of words from sortable
                const currentOrder = Array.from(
                    questionDiv.querySelectorAll('.sortable-word')
                ).map(word => word.textContent.trim());
                
                answers[questionId] = currentOrder;
                
            } else if (questionType === 'multiple_choice') {
                // Get selected radio button value
                const selectedOption = questionDiv.querySelector('input[name="' + questionId + '"]:checked');
                if (selectedOption) {
                    answers[questionId] = selectedOption.value;
                }
                
            } else if (questionType === 'fill_blanks') {
                // Get values from all blank inputs
                const blankInputs = questionDiv.querySelectorAll('.blank-input');
                const blankAnswers = {};
                
                blankInputs.forEach(input => {
                    const blankPosition = input.dataset.position;
                    blankAnswers[blankPosition] = input.value.trim();
                });
                
                answers[questionId] = blankAnswers;
            }
        });
        
        // Submit answers to server
        fetch('/submit_quiz', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ answers: answers }),
        })
        .then(response => {
            if (!response.ok) {
                // Log response details for debugging
                return response.json().then(errorDetails => {
                    console.error('Server error details:', errorDetails);
                    throw new Error(`HTTP error! status: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(results => {
            displayQuizResults(results);
            updateProgressInDatabase(results);
        })
        .catch(error => {
            console.error('Error submitting quiz:', error);
            showMessage('Error submitting quiz. Please try again.', 'error');
        });
    });        
    
    // Initialize sortable elements for reordering questions
    document.querySelectorAll('.sortable-container').forEach(container => {
        new Sortable(container, {
            animation: 150,
            ghostClass: 'sortable-ghost'
        });
    });
    
    // Initialize dropdown selectors for fill-in-the-blank questions
    document.querySelectorAll('.blank-input').forEach(input => {
        const datalist = input.nextElementSibling;
        if (datalist && datalist.tagName === 'DATALIST') {
            // Focus event to show dropdown
            input.addEventListener('focus', function() {
                input.setAttribute('list', datalist.id);
            });
            
            // Input event to filter options
            input.addEventListener('input', function() {
                const value = input.value.toLowerCase();
                const options = datalist.querySelectorAll('option');
                
                options.forEach(option => {
                    const optionValue = option.value.toLowerCase();
                    if (optionValue.startsWith(value)) {
                        option.style.display = '';
                    } else {
                        option.style.display = 'none';
                    }
                });
            });
        }
    });
}
function updateProgressInDatabase(results) {
    const date = new Date().toISOString().split('T')[0];
    fetch("/get_current_user")
        .then(response => response.json())
        .then(data => {
            if (data.user_id) {
                level=2
                let newProgress = level + 1;
                let score=3;
                let date = new Date().toISOString().split("T")[0];
                let score1 = results.score_percentage;

                // Log values for debugging
                
                fetch("/update_progress", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                    date: date, 
                    progress_level: results.level, 
                    threequiz:score1
                })
            })
            .then(response => {
                console.log("Update response status:", response.status);
                return response.json();
            })
            .then(data => {
                console.log("Update response data:", data);
                if (data.message) {
                    localStorage.setItem("threequiz",score1);
                    localStorage.setItem("quiz", score);
                    localStorage.setItem(`level${level}Completed`, "true");
                    
                    window.location.href = "roadmap.html";
                } else {
                    alert("Error updating progress: " + (data.error || "Unknown error"));
                }
            })
            .catch(error => {
                console.error("Error updating progress:", error);
                alert("Error updating progress. Check console for details.");
            });
        } else {
            alert("Please log in first.");
            window.location.href = "web.html";
        }
        })
        .catch(error => {
        console.error("Error getting user:", error);
        alert("Error getting user data. Check console for details.");
        });
}
function displayQuizResults(results) {
    const resultsContainer = document.getElementById('quizResults');
    const questionsContainer = document.getElementById('questionsContainer');
    
    if (!resultsContainer || !questionsContainer) return;
    
    // Show results section
    resultsContainer.style.display = 'block';
    
    // Update score
    const scoreElement = document.getElementById('scorePercentage');
    if (scoreElement) {
        scoreElement.textContent = `${Math.round(results.score_percentage)}%`;
        
        // Change color based on score
        if (results.score_percentage >= 80) {
            scoreElement.className = 'score high-score';
        } else if (results.score_percentage >= 60) {
            scoreElement.className = 'score medium-score';
        } else {
            scoreElement.className = 'score low-score';
        }
    }
    
    // Update correct/total counters
    const correctElement = document.getElementById('correctAnswers');
    const totalElement = document.getElementById('totalQuestions');
    
    if (correctElement) {
        correctElement.textContent = results.correct_answers;
    }
    
    if (totalElement) {
        totalElement.textContent = results.total_questions;
    }
    
    // Display feedback for each question
    for (const [questionId, result] of Object.entries(results.question_results)) {
        const questionDiv = document.querySelector(`.quiz-question[data-question-id="${questionId}"]`);
        
        if (questionDiv) {
            // Add correct/incorrect class
            questionDiv.classList.remove('correct', 'incorrect');
            questionDiv.classList.add(result.correct ? 'correct' : 'incorrect');
            
            // Display feedback
            let feedbackDiv = questionDiv.querySelector('.question-feedback');
            
            if (!feedbackDiv) {
                feedbackDiv = document.createElement('div');
                feedbackDiv.className = 'question-feedback';
                questionDiv.appendChild(feedbackDiv);
            }
            
            feedbackDiv.textContent = result.feedback;
        }
    }
    
    // Display next steps based on score
    const nextStepsElement = document.getElementById('nextSteps');
    
    if (nextStepsElement) {
        let nextStepsText = '';
        
        if (results.score_percentage >= 80) {
            nextStepsText = `Great job! You're ready to move to ${results.next_difficulty} difficulty.`;
        } else if (results.score_percentage >= 60) {
            nextStepsText = 'Good effort! Try reviewing the incorrect answers before moving on.';
        } else {
            nextStepsText = 'Keep practicing! Consider revisiting the learning materials before trying again.';
        }
        
        nextStepsElement.textContent = nextStepsText;
    }
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
    
    // Hide submit button
    const submitButton = document.getElementById('submitQuiz');
    if (submitButton) {
        submitButton.style.display = 'none';
    }
    
    // Show retry button
    const retryButton = document.getElementById('retryQuiz');
    if (retryButton) {
        retryButton.style.display = 'inline-block';
        retryButton.addEventListener('click', function() {
            // Reload page to get a new quiz
            window.location.reload();
        });
    }
}

function showMessage(message, type = 'info') {
    const messageContainer = document.getElementById('messageContainer');
    
    if (!messageContainer) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${type}`;
    messageElement.textContent = message;
    
    messageContainer.appendChild(messageElement);
    
    // Remove message after 5 seconds
    setTimeout(() => {
        messageElement.classList.add('fade-out');
        setTimeout(() => {
            messageContainer.removeChild(messageElement);
        }, 500);
    }, 5000);
}