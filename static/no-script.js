let currentQuestionIndex = 0;
let correctAnswers = 0;
let incorrectQuestions = [];
let answeredQuestions = [];
let totalQuestions = 0;
let initialTotalQuestions = 0; // New variable to store the original total number of questions

let wordData = [
    { word: '0' },
    { word: '1' },
    { word: '2' },
    { word: '3' },
    { word: '4' },
    { word: '5' },
    { word: '6' },
    { word: '7' },
    { word: '8' },
    { word: '9' }
];

// Shuffle function for randomizing questions
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// Update progress bar after each correct answer
function updateProgressBar(forceComplete = false) {
    const progressBar = document.getElementById('quiz-progress');
    const progressText = document.getElementById('progress-text');

    // Calculate progress percentage
    const progressPercentage = forceComplete
        ? 100
        : (correctAnswers / initialTotalQuestions) * 100;

    progressBar.value = progressPercentage;
    progressText.textContent = `${Math.round(progressPercentage)}% completed`;
}

function checkQuizCompletion() {
    let passingScore = 100; // Example: 80% required to pass
    let progressPercentage = (correctAnswers / initialTotalQuestions) * 100;
    if (correctAnswers === initialTotalQuestions) {
        progressPercentage = 100;
    }
    fetch("/get_current_user")
        .then(response => response.json())
        .then(data => {
        if (data.user_id) {
            let score=level+1;
            if (data.user_id) {
                fetch("/update_progress", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({noquiz:progressPercentage,
                            quiz:score
                     })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.message) {
                        localStorage.setItem("noquiz", progressPercentage);
                        localStorage.setItem("quiz", score);
                    }
                });   
            }
        }
    });
    if (progressPercentage >= passingScore) {
        alert("Congratulations! You have completed the quiz.");
        window.location.href = "temp.html"; // Redirect back to roadmap
    } else {
        alert("Try again! You need 100% to pass.");
        window.location.href = "number.html";
    }
}
// Load question and display image
function loadQuestion() {
    const feedback = document.getElementById('feedback');

    // If all questions, including incorrect ones, are done

    if (currentQuestionIndex >= wordData.length && incorrectQuestions.length === 0) {
        feedback.textContent = 'Quiz completed!';
        checkQuizCompletion();
        updateProgressBar(true); // Ensure progress bar reaches 100%

        // Redirect to temp.html after 2 seconds
        setTimeout(() => {
            window.location.href = 'temp.html';
        }, 2000);
        return;
    }

    // If revisiting incorrect questions

    const question = wordData[currentQuestionIndex];
    const imageElement = document.getElementById('quiz-image');
    const optionsContainer = document.getElementById('options-container');

    // Set image source
    imageElement.src = `/static/assets/${question.word}.jpg`;
    imageElement.alt = `Image of the letter ${question.word}`;
    // Clear previous options and feedback
    optionsContainer.innerHTML = '';
    feedback.textContent = ''; // Clear previous feedback

    // Generate randomized options
    const options = generateOptions(question.word);
    options.forEach((option, index) => {
        const button = document.createElement('button');
        button.classList.add('option');
        button.textContent = option;
        button.onclick = () => checkAnswer(index, question.word, options);
        optionsContainer.appendChild(button);
    });
}

function generateOptions(correctWord) {
    const options = new Set([correctWord]);
    while (options.size < 4) {
        const randomWord = wordData[Math.floor(Math.random() * wordData.length)].word;
        options.add(randomWord);
    }
    return Array.from(options).sort(() => Math.random() - 0.5); // Shuffle options
}

// Check answer and provide feedback
function checkAnswer(selectedIndex, correctWord, options) {
    const feedback = document.getElementById('feedback');

    // If answer is correct
    if (options[selectedIndex] === correctWord) {
        correctAnswers++;
        feedback.textContent = 'Correct!';
        updateProgressBar();
    } else {
        feedback.textContent = `Incorrect! Correct answer: ${correctWord}`;
         // Add to incorrect list
    }

    // Proceed to next question
    currentQuestionIndex++;
    setTimeout(loadQuestion, 2000); // Wait 2 seconds before loading next question
}

// Initialize quiz and start
document.addEventListener('DOMContentLoaded', () => {
    initialTotalQuestions = wordData.length; 
    totalQuestions = wordData.length;  
    shuffleArray(wordData); 
    loadQuestion();  
});