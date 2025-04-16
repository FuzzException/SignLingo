let currentQuestionIndex = 0;
let correctAnswers = 0;
let incorrectQuestions = [];
let answeredQuestions = [];
let totalQuestions = 0;
let initialTotalQuestions = 0; // New variable to store the original total number of questions

let wordData = [
    { word: 'A' },
    { word: 'B' },
    { word: 'C' },
    { word: 'D' },
    { word: 'E' },
    { word: 'F' },
    { word: 'G' },
    { word: 'H' },
    { word: 'J' },
    { word: 'K' },
    { word: 'L' },
    { word: 'M' },
    { word: 'N' },
    { word: 'O' },
    { word: 'P' },
    { word: 'Q' },
    { word: 'R' },
    { word: 'S' },
    { word: 'T' },
    { word: 'U' },
    { word: 'V' },
    { word: 'W' },
    { word: 'X' },
    { word: 'Y' },
    { word: 'Z' },
    { word: 'I' }
];

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
    completeLevel(0);
    if (progressPercentage >= passingScore) {
        alert("Congratulations! You have completed Level 1.");
        completeLevel(1);
        localStorage.setItem("level1Completed", "true");
        window.location.href = "temp.html"; // Redirect back to roadmap
    } else {
        alert("Try again! You need 100% to pass.");
        window.location.href = "quiz.html";
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
    imageElement.src = `/static/assets/${question.word}.png`;
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

function completeLevel(level) {
    // Remove dependency on user object
    fetch("/get_current_user")
        .then(response => response.json())
        .then(data => {
            if (data.user_id) {
                let newProgress = level + 1;
                let score=1;
                let date = new Date().toISOString().split("T")[0];
                let score1 = (correctAnswers / initialTotalQuestions) * 100;

                // Log values for debugging
                console.log("Sending to server:", {
                    date: date,
                    progress_level: newProgress,
                    alphaquiz: score1
                });
                
                fetch("/update_progress", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        date: date, 
                        progress_level: newProgress, 
                        alphaquiz: score1,
                        quiz: score
                    })
                })
                .then(response => {
                    console.log("Update response status:", response.status);
                    return response.json();
                })
                .then(data => {
                    console.log("Update response data:", data);
                    if (data.message) {
                        localStorage.setItem("alphaquiz", score1);
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
//document.getElementById("finish-quiz-btn").addEventListener("click", function() {
     // Change the number based on the current level
//});
// Initialize quiz and start
document.addEventListener('DOMContentLoaded', () => {
    initialTotalQuestions = wordData.length; // Store original total number of questions
    totalQuestions = wordData.length;  // Set total questions count
    shuffleArray(wordData);  // Shuffle the questions
    loadQuestion();  // Load the first question
});