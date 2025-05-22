// Function to save quiz result to local storage and prepare for chart update
function saveQuizResult() {
    const quizResult = {
        score_percentage: (correctAnswers / initialTotalQuestions) * 100,
        difficulty: 'easy', // Since this is an alphabet quiz, we'll mark it as easy
        date: new Date().toISOString()
    };
    
    // Get existing quiz history or initialize
    let quizHistory = JSON.parse(localStorage.getItem("quizHistory") || "[]");
    
    // Add new result
    quizHistory.push(quizResult);
    
    // Limit to last 12 quiz results
    if (quizHistory.length > 12) {
        quizHistory = quizHistory.slice(-12);
    }
    
    // Save back to local storage
    localStorage.setItem("quizHistory", JSON.stringify(quizHistory));
    
    return quizResult;
}

// Function to update the chart dynamically
function updateQuizChart(newQuizData) {
    // Find the chart instance
    const chartInstance = Chart.getChart("quizChart");
    
    if (!chartInstance) {
        console.error("Chart instance not found");
        return;
    }
    
    // Prepare chart data
    const labels = chartInstance.data.labels;
    const scores = chartInstance.data.datasets[0].data;
    const difficulties = chartInstance.data.datasets[0]._meta?.difficulty || [];
    
    // Add new data point
    labels.push(`Quiz ${labels.length + 1}`);
    scores.push(newQuizData.score_percentage);
    difficulties.push(newQuizData.difficulty);
    
    // Update chart data
    chartInstance.data.labels = labels;
    chartInstance.data.datasets[0].data = scores;
    chartInstance.data.datasets[0]._meta = { difficulty: difficulties };
    
    // Update point colors based on difficulty
    chartInstance.data.datasets[0].pointBackgroundColor = function(context) {
        const index = context.dataIndex;
        const difficulty = difficulties[index];
        
        if (difficulty === 'easy') return 'rgba(75, 192, 192, 1)';
        if (difficulty === 'medium') return 'rgba(255, 159, 64, 1)';
        return 'rgba(255, 99, 132, 1)';
    };
    
    // Update chart
    chartInstance.update();
    
    // Update average score
    const totalScore = scores.reduce((sum, score) => sum + score, 0);
    const averageScore = (totalScore / scores.length).toFixed(1);
    document.getElementById('average-score').textContent = averageScore + '%';
}

// Modify checkQuizCompletion to save quiz result
function enhancedCheckQuizCompletion() {
    let passingScore = 100; // Example: 80% required to pass
    let progressPercentage = (correctAnswers / initialTotalQuestions) * 100;
    
    // Save quiz result
    const quizResult = saveQuizResult();
    
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

// Override the existing checkQuizCompletion with our enhanced version
window.checkQuizCompletion = enhancedCheckQuizCompletion;

// Initialize chart on progress page load
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the progress page
    const quizChartElement = document.getElementById('quizChart');
    if (quizChartElement) {
        // Get quiz history from local storage
        const localQuizHistory = JSON.parse(localStorage.getItem("quizHistory") || "[]");
        
        // Prepare chart data
        if (localQuizHistory.length > 0) {
            // Prepare data for chart
            const labels = localQuizHistory.map((quiz, index) => `Quiz ${index + 1}`);
            const scores = localQuizHistory.map(quiz => quiz.score_percentage);
            const difficulties = localQuizHistory.map(quiz => quiz.difficulty);
            
            // Create the chart
            const ctx = quizChartElement.getContext('2d');
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Quiz Scores',
                        data: scores,
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2,
                        tension: 0.1,
                        pointBackgroundColor: function(context) {
                            const index = context.dataIndex;
                            const difficulty = difficulties[index];
                            
                            if (difficulty === 'easy') return 'rgba(75, 192, 192, 1)';
                            if (difficulty === 'medium') return 'rgba(255, 159, 64, 1)';
                            return 'rgba(255, 99, 132, 1)';
                        }
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Score (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Quiz Number'
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                afterLabel: function(context) {
                                    const index = context.dataIndex;
                                    const difficulty = difficulties[index];
                                    return 'Difficulty: ' + difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
                                }
                            }
                        }
                    }
                }
            });
        }
    }
});

// Add real-time update when returning from quiz
window.addEventListener('focus', function() {
    // Check if we're on the progress page
    const quizChartElement = document.getElementById('quizChart');
    if (quizChartElement) {
        // Get the latest quiz result
        const quizHistory = JSON.parse(localStorage.getItem("quizHistory") || "[]");
        if (quizHistory.length > 0) {
            const latestQuiz = quizHistory[quizHistory.length - 1];
            
            // Update the chart
            updateQuizChart(latestQuiz);
        }
    }
});