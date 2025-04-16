// Fetch user progress data from the server
async function fetchProgressData() {
    try {
        const response = await fetch('/api/user_progress');
        if (!response.ok) {
            // Try to build progress data from localStorage if API fails
            return buildProgressFromLocalStorage();
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching progress data:', error);
        return buildProgressFromLocalStorage();
    }
}

// Build progress data from localStorage as fallback
function buildProgressFromLocalStorage() {
    // Default empty structure
    const defaultData = {
        quizzes_taken: 0,
        words_learned: 0,
        learning_streak: 0,
        current_level: 'beginner',
        progress_percentage: 0,
        quiz_history: [],
        weak_areas: [],
        recent_activity: []
    };
    
    try {
        // Check if we have quiz data in localStorage
        const alphaQuizScore = localStorage.getItem('alphaquiz');
        const level1Completed = localStorage.getItem('level1Completed') === 'true';
        
        if (alphaQuizScore) {
            const score = parseFloat(alphaQuizScore);
            const today = new Date().toISOString().split('T')[0];
            
            defaultData.quizzes_taken = 1;
            defaultData.words_learned = Math.round(26 * (score / 100)); // Assuming 26 alphabet letters
            defaultData.current_level = level1Completed ? 'intermediate' : 'beginner';
            defaultData.progress_percentage = level1Completed ? 25 : Math.round(score / 4);
            
            // Add quiz history
            defaultData.quiz_history = [{
                score_percentage: score,
                difficulty: 'easy',
                date: today
            }];
            
            // Add recent activity
            defaultData.recent_activity = [{
                type: 'quiz',
                details: `Completed Alphabet Quiz with ${score}% score`,
                date: new Date().toISOString()
            }];
            
            // Check if user has completed level 1
            if (level1Completed) {
                defaultData.recent_activity.unshift({
                    type: 'learning',
                    details: 'Completed Level 1 - Alphabet',
                    date: new Date().toISOString()
                });
            }
        }
        
        return defaultData;
    } catch (error) {
        console.error('Error building local progress data:', error);
        return defaultData;
    }
}

// Update summary statistics
function updateSummaryStats(stats) {
    document.getElementById('quizzes-taken').textContent = stats.quizzes_taken;
    document.getElementById('words-learned').textContent = stats.words_learned;
    document.getElementById('learning-streak').textContent = stats.learning_streak;
    
    // Set level with first letter capitalized
    const levelDisplay = stats.current_level.charAt(0).toUpperCase() + stats.current_level.slice(1);
    document.getElementById('current-level').textContent = levelDisplay;
}

// Update progress bar
function updateProgressBar(percentage) {
    document.getElementById('progress-fill').style.width = `${percentage}%`;
    document.getElementById('progress-percentage').textContent = `${percentage}%`;
}

// Calculate and display average quiz score
function updateAverageScore(quizHistory) {
    const averageScoreElement = document.getElementById('average-score');
    
    if (!quizHistory || quizHistory.length === 0) {
        averageScoreElement.textContent = 'N/A';
        return;
    }
    
    const totalScore = quizHistory.reduce((sum, quiz) => sum + quiz.score_percentage, 0);
    const averageScore = (totalScore / quizHistory.length).toFixed(1);
    averageScoreElement.textContent = `${averageScore}%`;
}

// Create quiz performance chart
function createQuizChart(quizHistory) {
    if (!quizHistory || quizHistory.length === 0) {
        document.getElementById('quizChart').style.display = 'none';
        return;
    }
    
    const ctx = document.getElementById('quizChart').getContext('2d');
    
    // Prepare chart data
    const labels = quizHistory.map((quiz, index) => {
        // If quiz has a date, format it, otherwise use quiz number
        if (quiz.date) {
            const date = new Date(quiz.date);
            return isNaN(date.getTime()) ? 
                `Quiz ${index + 1}` : 
                date.toLocaleDateString('en-US', {month: 'short', day: 'numeric'});
        }
        return `Quiz ${index + 1}`;
    });
    
    const scores = quizHistory.map(quiz => quiz.score_percentage);
    const difficulties = quizHistory.map(quiz => quiz.difficulty || 'easy');
    
    // Define colors based on difficulty
    const getPointColor = (difficulty) => {
        switch(difficulty.toLowerCase()) {
            case 'easy': return 'rgba(75, 192, 192, 1)';
            case 'medium': return 'rgba(255, 159, 64, 1)';
            case 'hard': return 'rgba(255, 99, 132, 1)';
            default: return 'rgba(75, 192, 192, 1)';
        }
    };
    
    const pointColors = difficulties.map(getPointColor);
    
    // Create chart
    const quizChart = new Chart(ctx, {
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
                pointBackgroundColor: pointColors
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
                        text: 'Quiz Date'
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

// Update weak areas section
function updateWeakAreas(weakAreas, quizzesTaken) {
    const noWeakAreasElement = document.getElementById('no-weak-areas');
    const weakAreasListElement = document.getElementById('weak-areas-list');
    
    if (!weakAreas || weakAreas.length === 0) {
        // If no weak areas data available, infer from localStorage (if possible)
        const alphaQuizScore = localStorage.getItem('alphaquiz');
        
        if (alphaQuizScore && parseFloat(alphaQuizScore) < 100) {
            noWeakAreasElement.style.display = 'none';
            weakAreasListElement.style.display = 'block';
            weakAreasListElement.innerHTML = ''; // Clear existing items
            
            // Create weak area for alphabet
            const li = document.createElement('li');
            
            const areaName = document.createElement('div');
            areaName.className = 'area-name';
            areaName.textContent = 'Finger Spelling';
            
            const areaBar = document.createElement('div');
            areaBar.className = 'area-bar';
            
            const areaFill = document.createElement('div');
            areaFill.className = 'area-fill';
            // Calculate fill based on score (100 - score%)
            const fillPercentage = 100 - parseFloat(alphaQuizScore);
            areaFill.style.width = `${fillPercentage}%`;
            
            areaBar.appendChild(areaFill);
            li.appendChild(areaName);
            li.appendChild(areaBar);
            weakAreasListElement.appendChild(li);
            return;
        }
        
        noWeakAreasElement.style.display = 'block';
        weakAreasListElement.style.display = 'none';
        return;
    }
    
    noWeakAreasElement.style.display = 'none';
    weakAreasListElement.style.display = 'block';
    weakAreasListElement.innerHTML = ''; // Clear existing items
    
    weakAreas.forEach(area => {
        const li = document.createElement('li');
        
        const areaName = document.createElement('div');
        areaName.className = 'area-name';
        // Format area name (replace underscores with spaces and capitalize)
        areaName.textContent = area.area
            .replace(/_/g, ' ')
            .replace(/(^\w{1})|(\s+\w{1})/g, letter => letter.toUpperCase());
        
        const areaBar = document.createElement('div');
        areaBar.className = 'area-bar';
        
        const areaFill = document.createElement('div');
        areaFill.className = 'area-fill';
        // Calculate percentage width based on error count vs total quizzes
        const fillPercentage = Math.min(Math.round((area.count / quizzesTaken) * 100), 100);
        areaFill.style.width = `${fillPercentage}%`;
        
        areaBar.appendChild(areaFill);
        li.appendChild(areaName);
        li.appendChild(areaBar);
        weakAreasListElement.appendChild(li);
    });
}

// Update recent activity section
function updateRecentActivity(activities) {
    const noActivityElement = document.getElementById('no-activity');
    const activityListElement = document.getElementById('activity-list');
    
    if (!activities || activities.length === 0) {
        // Try to generate activity from localStorage if none provided
        const alphaQuizScore = localStorage.getItem('alphaquiz');
        const level1Completed = localStorage.getItem('level1Completed') === 'true';
        
        if (alphaQuizScore || level1Completed) {
            noActivityElement.style.display = 'none';
            activityListElement.style.display = 'block';
            activityListElement.innerHTML = ''; // Clear existing activities
            
            const activities = [];
            
            if (level1Completed) {
                activities.push({
                    type: 'learning',
                    details: 'Completed Level 1 - Alphabet',
                    date: new Date().toISOString()
                });
            }
            
            if (alphaQuizScore) {
                activities.push({
                    type: 'quiz',
                    details: `Completed Alphabet Quiz with ${alphaQuizScore}% score`,
                    date: new Date().toISOString()
                });
            }
            
            renderActivities(activities, activityListElement);
            return;
        }
        
        noActivityElement.style.display = 'block';
        activityListElement.style.display = 'none';
        return;
    }
    
    noActivityElement.style.display = 'none';
    activityListElement.style.display = 'block';
    activityListElement.innerHTML = ''; // Clear existing activities
    
    renderActivities(activities, activityListElement);
}

// Helper function to render activities
function renderActivities(activities, container) {
    activities.forEach(activity => {
        const li = document.createElement('li');
        li.className = `activity-item ${activity.type}`;
        
        // Create activity icon
        const activityIcon = document.createElement('div');
        activityIcon.className = 'activity-icon';
        
        const iconSpan = document.createElement('span');
        iconSpan.className = `icon ${activity.type}-icon`;
        iconSpan.textContent = activity.type === 'quiz' ? 'Q' : 'L'; // Q for quiz, L for learning
        
        activityIcon.appendChild(iconSpan);
        
        // Create activity details
        const activityDetails = document.createElement('div');
        activityDetails.className = 'activity-details';
        
        const activityType = document.createElement('div');
        activityType.className = 'activity-type';
        activityType.textContent = activity.type.charAt(0).toUpperCase() + activity.type.slice(1);
        
        const activityInfo = document.createElement('div');
        activityInfo.className = 'activity-info';
        activityInfo.textContent = activity.details;
        
        const activityDate = document.createElement('div');
        activityDate.className = 'activity-date';
        // Format date string
        activityDate.textContent = formatDate(activity.date);
        
        activityDetails.appendChild(activityType);
        activityDetails.appendChild(activityInfo);
        activityDetails.appendChild(activityDate);
        
        li.appendChild(activityIcon);
        li.appendChild(activityDetails);
        container.appendChild(li);
    });
}

// Format date string
function formatDate(dateString) {
    if (!dateString) return 'Unknown date';
    
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return dateString; // Return original if invalid
    
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Generate learning recommendations based on progress data
function generateRecommendations(stats) {
    const recommendationsList = document.getElementById('recommendations-list');
    recommendationsList.innerHTML = ''; // Clear existing recommendations
    
    // Check localStorage for additional info
    const alphaQuizScore = localStorage.getItem('alphaquiz');
    const level1Completed = localStorage.getItem('level1Completed') === 'true';
    
    // Generate personalized recommendations
    const recommendations = [];
    
    // Recommendation based on alphabet quiz
    if (alphaQuizScore) {
        const score = parseFloat(alphaQuizScore);
        if (score < 100) {
            recommendations.push('Practice finger spelling to improve your alphabet recognition.');
        } else {
            recommendations.push('Great job mastering the alphabet! Continue to the next level.');
        }
    } else {
        recommendations.push('Take the alphabet quiz to begin your learning journey.');
    }
    
    // Recommendation based on level completion
    if (level1Completed) {
        recommendations.push('You\'ve completed Level 1! Move on to the next challenge.');
    }
    
    // Recommendation based on learning streak
    if (stats.learning_streak > 0) {
        recommendations.push(`You have a ${stats.learning_streak} day streak. Keep it up for better retention!`);
    } else {
        recommendations.push('Try to practice daily to build a learning streak and improve retention.');
    }
    
    // Recommendation based on weak areas
    if (stats.weak_areas && stats.weak_areas.length > 0) {
        const weakAreaName = stats.weak_areas[0].area.replace(/_/g, ' ');
        recommendations.push(`Focus on practicing ${weakAreaName} exercises to improve your skills.`);
    }
    
    // Add recommendations to the page
    recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        recommendationsList.appendChild(li);
    });
    
    // Add default recommendation if none generated
    if (recommendations.length === 0) {
        const li = document.createElement('li');
        li.textContent = 'Complete more quizzes to receive personalized recommendations.';
        recommendationsList.appendChild(li);
    }
}

// Initialize chart legend
function initializeChartLegend() {
    // Find or create container for legend
    let chartContainer = document.querySelector('.quiz-performance .card');
    let existingLegend = document.querySelector('.chart-legend');
    
    if (existingLegend) {
        existingLegend.remove(); // Remove existing legend if present
    }
    
    // Create new legend
    const legendContainer = document.createElement('div');
    legendContainer.className = 'chart-legend';
    
    const difficulties = ['easy', 'medium', 'hard'];
    const colors = [
        'rgba(75, 192, 192, 1)',
        'rgba(255, 159, 64, 1)',
        'rgba(255, 99, 132, 1)'
    ];
    
    difficulties.forEach((difficulty, index) => {
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        
        const colorBox = document.createElement('div');
        colorBox.className = 'legend-color';
        colorBox.style.backgroundColor = colors[index];
        
        const label = document.createElement('span');
        label.textContent = difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
        
        legendItem.appendChild(colorBox);
        legendItem.appendChild(label);
        legendContainer.appendChild(legendItem);
    });
    
    // Add legend to chart container
    if (chartContainer) {
        chartContainer.appendChild(legendContainer);
    }
}

// Initialize the progress page with data
async function initializeProgressPage() {
    try {
        // Fetch progress data from server
        const progressData = await fetchProgressData();
        
        // Update page components with data
        updateSummaryStats(progressData);
        updateProgressBar(progressData.progress_percentage);
        updateAverageScore(progressData.quiz_history);
        createQuizChart(progressData.quiz_history);
        updateWeakAreas(progressData.weak_areas, progressData.quizzes_taken);
        updateRecentActivity(progressData.recent_activity);
        generateRecommendations(progressData);
        initializeChartLegend();
    } catch (error) {
        console.error('Error initializing progress page:', error);
        // Display error message to user
        const errorMessage = document.createElement('div');
        errorMessage.className = 'error-message';
        errorMessage.textContent = 'Failed to load progress data. Please try again later.';
        document.querySelector('.progress-content').prepend(errorMessage);
    }
}

// Add some CSS for the chart legend and other elements
function addLegendStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .chart-legend {
            display: flex;
            justify-content: center;
            margin-top: 10px;
            gap: 15px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            font-size: 0.85rem;
        }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .error-message {
            background-color: #ffecec;
            color: #721c24;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 15px;
            text-align: center;
        }
        .activity-item {
            display: flex;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .activity-icon {
            margin-right: 12px;
        }
        .icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            color: white;
            font-weight: bold;
        }
        .quiz-icon {
            background-color: #4caf50;
        }
        .learning-icon {
            background-color: #2196f3;
        }
        .activity-details {
            flex: 1;
        }
        .activity-type {
            font-weight: bold;
            margin-bottom: 4px;
        }
        .activity-info {
            margin-bottom: 4px;
        }
        .activity-date {
            font-size: 0.8rem;
            color: #777;
        }
    `;
    document.head.appendChild(style);
}

// Run on page load
document.addEventListener('DOMContentLoaded', () => {
    addLegendStyles();
    initializeProgressPage();
});