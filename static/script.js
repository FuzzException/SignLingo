function goToInfoPage() {
    document.getElementById("home").classList.add("hidden");
    document.getElementById("user-info").classList.remove("hidden");
}

// Function to handle login
function loginUser() {
    let username = document.getElementById("name").value;  // Getting value from "name" input
    let gender = document.getElementById("gender").value;

    if (!username || !gender) {
        alert("Please enter your name and select your gender");
        return;
    }

    fetch("/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, gender })  // IMPORTANT: Match the keys with what your backend expects
    })
    .then(response => response.json())
    .then(data => {
        if (data.user_id) {
            console.log("Login Successful. User ID:", data.user_id);
            localStorage.setItem("user_id", data.user_id);
            localStorage.setItem("username", username);
            localStorage.setItem("userGender", gender);
            window.location.href = "/dashboard.html";
        } else {
            alert("Sign Up first.");
        }
    })
    .catch(error => console.error("Error logging in:", error));
}

// Function to handle signup
function signupUser() {
    let username = document.getElementById("name").value;  // Getting value from "name" input
    let gender = document.getElementById("gender").value;

    if (!username || !gender) {
        alert("Please enter your name and select your gender");
        return;
    }

    fetch("/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, gender })  // IMPORTANT: Match the keys with what your backend expects
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        if (data.message === "Signup successful!") {
            localStorage.setItem("user_id", data.user_id);
            localStorage.setItem("username", username);
            localStorage.setItem("userGender", gender);
            window.location.href = "/dashboard.html";
        }
    })
    .catch(error => console.error("Error signing up:", error));
}
// Don't automatically call these functions on page load
// They should only be called when buttons are clicked
document.addEventListener("DOMContentLoaded", function() {
    // Only set up event listeners here, don't call the functions directly
    console.log("Page loaded and ready");
});