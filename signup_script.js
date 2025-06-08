function myfunction(){
    var x = document.getElementById("pass");
    if (x.type === "password") {
      x.type = "text";
    } else {
      x.type = "password";
    }
  }
  const validCredentials = {
    name: "admin", // Replace with your valid username
    password: "12345#", // Replace with your valid password
    email: "aryachuri00@gmail.com" // Replace with your valid email
};

document.querySelector('form').addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent default form submission

    const name = document.getElementById('name').value.trim();
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('pass').value.trim();

    // Basic Validation
    if (!name || !email || !password) {
        alert('Please fill out all fields.');
        return;
    }

    // Match credentials with validCredentials
    if (
        name === validCredentials.name &&
        email === validCredentials.email &&
        password === validCredentials.password
    ) {
        alert('Registration successful! Redirecting to login page.');
        window.location.href = "login.html"; // Redirect to login page
    } else {
        alert('Invalid credentials. Please check your inputs.');
    }
});


//backend 
