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
        password: "12345#" // Replace with your valid password
    };
    
    document.querySelector('form').addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent default form submission
    
        const name = document.querySelector('input[placeholder="Name"]').value.trim();
        const password = document.getElementById('pass').value.trim();
    
        // Basic Validation
        if (!name || !password) {
            alert('Please fill out all fields.');
            return;
        }
    
        // Validate Credentials
        if (name === validCredentials.name && password === validCredentials.password) {
            alert('Login successful! Redirecting to the home page.');
            window.location.href = "index.html"; // Redirect to the home page
        } else {
            alert('Invalid credentials. Please try again.');
        }
    });
    