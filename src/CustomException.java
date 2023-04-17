package src;

public class CustomException extends Exception {
    // Constructors
    public CustomException() {
        super(); // Call superclass constructor
    }

    public CustomException(String message) {
        super(message); // Call superclass constructor with custom message
    }

    // Optional: Define additional constructors or methods specific to your exception
}