package tests;

public class UngaBungaParallelizer {
    public void ungaBungaParallel() {
        // Number of threads to use
        int numThreads = 4;

        // Create an array to hold the threads
        Thread[] threads = new Thread[numThreads];

        // Define the parameter for ungaBunga()
        String parameter = "Hello from thread";

        // Start the threads
        for (int i = 0; i < numThreads; i++) {
            final int threadIndex = i; // final variable for lambda expression
            threads[i] = new Thread(() -> ungaBunga(parameter, threadIndex));
            threads[i].start();
        }

        // Wait for all threads to complete
        for (int i = 0; i < numThreads; i++) {
            try {
                threads[i].join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public void ungaBunga(String parameter, int threadIndex) {
        // Code to perform the ungaBunga() function with the given parameter and thread index
        // This will be executed concurrently in multiple threads
        System.out.println(parameter + " " + threadIndex + ": " + Thread.currentThread().getName());
    }

    public static void main(String[] args) {
        UngaBungaParallelizer parallelizer = new UngaBungaParallelizer();
        parallelizer.ungaBungaParallel();
    }
}