package tests;

public class UngaBungaParallelizer {
    static long sum;
    public static void main(String[] args) throws InterruptedException {
        Thread[] threads = new Thread[10];
        sum = 0;
        sum = 0;
        for(int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for(int j = 0; j < 100_000_000; j++) {
                    sum++;
                }
            });
            threads[i].start();
        }
        for(int i = 0; i < 10; i++) {
            threads[i].join();
        }
        System.out.println(sum);

    }
}

