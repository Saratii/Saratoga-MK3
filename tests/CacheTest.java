package tests;

public class CacheTest {
    public static void main(String[] args) {
        CacheTest cacheLine = new CacheTest();
        cacheLine.startTesting();
    }

    private void startTesting() {
        byte[] array = new byte[128 * 1024];
        for (int testIndex = 0; testIndex < 10; testIndex++) {
            testMethod(array);
            System.out.println("--------- // ---------");
        }

    }

    private void testMethod(byte[] array) {
        for (int len = 8192; len <= array.length; len += 8192) {

            long t0 = System.nanoTime();
            for (int i = 0; i < 10000; i++) {
                for (int k = 0; k < len; k += 64) {
                    array[k] = 1;
                }
            }

            long dT = System.nanoTime() - t0;
            System.out.println("len: " + len/1024 + " dT: " + dT + " dT/stepCount: " + (dT) / len);
        }
    }
}
