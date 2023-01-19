package SaratogaMK3;


public class Main {
    public static void main(String[] args) {
        Matrix inputMatrix = new Matrix(3, 3);
        Matrix weights = new Matrix(3, 3);
        
        inputMatrix.testSeed();
        System.out.println(inputMatrix);
        
    }
}
