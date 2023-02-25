package src;

public class Flatten extends Layer{
    int z;
    int rows;
    int cols;
    public Matrix forward(Matrix input){
        Matrix result = new Matrix(1, input.z * input.cols * input.rows, 1);
        this.z = input.z;
        this.rows = input.rows;
        this.cols = input.cols;
        for(int i = 0; i < input.z; i++){
            for(int j = 0; j < input.rows * input.cols; j++){
                result.matrix[0][i * input.rows * input.cols + j] = input.matrix[i][j];
            }
        }
        return result;
    }
    public Matrix backward(Matrix notInput){
        Matrix result = new Matrix(z, rows, cols);
        for(int i = 0; i < z; i++){
            for(int j = 0; j < rows * cols; j++){
                result.matrix[i][j] = notInput.matrix[0][i * rows * cols + j];
            }
        }
        return result;
    }
}
