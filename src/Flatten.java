package src;

public class Flatten extends Layer{
    public Matrix forward(Matrix input){
        Matrix result = new Matrix(1, input.z * input.cols * input.rows, 1);
        for(int i = 0; i < input.z; i++){
            for(int j = 0; j < input.rows * input.cols; j++){
                result.matrix[0][i * input.rows * input.cols + j] = input.matrix[i][j];
            }
        }
        return result;
    }
}
