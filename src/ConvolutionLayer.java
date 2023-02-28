package src;

public class ConvolutionLayer extends Layer{
    Matrix input;
    int NUM_FEATURE_SETS;
    int STRIDE;
    int KERNAL_SIZE;
    public Matrix[] kernals;
    boolean initialized = false;
    public ConvolutionLayer(int NUM_FEATURE_SETS, int STRIDE, int KERNAL_SIZE){
        this.NUM_FEATURE_SETS = NUM_FEATURE_SETS;
        this.STRIDE = STRIDE;
        this.KERNAL_SIZE = KERNAL_SIZE;
    }
    
    public Matrix forward(Matrix input){
        this.input = input;
        if(!initialized){
            this.kernals = new Matrix[NUM_FEATURE_SETS];
            for(int i = 0; i < kernals.length; i++){
                kernals[i] = new Matrix(1, KERNAL_SIZE, KERNAL_SIZE); 
                kernals[i].seed();
            }
            initialized = true;
        }
        Matrix result = new Matrix(NUM_FEATURE_SETS * input.z, input.rows - KERNAL_SIZE + 1, input.cols - KERNAL_SIZE + 1);
        for(int i = 0 ; i < result.matrix.length; i++){
            result.matrix[i] = input.convolution(kernals[i % kernals.length]).matrix[0];
        }
        return result;
    }
    public Matrix backward(Matrix previousGradients){ //dLdO = previousGradients
        Matrix dldf = input.convolution(previousGradients);
        Matrix result = new Matrix(previousGradients.z / kernals.length, input.rows, input.cols);
        for(int i = 0; i < kernals.length; i++){
            Matrix temp = kernals[i];
            temp.reverse();
            Matrix dldx = temp.bigConvolution(previousGradients);
            for(int h = 0; h < dldx.z; h++){
                for(int k = 0; k < dldx.rows * dldx.cols; k++){
                    result.matrix[h % result.z][k] = dldx.matrix[h][k];
                }
            }
            for(int j = 0; j < kernals[i].size; j++){
                kernals[i].matrix[0][j] = kernals[i].matrix[0][j] - dldf.matrix[i][j] * Main.ALPHA;
            }
        }
        return result;
    }
}