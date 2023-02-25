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
            this.kernals = new Matrix[NUM_FEATURE_SETS * input.z];
            for(int i = 0; i < kernals.length; i++){
                kernals[i] = new Matrix(1, KERNAL_SIZE, KERNAL_SIZE); //its not doing anything when i hit step over //i am clicking it
                kernals[i].seed();
            }
            initialized = true;
        }
        Matrix result = new Matrix(NUM_FEATURE_SETS * input.z, input.rows - KERNAL_SIZE + 1, input.cols - KERNAL_SIZE + 1);
        for(int i = 0 ; i < result.matrix.length; i++){
            result.matrix[i] = input.convolution(kernals[i]).matrix[0];
        }
        return result;
    }
    public Matrix backward(Matrix previousGradients){ //dLdO = previousGradients
        Matrix dldf = input.convolution(previousGradients);
        Matrix result = new Matrix(1, input.rows + previousGradients.rows - 1, input.cols + previousGradients.cols - 1);
        for(int i = 0; i < kernals.length; i++){
            
            Matrix temp = kernals[i];
            temp.reverse();
            Matrix dldx = temp.bigConvolution(previousGradients); //idk how to write a big convolution
            for(int k = 0; k < dldx.size; k++){
                result.matrix[0][k] += dldx.matrix[0][k];
            }
            for(int j = 0; j < kernals[i].size; j++){
                kernals[i].matrix[0][j] = kernals[i].matrix[0][j] - dldf.matrix[i][j] * Main.ALPHA;
            }
        }
        return result;
    }
}
