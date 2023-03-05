package src;
import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

public class Main {
    public static double ALPHA = 0.001;
    public static Matrix[] batch;
    public static void main(String[] args) throws IOException {
        BufferedImage image = ImageIO.read(new File("images/bird.png"));
        image = Matrix.makeSquare(image); //625x625 range(0:255)
        Matrix imageMatrix = Matrix.imageToMatrix(image); //(625*625 x 1) range(0:255)
        imageMatrix.normalizePixels(); //(625*625 x 1) range(-1:1) 
        // MaxPool makeFater = new MaxPool(11);
        // imageMatrix = makeFater.forward(imageMatrix);
      
        Matrix expected = new Matrix(1, 5, 1);
        expected.matrix = new Double[][]{{0.0, 1.0, 0.0, 0.0, 0.0}};
        Matrix testInput = new Matrix(1, 28900, 1);
        testInput.seed();

        Model model = new Model(); //1, 70, 70
        model.layers.add(new ConvolutionLayer(10, 1, 3)); //10, 68, 68
        model.layers.add(new ReLU()); //10, 68, 68
        model.layers.add(new MaxPool(6)); //10, 12, 12
        model.layers.add(new ConvolutionLayer(10, 1, 3)); //100, 10, 10
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(6));
        model.layers.add(new Flatten());
        model.layers.add(new DenseLayer(256));
        model.layers.add(new ReLU());
        model.layers.add(new DenseLayer(5));
        model.layers.add(new Softmax());

        double loss = Double.POSITIVE_INFINITY;
        int epoch = 0;
        batch = new Matrix[]{imageMatrix};
        Matrix[] batchExpected = new Matrix[]{expected};
        // model.profiling = true;
        while(loss > 0.01){
            for(int i = 0; i < batch.length; i++){
                loss = model.forward(batch[i], batchExpected[i]);
                model.backward();
            }
            
            for(Layer layer: model.layers){
                layer.updateParams();
            }
            epoch++;
        }
        System.out.println("Completed in " + epoch + " epochs");
          //ima smack you with my pimp cane
    }
} 