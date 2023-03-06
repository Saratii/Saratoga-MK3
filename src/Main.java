package src;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

public class Main {
    public static double ALPHA = 0.001;
    public static Matrix[] batch;
    public static void main(String[] args) throws IOException {
        
        File[] dolphinDirectory = new File("Aminals/").listFiles();
        List<Image> images = new ArrayList<>();
        for(File file: dolphinDirectory){
            images.add(new Image(file, "dolphin"));
        }
       
       
      
        Matrix expected = new Matrix(1, 5, 1);
        expected.matrix = new Double[][]{{0.0, 1.0, 0.0, 0.0, 0.0}};
        Matrix testInput = new Matrix(1, 28900, 1);
        testInput.seed();

        Model model = new Model();
        model.layers.add(new ConvolutionLayer(10, 1, 3));
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(6));
        model.layers.add(new ConvolutionLayer(10, 1, 3));
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(6));
        model.layers.add(new Flatten());
        model.layers.add(new DenseLayer(256));
        model.layers.add(new ReLU());
        model.layers.add(new DenseLayer(5));
        model.layers.add(new Softmax());

        int epoch = 0;
        List<Matrix[]>batches = new ArrayList<>();
        batches.add(batch);
        List<Matrix[]> batchesExpected = new ArrayList<>();
        Matrix[] batchExpected = new Matrix[]{expected};
        batchesExpected.add(batchExpected);
        double avgLoss = Double.POSITIVE_INFINITY;
        for(int i = 0; i < batches.size(); i++){
            while(avgLoss > 0.01){
                avgLoss = 0;
                for(int j = 0; j < batches.get(i).length; j++){
                    avgLoss += model.forward(batches.get(i)[j], batchesExpected.get(i)[j]);
                    model.backward();
                }
                avgLoss /= batches.get(i).length; 
                for(Layer layer: model.layers){
                    layer.updateParams();
                }
                epoch++;
                System.out.println("Average Loss: " + avgLoss);
            }
            System.out.println("Completed in " + epoch + " epochs");
        }
        //ima smack you with my pimp cane
        //goofy ahh
    }
} 