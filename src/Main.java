package src;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static double ALPHA = 0.001;
    public static double batchSize = 30;
    public static void main(String[] args) throws IOException {
        
        List<Path> dolphinDirectory = Files.list(Path.of("Aminals/animals/dolphin")).toList();
        List<Path> antelopeDirectory = Files.list(Path.of("Aminals/animals/antelope")).toList();
        List<Image> images = new ArrayList<>();
        for(Path path: dolphinDirectory){
            images.add(new Image(path, "dolphin"));
        }
        for(Path path: antelopeDirectory){
            images.add(new Image(path, "antelope"));
        }
        Image.shuffle(images);
        Image[][] batches = new Image[4][30];
        for(int i = 0; i < batches.length; i++){
            for(int j = 0; j < batches[0].length; j++){
                batches[i][j] = images.get(i * batches.length + j);
            }
        }

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
        model.layers.add(new DenseLayer(2));
        model.layers.add(new Softmax());

        int epoch = 0;
        double avgLoss = Double.POSITIVE_INFINITY;
        for(int i = 0; i < batches.length; i++){
            while(avgLoss > 0.01){
                avgLoss = 0;
                for(int j = 0; j < batches[i].length; j++){
                    avgLoss += model.forward(batches[i][j].imageData, batches[i][j].label);
                    model.backward();
                }
                avgLoss /= batches[i].length; 
                for(Layer layer: model.layers){
                    layer.updateParams();
                }
                epoch++;
                System.out.println("Average Loss: " + avgLoss);
            }
            avgLoss = Double.POSITIVE_INFINITY;
            System.out.println("Completed in " + epoch + " epochs");
            epoch = 0;
        }
        // ima smack you with my pimp cane
        // goofy ahh
    }
} 