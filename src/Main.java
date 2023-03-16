package src;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static double ALPHA = 0.001;
    public static double batchSize = 30;
    public static void main(String[] args) throws IOException {
        File folder = new File("logs");
        if(folder.exists()) {
            folder.delete();
        }
        folder.mkdirs();
        folder.close();
        PrintWriter writer = new PrintWriter("logs/log-graph", "UTF-8");
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
        model.layers.add(new ConvolutionLayer(5, 1, 3));
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(6));
        model.layers.add(new ConvolutionLayer(5, 1, 3));
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(6));
        model.layers.add(new Flatten());
        model.layers.add(new DenseLayer(16));
        model.layers.add(new ReLU());
        model.layers.add(new DenseLayer(2));
        model.layers.add(new Softmax());

        int epoch = 0;
        ALPHA /= batchSize;
        double avgLoss = Double.POSITIVE_INFINITY;
        // while(avgLoss > 0.1){
        for(int p = 0; p < 2; p++){
            avgLoss = 0;
            for(int i = 0; i < batches.length; i++){
                for(int j = 0; j < batches[i].length; j++){
                    avgLoss += model.forward(batches[i][j].imageData, batches[i][j].label);
                    model.backward();
                }
            }
            model.updateParams();
            avgLoss /= (batches[0].length * batches.length);
            
            System.out.println(ANSI_GREEN +"Average Loss: " + avgLoss + ANSI_RESET);
            writer.println(epoch + ", " + avgLoss);
            epoch++;

        }
        System.out.println(ANSI_CYAN + "Completed in " + epoch + " epochs" + ANSI_RESET);
        writer.close();
        model.write();
        // ima smack you with my pimp cane
        // goofy ahh
    }
} 