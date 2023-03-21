package src;
import java.io.File;
import java.io.FileNotFoundException;
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
    public static double ALPHA = 0.0001;
    public static double batchSize = 30;
    public static Model model;
    public static void main(String[] args) throws IOException {
        List<Path> dolphinDirectory = Files.list(Path.of("Aminals/animals/dolphin")).toList();
        List<Path> antelopeDirectory = Files.list(Path.of("Aminals/animals/antelope")).toList();
        List<Path> testDolphins = Files.list(Path.of("images/testDolphins")).toList();
        List<Path> testAntelopes = Files.list(Path.of("images/testAntelopes")).toList();
        // train();
        
        for(Path p: testDolphins){
            Image im = new Image(p, "dolphin");
            classify(im);
            System.out.println("\n");
        }
        for(Path p: testAntelopes){
            Image im = new Image(p, "antelope");
            classify(im);
            System.out.println("\n");
        }
       
        // ima smack you with my pimp cane
        // goofy ahh
    }
    public static void train() throws IOException{
        File folder = new File("logs");
        if(folder.exists()) {
            String[] entries = folder.list();
            for(String s: entries){
                File currentFile = new File(folder.getPath(), s);
                currentFile.delete();
            }
        } else {
            folder.mkdirs();
        }
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
        model = new Model();
        model.layers.add(new ConvolutionLayer(15, 1, 3));
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(3));
        model.layers.add(new ConvolutionLayer(15, 1, 3));
        model.layers.add(new ReLU());
        model.layers.add(new MaxPool(3));
        model.layers.add(new Flatten());
        model.layers.add(new DenseLayer(128));
        model.layers.add(new ReLU());
        model.layers.add(new DenseLayer(2));
        model.layers.add(new Softmax());
        model.profiling = false;
        int epoch = 0;
        ALPHA /= batchSize;
        double avgLoss = Double.POSITIVE_INFINITY;
        while(avgLoss > 0.01){
        // for(int p = 0; p < 2; p++){
            avgLoss = 0;
            for(int i = 0; i < batches.length; i++){
                for(int j = 0; j < batches[i].length; j++){
                    avgLoss += model.forward(batches[i][j].imageData, batches[i][j].label);
                    model.backward();
                }
                model.updateParams();
            }
            avgLoss /= (batches[0].length * batches.length);
            System.out.println(ANSI_GREEN +"Average Loss: " + avgLoss + ANSI_RESET);
            writer.println(epoch + ", " + avgLoss);
            epoch++;

        }
        System.out.println(ANSI_CYAN + "Completed in " + epoch + " epochs" + ANSI_RESET);
        if(model.profiling){
            for(int i = 0; i < model.layers.size(); i++){
                System.out.println("Forward: " + model.layers.get(i).forwardTime);
                System.out.println("Backward: " + model.layers.get(i).backwardTime + "\n");
            }
        }
        writer.close();
        model.write();
    }
    public static void classify(Image im) throws FileNotFoundException, IOException{
        Model model2 = build.buildModel();
        
        model2.forward(im.imageData, im.label);
        Softmax soft = (Softmax) model2.layers.get(model2.layers.size() - 1);
        System.out.println("Predicted: "  + soft.result);
        System.out.println("Actual: " + im.label);
        int j = 0;
    }
}