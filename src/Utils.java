package src;

public class Utils {
    public static double[] DoubleTodouble(Double[] d){
        double[] dd = new double[d.length];
        for(int i = 0; i < d.length; i++){
            dd[i] = d[i];
        }
        return dd;
    }
}
