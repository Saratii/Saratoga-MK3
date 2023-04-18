package src;

public class MathUtils {
	public static int ceilDiv(int x, int y) {
		return -Math.floorDiv(-x, y);
	}

	public static double sum(Double[] inputs) {
        double sum = 0;
        for(Double d: inputs){
            sum += d;
        }
        return sum;
    }

	public static double maxValue(Double[] inputs) {
		double max = inputs[0];
		for(double value : inputs){
			if(value > max){
				max = value;
			}
		}
		return max;
	}

	public static double minValue(Double[] inputs) {
		double min = inputs[0];
		for(double value : inputs){
			if(value < min){
				min = value;
			}
		}
		return min;
	}
}