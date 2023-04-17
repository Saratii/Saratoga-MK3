package src;

import java.util.concurrent.atomic.AtomicReference;

public class AtomicDouble {
    private final AtomicReference<Double> value;

    public AtomicDouble(double initialValue) {
        value = new AtomicReference<>(initialValue);
    }

    public double get() {
        return value.get();
    }

    public void set(double newValue) {
        value.set(newValue);
    }

    public void add(double delta) {
        value.updateAndGet(oldValue -> oldValue + delta);
    }

    public void divide(double divisor) {
        value.updateAndGet(oldValue -> oldValue / divisor);
    }

    @Override
    public String toString() {
        return String.valueOf(value.get());
    }
}