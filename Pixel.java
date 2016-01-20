package net.imagej.ops.labeling.watershed;

import net.imglib2.type.numeric.RealType;

public class Pixel<T extends RealType<T>, L extends Comparable<L>> {
    private final long pointer;
    L label;
    int Rnk;
    Pixel<T, L> Fth = this;
    int indic_VP;
    protected static long[] dimensions;

    Pixel(long pointer) {
        this.pointer = pointer;
    }

    Pixel<T, L> find() {
        if (Fth != this) {
            Fth = Fth.find();
        }
        return Fth;
    }

    int getPointer() {
        return (int) pointer;
    }
}
