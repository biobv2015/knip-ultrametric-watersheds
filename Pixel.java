package net.imagej.ops.labeling.watershed;

import net.imglib2.type.numeric.RealType;

public class Pixel<T extends RealType<T>, L extends Comparable<L>> {
    private final long pointer;
    L label;
    int Rnk;
    Pixel<T, L> Fth = this;
    int indic_VP;
    protected static long[] dimensions;

    Pixel(long pointer, L label) {
        this.pointer = pointer;
        this.label = label;
    }

    Pixel<T, L> find() {
        if (Fth != this) {
            Fth = Fth.find();
        }
        return Fth;
    }

    int getX() {
        return (int) (pointer % dimensions[0]);
    }

    int getY() {
        return (int) Math.floorDiv((pointer % (dimensions[0] * dimensions[1])), dimensions[1]);
    }

    int getZ() {
        return (int) Math.floorDiv(pointer, (dimensions[0] * dimensions[1]));
    }

    int getPointer() {
        return (int) pointer;
    }
}
