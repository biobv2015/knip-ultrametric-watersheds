package net.imagej.ops.labeling.watershed;

import net.imglib2.type.numeric.RealType;

public class Edge<T extends RealType<T>, L extends Comparable<L>> implements Comparable<Edge<T, L>> {
    final double normal_weight;
    protected static long[] dimensions;
    double weight = 0;
    Edge<T, L>[] neighbors;
    boolean visited;
    final Pixel<T, L> p1;
    final Pixel<T, L> p2;
    Edge<T, L> Fth = this;
    boolean Mrk;

    static boolean weights;
    static boolean ascending;

    Edge(Pixel<T, L> p1, Pixel<T, L> p2, double normal_weight) {
        neighbors = new Edge[dimensions.length * 4 - 2];
        this.p1 = p1;
        this.p2 = p2;
        this.normal_weight = normal_weight;
    }

    Edge<T, L> find() {
        if (Fth != this) {
            Fth = Fth.find();
        }
        return Fth;
    }

    @Override
    public int compareTo(Edge<T, L> e) {
        int result = 0;
        if (weights) {
            if (weight < e.weight) {
                result = -1;
            } else if (weight > e.weight) {
                result = 1;
            } else {
                result = 0;
            }
        } else {
            if (normal_weight < e.normal_weight) {
                result = -1;
            } else if (normal_weight > e.normal_weight) {
                result = 1;
            } else {
                result = 0;
            }
        }
        if (!ascending) {
            result *= -1;
        }
        return result;
    }

    boolean isVertical() {
        return Math.floorDiv((p1.getPointer() % (dimensions[0] * dimensions[1])), dimensions[1]) != Math
                .floorDiv((p2.getPointer() % (dimensions[0] * dimensions[1])), dimensions[1]);
    }

    boolean isDepth() {
        return Math.floorDiv(p1.getPointer(), (dimensions[0] * dimensions[1])) != Math.floorDiv(p2.getPointer(),
                (dimensions[0] * dimensions[1]));
    }
}
