package org.knime.knip.example;

import net.imglib2.type.numeric.IntegerType;

public class Edge<T extends IntegerType<T>, L extends Comparable<L>> implements Comparable<Edge<T, L>> {
        final int normal_weight;
        int weight = 0;
        Edge<T, L>[] neighbors;
        boolean visited;
        final Pixel<T, L> p1;
        final Pixel<T, L> p2;
        Edge<T, L> Fth = this;
        boolean Mrk;

        static boolean weights;

        Edge(Pixel<T, L> p1, Pixel<T, L> p2, int normal_weight) {
                neighbors = new Edge[10];
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
                if (weights) {
                        if (weight < e.weight) {
                                return -1;
                        } else if (weight > e.weight) {
                                return 1;
                        } else {
                                return 0;
                        }
                }
                if (normal_weight < e.normal_weight) {
                        return -1;
                } else if (normal_weight > e.normal_weight) {
                        return 1;
                } else {
                        return 0;
                }
        }

        boolean isVertical() {
                return p2.getX() == p1.getX() && p1.getZ() == p2.getZ();
        }

        boolean isDepth() {
                return p2.getZ() != p1.getZ();
        }
}
