package org.knime.knip.example;

import net.imglib2.type.numeric.IntegerType;

public class Edge<T extends IntegerType<T>, L extends Comparable<L>> implements Comparable<Edge<T, L>> {
        int n1x;
        int n1y;
        int n2x;
        int n2y;
        int normal_weight;
        int weight;
        boolean vertical;
        Edge<T, L>[] neighbors;
        int number;
        boolean visited;
        boolean visitedPlateau;
        Pixel<T, L> p1;
        Pixel<T, L> p2;
        Edge<T, L> Fth = this;
        boolean Mrk;
        static boolean weights;

        Edge(int n1x, int n1y, int n2x, int n2y, int num) {
                this.n1x = n1x;
                this.n1y = n1y;
                this.n2x = n2x;
                this.n2y = n2y;
                vertical = n2x == n1x;
                number = num;
                neighbors = new Edge[6];
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
}
