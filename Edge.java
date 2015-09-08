package org.knime.knip.example;

import net.imglib2.type.numeric.IntegerType;

public class Edge<T extends IntegerType<T>, L extends Comparable<L>> implements Comparable<Edge<T, L>> {
        private final int n1x;
        private final int n1y;
        private final int n2x;
        private final int n2y;
        private final boolean vertical;
        int normal_weight;
        int weight = 0;
        Edge<T, L>[] neighbors;
        int number;
        boolean visited;
        boolean visitedPlateau;
        Pixel<T, L> p1;
        Pixel<T, L> p2;
        Edge<T, L> Fth = this;
        boolean Mrk;
        static boolean weights;

        /**
         * 
         * @param n1x
         *                X-Position of Node 1
         * @param n1y
         *                Y-Position of Node 1
         * @param n2x
         *                X-Position of Node 2
         * @param n2y
         *                Y-Position of Node 2
         * @param num
         */
        Edge(int n1x, int n1y, int n2x, int n2y, int num) {
                this.n1x = n1x;
                this.n1y = n1y;
                this.n2x = n2x;
                this.n2y = n2y;
                vertical = n2x == n1x;
                number = num;
                neighbors = new Edge[6];
        }

        Edge(Pixel<T, L> p1, Pixel<T, L> p2, int num) {
                this.n1x = p1.getX();
                this.n1y = p1.getY();
                this.n2x = p2.getX();
                this.n2y = p2.getY();
                vertical = n2x == n1x;
                number = num;
                neighbors = new Edge[6];
                this.p1 = p1;
                this.p2 = p2;
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

        int getN1x() {
                return n1x;
        }

        int getN1y() {
                return n1y;
        }

        int getN2x() {
                return n2x;
        }

        int getN2y() {
                return n2y;
        }

        boolean isVertical() {
                return vertical;
        }
}
