package org.knime.knip.example;

import net.imglib2.type.numeric.IntegerType;

public class Pixel<T extends IntegerType<T>, L extends Comparable<L>> {
        private final int x;
        private final int y;
        private final int pointer;
        Edge<T, L>[] neighbors;
        L label;
        int Rnk;
        Pixel<T, L> Fth = this;
        Pixel<T, L> indic_VP;
        Pixel<T, L> local_seed;

        Pixel(int x, int y, L label, int width) {
                this.x = x;
                this.y = y;
                this.label = label;
                this.pointer = x + width * y;
                neighbors = new Edge[4];
        }

        Pixel<T, L> find() {
                if (Fth != this) {
                        Fth = Fth.find();
                }
                return Fth;
        }

        int getX() {
                return x;
        }

        int getY() {
                return y;
        }

        int getPointer() {
                return pointer;
        }
}
