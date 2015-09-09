package org.knime.knip.example;

import net.imglib2.type.numeric.IntegerType;

public class Pixel<T extends IntegerType<T>, L extends Comparable<L>> {
        private final int x;
        private final int y;
        L label;
        int Rnk;
        Pixel<T, L> Fth = this;
        int indic_VP;
        static int width;

        Pixel(int x, int y, L label) {
                this.x = x;
                this.y = y;
                this.label = label;
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
                return x + width * y;
        }
}
