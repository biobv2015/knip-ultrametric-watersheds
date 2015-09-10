package org.knime.knip.example;

import net.imglib2.type.numeric.IntegerType;

public class Pixel<T extends IntegerType<T>, L extends Comparable<L>> {
        private final int x;
        private final int y;
        private final int z;
        L label;
        int Rnk;
        Pixel<T, L> Fth = this;
        int indic_VP;
        static int width;
        static int height;

        Pixel(int x, int y, int z, L label) {
                this.x = x;
                this.y = y;
                this.z = z;
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

        int getZ() {
                return z;
        }

        int getPointer() {
                return width * height * z + width * y + x;
        }
}
