package org.knime.knip.example;

import net.imglib2.type.numeric.IntegerType;

public class PseudoEdge<T extends IntegerType<T>, L extends Comparable<L>> implements Comparable<PseudoEdge<T, L>> {
        Pixel<T, L> p1;
        Pixel<T, L> p2;

        PseudoEdge(Pixel<T, L> p, Pixel<T, L> q) {
                p1 = p;
                p2 = q;
        }

        @Override
        public int compareTo(PseudoEdge<T, L> p) {
                if (p1.getPointer() < p.p1.getPointer()) {
                        return -1;
                } else if (p1.getPointer() > p.p1.getPointer()) {
                        return 1;
                } else {
                        if (p2.getPointer() < p.p2.getPointer()) {
                                return -1;
                        } else if (p2.getPointer() > p.p2.getPointer()) {
                                return 1;
                        } else {
                                return 0;
                        }
                }
        }
}
