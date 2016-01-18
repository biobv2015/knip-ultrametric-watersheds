package net.imagej.ops.labeling.watershed;

import net.imglib2.type.numeric.RealType;

public class PseudoEdge<T extends RealType<T>, L extends Comparable<L>> implements Comparable<PseudoEdge<T, L>> {
    Pixel<T, L> p1;
    Pixel<T, L> p2;

    PseudoEdge(Pixel<T, L> p, Pixel<T, L> q) {
        p1 = p;
        p2 = q;
    }

    @Override
    public int compareTo(PseudoEdge<T, L> p) {
        if (p1.indic_VP < p.p1.indic_VP) {
            return -1;
        } else if (p1.indic_VP > p.p1.indic_VP) {
            return 1;
        } else {
            if (p2.indic_VP < p.p2.indic_VP) {
                return -1;
            } else if (p2.indic_VP > p.p2.indic_VP) {
                return 1;
            } else {
                return 0;
            }
        }
    }
}
