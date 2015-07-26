package org.knime.knip.example;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Stack;

import org.scijava.ItemIO;
import org.scijava.plugin.Menu;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import Jama.Matrix;
import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import net.imagej.ImgPlus;
import net.imagej.ops.Op;
import net.imagej.ops.OpService;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.labeling.Labeling;
import net.imglib2.labeling.NativeImgLabeling;
import net.imglib2.type.numeric.IntegerType;

@Plugin(menu = {@Menu(label = "DeveloperPlugins"),
                @Menu(label = "PowerWatershed")}, description = "TODO", headless = true, type = Op.class, name = "PWSHED")
public class PWSHEDOP<T extends IntegerType<T>, L extends Comparable<L>> implements Op {

        class Pixel {
                int x;
                int y;
                int pointer;
                Edge[] neighbors;
                int label;

                int Rnk;
                PWSHEDOP<T, L>.Pixel Fth = this;
                boolean visited;
                PWSHEDOP<T, L>.Pixel indic_VP;
                PWSHEDOP<T, L>.Pixel local_seed;

                Pixel(int x, int y, int label) {
                        this.x = x;
                        this.y = y;
                        this.label = label;
                        this.pointer = x + width * y;
                        neighbors = new PWSHEDOP.Edge[4];
                }

                PWSHEDOP<T, L>.Pixel find() {
                        if (Fth != this) {
                                Fth = Fth.find();
                        }
                        return Fth;
                }

        }

        class PseudoEdge implements Comparable<PseudoEdge> {
                PWSHEDOP<T, L>.Pixel p1;
                PWSHEDOP<T, L>.Pixel p2;

                PseudoEdge(PWSHEDOP<T, L>.Pixel p, PWSHEDOP<T, L>.Pixel q) {
                        p1 = p;
                        p2 = q;
                }

                @Override
                public int compareTo(PseudoEdge p) {
                        if (p1.pointer < p.p1.pointer) {
                                return -1;
                        } else if (p1.pointer > p.p1.pointer) {
                                return 1;
                        } else {
                                if (p2.pointer < p.p2.pointer) {
                                        return -1;
                                } else if (p2.pointer > p.p2.pointer) {
                                        return 1;
                                } else {
                                        return 0;
                                }
                        }
                }
        }

        class Edge implements Comparable<Edge> {
                int n1x;
                int n1y;
                int n2x;
                int n2y;
                int normal_weight;
                int weight;
                boolean vertical;
                Edge[] neighbors;
                int number;
                boolean visited;
                boolean visitedPlateau;
                PWSHEDOP<T, L>.Pixel p1;
                PWSHEDOP<T, L>.Pixel p2;
                Edge Fth = this;
                boolean Mrk;

                Edge(int n1x, int n1y, int n2x, int n2y, int num) {
                        this.n1x = n1x;
                        this.n1y = n1y;
                        this.n2x = n2x;
                        this.n2y = n2y;
                        vertical = n2x == n1x;
                        number = num;
                        neighbors = new PWSHEDOP.Edge[6];
                }

                Edge find() {
                        if (Fth != this) {
                                Fth = Fth.find();
                        }
                        return Fth;
                }

                @Override
                public int compareTo(Edge e) {
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

        public static int SIZE_MAX_PLATEAU = 1000000;
        public static double epsilon = 0.000001;

        int[][] r_pixels;
        int[][] g_pixels;
        int[][] b_pixels;
        ArrayList<Edge> edges;
        int numOfEdges;
        int numOfPixels;
        ArrayList<Pixel> seedsL;
        int width;
        int height;
        ArrayList<Integer> labels;
        Pixel[] gPixels;
        Pixel[][] gPixelsT;
        float[][] proba;
        boolean weights = false;//sort egdges by weight

        @Parameter(type = ItemIO.OUTPUT)
        private ImagePlus output;

        @Parameter
        private ImgPlus<T> image_path;

        @Parameter
        private Labeling<L> seed_path;

        @Parameter
        private boolean color;

        @Parameter
        private OpService ops;

        @Override
        public void run() {

                final RandomAccess<T> imgRA = image_path.randomAccess();
                final Cursor<? extends IntegerType<?>> seedCursor = ((NativeImgLabeling) seed_path).getStorageImg().cursor();

                // save the width for easy access
                width = (int) image_path.dimension(0);
                // save the height for easy access
                height = (int) image_path.dimension(1);
                // save the number of edges for easy access
                numOfEdges = width * (height - 1) + (width - 1) * height;
                // save the number of pixels for easy access
                numOfPixels = height * width;
                // If the image is colored the pixels are split into their R, G and B
                // values.
                if (color) {
                        r_pixels = new int[width][height];
                        g_pixels = new int[width][height];
                        b_pixels = new int[width][height];
                        for (int y = 0; y < height; y++) {
                                for (int x = 0; x < width; x++) {
                                        //                                        r_pixels[x][y] = (image[x][y] & (255 << 16)) >> 16;
                                        //                                        g_pixels[x][y] = (image[x][y] & (255 << 8)) >> 8;
                                        //                                        b_pixels[x][y] = image[x][y] & 255;
                                }
                        }
                }
                seedsL = new ArrayList<>();
                labels = new ArrayList<Integer>();
                while (seedCursor.hasNext()) {
                        seedCursor.fwd();
                        if (seedCursor.get().getInteger() > 0) {
                                seedsL.add(new Pixel(seedCursor.getIntPosition(0), seedCursor.getIntPosition(1), seedCursor.get().getInteger()));
                                if (!labels.contains(seedCursor.get().getInteger())) {
                                        labels.add(seedCursor.get().getInteger());
                                }
                        }
                }
                {
                        gPixels = new PWSHEDOP.Pixel[numOfPixels];
                        gPixelsT = new PWSHEDOP.Pixel[width][height];
                        for (int i = 0; i < width; i++) {
                                for (int j = 0; j < height; j++) {
                                        gPixelsT[i][j] = new PWSHEDOP.Pixel(i, j, 0);
                                        gPixels[i + width * j] = gPixelsT[i][j];
                                }
                        }
                        Edge[][] hor_edges = new PWSHEDOP.Edge[width - 1][height];
                        Edge[][] ver_edges = new PWSHEDOP.Edge[width][height - 1];
                        edges = new ArrayList<>();
                        for (int i = 0; i < height - 1; i++) {
                                for (int j = 0; j < width; j++) {
                                        ver_edges[j][i] = new Edge(j, i, j, i + 1, edges.size());
                                        ver_edges[j][i].p1 = gPixelsT[j][i];
                                        ver_edges[j][i].p2 = gPixelsT[j][i + 1];
                                        edges.add(ver_edges[j][i]);
                                }
                        }
                        for (int i = 0; i < height; i++) {
                                for (int j = 0; j < width - 1; j++) {
                                        hor_edges[j][i] = new Edge(j, i, j + 1, i, edges.size());
                                        hor_edges[j][i].p1 = gPixelsT[j][i];
                                        hor_edges[j][i].p2 = gPixelsT[j + 1][i];
                                        edges.add(hor_edges[j][i]);
                                }
                        }
                        for (int i = 0; i < numOfEdges; i++) {
                                Edge e = edges.get(i);
                                if (!e.vertical) {
                                        if (e.n1y > 0) {
                                                e.neighbors[0] = ver_edges[e.n2x][e.n1y - 1];
                                                e.neighbors[1] = ver_edges[e.n1x][e.n1y - 1];
                                        }
                                        if (e.n1x > 0) {
                                                e.neighbors[2] = hor_edges[e.n1x - 1][e.n1y];
                                        }
                                        if (e.n1y < ver_edges[0].length - 1) {
                                                e.neighbors[3] = ver_edges[e.n1x][e.n1y];
                                                e.neighbors[4] = ver_edges[e.n2x][e.n1y];
                                        }
                                        if (e.n1x < hor_edges.length - 1) {
                                                e.neighbors[5] = hor_edges[e.n2x][e.n1y];
                                        }
                                } else { // vertical
                                        if (e.n1y > 0) {
                                                e.neighbors[0] = ver_edges[e.n1x][e.n1y - 1];
                                        }
                                        if (e.n1x > 0) {
                                                e.neighbors[1] = hor_edges[e.n1x - 1][e.n1y];
                                                e.neighbors[2] = hor_edges[e.n2x - 1][e.n2y];
                                        }
                                        if (e.n2y < ver_edges[0].length - 1) {
                                                e.neighbors[3] = ver_edges[e.n2x][e.n2y];
                                        }
                                        if (e.n1x < hor_edges.length - 1) {
                                                e.neighbors[4] = hor_edges[e.n2x][e.n2y];
                                                e.neighbors[5] = hor_edges[e.n1x][e.n1y];
                                        }
                                }
                        }
                        for (Pixel p : seedsL) {
                                if (p.x < hor_edges.length) {
                                        p.neighbors[0] = hor_edges[p.x][p.y];
                                }
                                if (p.y < ver_edges.length) {
                                        p.neighbors[1] = ver_edges[p.x][p.y];
                                }
                                if (p.x > 0) {
                                        p.neighbors[2] = hor_edges[p.x - 1][p.y];
                                }
                                if (p.y > 0) {
                                        p.neighbors[3] = ver_edges[p.x][p.y - 1];
                                }
                        }
                }
                if (color) {
                        for (Edge e : edges) {
                                int wr = Math.abs(r_pixels[e.n1x][e.n1y] - r_pixels[e.n2x][e.n2y]);
                                int wg = Math.abs(g_pixels[e.n1x][e.n1y] - g_pixels[e.n2x][e.n2y]);
                                int wb = Math.abs(b_pixels[e.n1x][e.n1y] - b_pixels[e.n2x][e.n2y]);
                                e.weight = 255 - wr;
                                if (255 - wg < e.weight) {
                                        e.weight = 255 - wg;
                                }
                                if (255 - wb < e.weight) {
                                        e.weight = 255 - wb;
                                }
                        }
                        for (Edge e : edges) {
                                e.normal_weight = e.weight;
                        }
                } else {
                        for (Edge e : edges) {
                                imgRA.setPosition(e.n1x, 0);
                                imgRA.setPosition(e.n1y, 1);
                                int v1 = imgRA.get().getInteger();
                                imgRA.setPosition(e.n2x, 0);
                                imgRA.setPosition(e.n2y, 1);
                                int v2 = imgRA.get().getInteger();
                                e.normal_weight = 255 - Math.abs(v1 - v2);
                        }
                }
                int[] seeds_function = new int[numOfEdges];
                for (Pixel p : seedsL) {
                        for (Edge e : p.neighbors) {
                                if (e != null) {
                                        seeds_function[e.number] = e.normal_weight;
                                }
                        }
                }
                for (Edge e : edges) {
                        e.weight = seeds_function[e.number];
                }
                gageodilate_union_find();
                int[] outputPixels = null;
                outputPixels = PowerWatershed_q2();
                output = new ImagePlus("mask.pgm", new ByteProcessor(new FloatProcessor(toTwoDim(outputPixels)), true));
        }

        private int[] PowerWatershed_q2() {
                proba = new float[labels.size() - 1][numOfPixels];
                for (int i = 0; i < labels.size() - 1; i++) {
                        for (int j = 0; j < numOfPixels; j++) {
                                proba[i][j] = -1;
                        }
                }
                PWSHEDOP<T, L>.Pixel[][] edgesLCP = new PWSHEDOP.Pixel[2][numOfEdges];
                // proba[i][j] =1 <=> pixel[i] has label j+1
                for (PWSHEDOP<T, L>.Pixel pix : seedsL) {
                        for (int j = 0; j < labels.size() - 1; j++) {
                                if (pix.label == j + 1) {
                                        proba[j][pix.pointer] = 1;
                                } else {
                                        proba[j][pix.pointer] = 0;
                                }
                        }
                }
                float[][] local_labels = new float[labels.size() - 1][numOfPixels];
                @SuppressWarnings("unchecked")
                ArrayList<PWSHEDOP<T, L>.Edge> sorted_weights = (ArrayList<PWSHEDOP<T, L>.Edge>) edges.clone();
                weights = true;
                Collections.sort(sorted_weights);
                Collections.reverse(sorted_weights);

                /* beginning of main loop */
                for (Edge e_max : sorted_weights) {
                        if (e_max.visited)
                                continue;

                        // 1. Computing the edges of the plateau LCP linked to the edge
                        // e_max
                        Stack<Edge> LIFO = new Stack<>();
                        Stack<Edge> LCP = new Stack<>();
                        ArrayList<Pixel> Plateau = new ArrayList<>(); // vertices of a
                                                                      // plateau.
                        LIFO.add(e_max);
                        e_max.visitedPlateau = true;
                        e_max.visited = true;
                        LCP.add(e_max);
                        int nb_edges = 0;
                        int wmax = e_max.weight;
                        PWSHEDOP<T, L>.Edge x;
                        ArrayList<PWSHEDOP<T, L>.Edge> sorted_weights2 = new ArrayList<PWSHEDOP<T, L>.Edge>();

                        // 2. putting the edges and vertices of the plateau into arrays
                        while (!LIFO.empty()) {
                                x = LIFO.pop();
                                PWSHEDOP<T, L>.Pixel re1 = x.p1.find();
                                PWSHEDOP<T, L>.Pixel re2 = x.p2.find();
                                if (proba[0][re1.pointer] < 0 || proba[0][re2.pointer] < 0) {
                                        if (!x.p1.visited) {
                                                Plateau.add(x.p1);
                                                x.p1.visited = true;
                                        }
                                        if (!x.p2.visited) {
                                                Plateau.add(x.p2);
                                                x.p2.visited = true;
                                        }
                                        edgesLCP[0][nb_edges] = x.p1;
                                        edgesLCP[1][nb_edges] = x.p2;
                                        sorted_weights2.add(x);
                                        nb_edges++;
                                }

                                for (PWSHEDOP<T, L>.Edge edge : x.neighbors) {
                                        if (edge != null) {
                                                if ((!edge.visitedPlateau) && (edge.weight == wmax)) {
                                                        edge.visitedPlateau = true;
                                                        LIFO.add(edge);
                                                        LCP.add(edge);
                                                        edge.visited = true;
                                                }
                                        }
                                }
                        }

                        for (Pixel p : Plateau)
                                p.visited = false;
                        for (Edge e : LCP)
                                e.visitedPlateau = false;

                        // 3. If e_max belongs to a plateau
                        if (nb_edges > 0) {
                                // 4. Evaluate if there are differents seeds on the plateau
                                boolean different_seeds = false;

                                for (int i = 0; i < labels.size() - 1; i++) {
                                        int p = 0;
                                        double val = -0.5;
                                        for (Pixel j : Plateau) {
                                                int xr = j.find().pointer;
                                                if (Math.abs(proba[i][xr] - val) > epsilon && proba[i][xr] >= 0) {
                                                        p++;
                                                        val = proba[i][xr];
                                                        if (p == 2)
                                                                break;
                                                }
                                        }
                                        if (p >= 2) {
                                                different_seeds = true;
                                                break;
                                        }
                                }

                                if (different_seeds == true) {
                                        // 5. Sort the edges of the plateau according to their
                                        // normal weight
                                        weights = false;
                                        Collections.sort(sorted_weights2);
                                        Collections.reverse(sorted_weights2);

                                        // Merge nodes for edges of real max weight
                                        Plateau.clear();
                                        int Nnb_edges = 0;
                                        for (PWSHEDOP<T, L>.Edge Ne_max : sorted_weights2) {
                                                PWSHEDOP<T, L>.Pixel re1 = Ne_max.p1.find();
                                                PWSHEDOP<T, L>.Pixel re2 = Ne_max.p2.find();
                                                if (Ne_max.normal_weight != wmax) {
                                                        merge_node(re1, re2);
                                                } else {
                                                        if ((re1 != re2) && ((proba[0][re1.pointer] < 0 || proba[0][re2.pointer] < 0))) {
                                                                if (!re1.visited) {
                                                                        Plateau.add(re1);
                                                                        re1.visited = true;
                                                                }
                                                                if (!re2.visited) {
                                                                        Plateau.add(re2);
                                                                        re2.visited = true;
                                                                }
                                                                edgesLCP[0][Nnb_edges] = re1;
                                                                edgesLCP[1][Nnb_edges] = re2;
                                                                Nnb_edges++;
                                                        }
                                                }
                                        }

                                        int k = 0;
                                        for (int i = 0; i < labels.size() - 1; i++) {
                                                k = 0;
                                                for (Pixel xr : Plateau) {
                                                        if (proba[i][xr.pointer] >= 0) {
                                                                local_labels[i][k] = proba[i][xr.pointer];
                                                                gPixels[k].local_seed = xr;
                                                                k++;
                                                        }
                                                }
                                        }

                                        // 6. Execute Random Walker on plateaus
                                        if (Plateau.size() < SIZE_MAX_PLATEAU) {
                                                RandomWalker(edgesLCP, Nnb_edges, Plateau, local_labels, k);
                                        } else {
                                                System.out.printf("Plateau too big (%d vertices,%d edges), RW is not performed\n", Plateau.size(),
                                                                Nnb_edges);
                                                for (int j = 0; j < Nnb_edges; j++) {
                                                        Pixel e1 = edgesLCP[0][j];
                                                        Pixel e2 = edgesLCP[1][j];
                                                        merge_node(e1.find(), e2.find());
                                                }
                                        }

                                        for (Pixel pix : Plateau)
                                                pix.visited = false;
                                } else // if different seeds = false
                                       // 7. Merge nodes for edges of max weight
                                {
                                        for (int j = 0; j < nb_edges; j++) {
                                                Pixel e1 = edgesLCP[0][j];
                                                Pixel e2 = edgesLCP[1][j];
                                                merge_node(e1.find(), e2.find());
                                        }
                                }
                        }
                        LCP.clear();
                } // end main loop

                // building the final proba map (find the root vertex of each tree)
                for (Pixel j : gPixels) {
                        Pixel i = j;
                        Pixel xr = i.Fth;
                        while (i.Fth != i) {
                                i = xr;
                                xr = i.Fth;
                        }
                        for (int k = 0; k < labels.size() - 1; k++)
                                proba[k][j.pointer] = proba[k][i.pointer];
                }

                // writing results
                int[] Temp = new int[width * height];
                for (int j = 0; j < numOfPixels; j++) {
                        double maxi = 0;
                        int argmax = 0;
                        double val = 1;
                        for (int k = 0; k < labels.size() - 1; k++) {
                                if (proba[k][j] > maxi) {
                                        maxi = proba[k][j];
                                        argmax = k;
                                }
                                val = val - proba[k][j];

                        }
                        if (val > maxi)
                                argmax = labels.size() - 1;
                        Temp[j] = ((argmax) * 255) / (labels.size() - 1);
                }

                return Temp;
        }

        static class cs {

                static class css {
                        int[] pinv;
                        int[] q;
                        int[] parent;
                        int[] cp;
                        int[] leftmost;
                        int m2;
                        double lnz;
                        double unz;
                }

                static class csn {
                        cs L;
                        cs U;
                        int[] pinv;
                        double[] B;
                }

                public int nz; // # of entries in triplet matrix, -1 for compressed-col
                public int m; // number of rows
                public int n; // number of columns
                public double[] x; // numerical values, size nzmax
                public int[] p; // column pointers (size n+1) or col indices (size
                                // nzmax)
                public int[] i; // row indices, size nzmax
                public int nzmax; // maximum number of entries

                public cs(int m, int n, int nzmax, int values, boolean triplet) {
                        this.m = m; /* define dimensions and nzmax */
                        this.n = n;
                        this.nzmax = nzmax > 1 ? nzmax : 1;
                        this.nz = triplet ? 0 : -1; /* allocate triplet or comp.col */
                        this.p = new int[triplet ? this.nzmax : n + 1];
                        this.i = new int[this.nzmax];
                        this.x = values != 0 ? new double[this.nzmax] : null;
                }

                public static cs cs_compress(cs T) {
                        int m, n, nz, p, k;
                        int[] Cp, Ci, w, Ti, Tj;
                        double[] Cx, Tx;
                        cs C;
                        if (!(T != null && (T.nz >= 0)))
                                return (null); /* check inputs */
                        m = T.m;
                        n = T.n;
                        Ti = T.i;
                        Tj = T.p;
                        Tx = T.x;
                        nz = T.nz;
                        C = new cs(m, n, nz, (Tx != null ? 1 : 0), false); // allocate
                                                                           // result
                        w = new int[n]; /* get workspace */
                        Cp = C.p;
                        Ci = C.i;
                        Cx = C.x;
                        for (k = 0; k < nz; k++)
                                w[Tj[k]]++; /* column counts */
                        cs_cumsum(Cp, w, n); /* column pointers */
                        for (k = 0; k < nz; k++) {
                                Ci[p = w[Tj[k]]++] = Ti[k]; /* A(i,j) is the pth entry in C */
                                if (Cx != null)
                                        Cx[p] = Tx[k];
                        }
                        return C; /* success; free w and return C */
                }

                private static double cs_cumsum(int[] p, int[] c, int n) {
                        int i, nz = 0;
                        double nz2 = 0;
                        if (p == null || c == null)
                                return (-1); /* check inputs */
                        for (i = 0; i < n; i++) {
                                p[i] = nz;
                                nz += c[i];
                                nz2 += c[i]; /* also in double to avoid int overflow */
                                c[i] = p[i]; /* also copy p[0..n-1] back into c[0..n-1] */
                        }
                        p[n] = nz;
                        return (nz2); /* return sum (c [0..n-1]) */
                }

                public static cs cs_multiply(cs A, cs B) {
                        int p, j, nz = 0, anz, m, n, bnz;
                        boolean values;
                        int[] Cp, Ci, Bp, w, Bi;
                        double[] x, Bx, Cx;
                        cs C;
                        if (!(A != null && (A.nz == -1)) || !(B != null && (B.nz == -1)))
                                return (null); /* check inputs */
                        if (A.n != B.m)
                                return (null);
                        m = A.m;
                        anz = A.p[A.n];
                        n = B.n;
                        Bp = B.p;
                        Bi = B.i;
                        Bx = B.x;
                        bnz = Bp[n];
                        w = new int[m]; /* get workspace */
                        values = (A.x != null) && (Bx != null);
                        x = values ? new double[m] : null; /* get workspace */
                        C = new cs(m, n, anz + bnz, values ? 1 : 0, false); // allocate
                                                                            // result
                        if (values && x == null)
                                return null;
                        Cp = C.p;
                        for (j = 0; j < n; j++) {
                                if (nz + m > C.nzmax && !cs_sprealloc(C, 2 * (C.nzmax) + m)) {
                                        return null; /* out of memory */
                                }
                                Ci = C.i;
                                Cx = C.x; /* C.i and C.x may be reallocated */
                                Cp[j] = nz; /* column j of C starts here */
                                for (p = Bp[j]; p < Bp[j + 1]; p++) {
                                        nz = cs_scatter(A, Bi[p], Bx != null ? Bx[p] : 1, w, x, j + 1, C, nz);
                                }
                                if (values)
                                        for (p = Cp[j]; p < nz; p++)
                                                Cx[p] = x[Ci[p]];
                        }
                        Cp[n] = nz; /* finalize the last column of C */
                        cs_sprealloc(C, 0); /* remove extra space from C */
                        return C; /* success; free workspace, return C */
                }

                private static int cs_scatter(cs A, int j, double beta, int[] w, double[] x, int mark, cs C, int nzIN) {
                        int nz = nzIN;
                        int i, p;
                        int[] Ap, Ai, Ci;
                        double[] Ax;
                        if (!(A != null && (A.nz == -1)) || w == null || !(C != null && (C.nz == -1)))
                                return (-1); /* check inputs */
                        Ap = A.p;
                        Ai = A.i;
                        Ax = A.x;
                        Ci = C.i;
                        for (p = Ap[j]; p < Ap[j + 1]; p++) {
                                i = Ai[p]; /* A(i,j) is nonzero */
                                if (w[i] < mark) {
                                        w[i] = mark; /* i is new entry in column j */
                                        Ci[nz++] = i; /* add i to pattern of C(:,j) */
                                        if (x != null)
                                                x[i] = beta * Ax[p]; /* x(i) = beta*A(i,j) */
                                } else if (x != null)
                                        x[i] += beta * Ax[p]; /* i exists in C(:,j) already */
                        }
                        return (nz);
                }

                private static boolean cs_sprealloc(cs A, int nzmaxIN) {
                        int nzmax = nzmaxIN;
                        if (A == null)
                                return false;
                        if (nzmax <= 0)
                                nzmax = (A.nz == -1) ? (A.p[A.n]) : A.nz;
                        A.i = cs_realloc(A.i, nzmax);
                        if (A.nz >= 0)
                                A.p = cs_realloc(A.p, nzmax);
                        if (A.x != null)
                                A.x = cs_realloc(A.x, nzmax);
                        A.nzmax = nzmax;
                        return true;
                }

                private static int[] cs_realloc(int[] p, int n) {
                        int[] out = new int[n];
                        for (int i = 0; i < (n < p.length ? n : p.length); i++) {
                                out[i] = p[i];
                        }
                        return out;
                }

                private static double[] cs_realloc(double[] p, int n) {
                        double[] out = new double[n];
                        for (int i = 0; i < (n < p.length ? n : p.length); i++) {
                                out[i] = p[i];
                        }
                        return out;
                }

                boolean cs_lusol(int order, double[] b, double tol) {
                        if (nz != -1 || b == null)
                                return false; /* check inputs */
                        css S = cs_sqr(order, this); /* ordering and symbolic analysis */
                        csn N = cs_lu(this, S, tol); /* numeric LU factorization */
                        double[] y = new double[n];/* get workspace */
                        boolean ok = (S != null && N != null);
                        if (ok) {
                                /* x = b(p) */
                                {
                                        /* cs_ipvec(N.pinv, b, y, n); */
                                        for (int k = 0; k < n; k++)
                                                y[N.pinv != null ? N.pinv[k] : k] = b[k];
                                }
                                /* x = L\x */
                                {
                                        /* N.L.cs_lsolve(y); */
                                        for (int j = 0; j < N.L.n; j++) {
                                                y[j] /= N.L.x[N.L.p[j]];
                                                for (int k = N.L.p[j] + 1; k < N.L.p[j + 1]; k++) {
                                                        y[N.L.i[k]] -= N.L.x[k] * y[j];
                                                }
                                        }
                                }
                                /* x = U\x */
                                {
                                        /* N.U.cs_usolve(x); */
                                        for (int k = N.U.n - 1; k >= 0; k--) {
                                                y[k] /= N.U.x[N.U.p[k + 1] - 1];
                                                for (int j = N.U.p[k]; j < N.U.p[k + 1] - 1; j++) {
                                                        y[N.U.i[j]] -= N.U.x[j] * y[k];
                                                }
                                        }
                                }
                                /* b(q) = x */
                                {
                                        /* cs_ipvec(S.q, y, b, n); */
                                        for (int k = 0; k < n; k++)
                                                b[S.q != null ? S.q[k] : k] = y[k];
                                }
                        }
                        return ok;
                }

                private static csn cs_lu(cs A, css S, double tol) {
                        cs L, U;
                        csn N;
                        double pivot, a, t;
                        double[] Lx, Ux, x;
                        int n, ipiv, k, top, p, i, col, lnz, unz;
                        int[] Lp, Li, Up, Ui, pinv, xi, q;
                        if (!(A != null && A.nz == -1) || S == null)
                                return (null); /* check inputs */
                        n = A.n;
                        q = S.q;
                        lnz = (int) S.lnz;
                        unz = (int) S.unz;
                        x = new double[n];/* get double workspace */
                        xi = new int[2 * n];/* get int workspace */
                        N = new csn();
                        ; /* allocate result */
                        N.L = L = new cs(n, n, lnz, 1, false); // allocate result L
                        N.U = U = new cs(n, n, unz, 1, false); // allocate result U
                        N.pinv = pinv = new int[n];/* allocate result pinv */
                        Lp = L.p;
                        Up = U.p;
                        for (i = 0; i < n; i++)
                                x[i] = 0; /* clear workspace */
                        for (i = 0; i < n; i++)
                                pinv[i] = -1; /* no rows pivotal yet */
                        for (k = 0; k <= n; k++)
                                Lp[k] = 0; /* no cols of L yet */
                        lnz = unz = 0;
                        for (k = 0; k < n; k++) /* compute L(:,k) and U(:,k) */
                        {
                                /*
                                 * --- Triangular solve
                                 * ---------------------------------------------
                                 */
                                Lp[k] = lnz; /* L(:,k) starts here */
                                Up[k] = unz; /* U(:,k) starts here */
                                if ((lnz + n > L.nzmax && !cs_sprealloc(L, 2 * L.nzmax + n))
                                                || (unz + n > U.nzmax && !cs_sprealloc(U, 2 * U.nzmax + n))) {
                                        return null;
                                }
                                Li = L.i;
                                Lx = L.x;
                                Ui = U.i;
                                Ux = U.x;
                                col = q != null ? (q[k]) : k;
                                top = cs_spsolve(L, A, col, xi, x, pinv, true); /* x = L\A(:,col) */
                                /*
                                 * --- Find pivot
                                 * ---------------------------------------------------
                                 */
                                ipiv = -1;
                                a = -1;
                                for (p = top; p < n; p++) {
                                        i = xi[p]; /* x(i) is nonzero */
                                        if (pinv[i] < 0) /* row i is not yet pivotal */
                                        {
                                                if ((t = Math.abs(x[i])) > a) {
                                                        a = t; /* largest pivot candidate so far */
                                                        ipiv = i;
                                                }
                                        } else /* x(i) is the entry U(pinv[i],k) */
                                        {
                                                Ui[unz] = pinv[i];
                                                Ux[unz++] = x[i];
                                        }
                                }
                                if (ipiv == -1 || a <= 0)
                                        return null;
                                if (pinv[col] < 0 && Math.abs(x[col]) >= a * tol)
                                        ipiv = col;
                                /*
                                 * --- Divide by pivot
                                 * ----------------------------------------------
                                 */
                                pivot = x[ipiv]; /* the chosen pivot */
                                Ui[unz] = k; /* last entry in U(:,k) is U(k,k) */
                                Ux[unz++] = pivot;
                                pinv[ipiv] = k; /* ipiv is the kth pivot row */
                                Li[lnz] = ipiv; /* first entry in L(:,k) is L(k,k) = 1 */
                                Lx[lnz++] = 1;
                                for (p = top; p < n; p++) /* L(k+1:n,k) = x / pivot */
                                {
                                        i = xi[p];
                                        if (pinv[i] < 0) /* x(i) is an entry in L(:,k) */
                                        {
                                                Li[lnz] = i; /* save unpermuted row in L */
                                                Lx[lnz++] = x[i] / pivot; /* scale pivot column */
                                        }
                                        x[i] = 0; /* x [0..n-1] = 0 for next k */
                                }
                        }
                        /*
                         * --- Finalize L and U
                         * -------------------------------------------------
                         */
                        Lp[n] = lnz;
                        Up[n] = unz;
                        Li = L.i; /* fix row indices of L for final pinv */
                        for (p = 0; p < lnz; p++)
                                Li[p] = pinv[Li[p]];
                        cs_sprealloc(L, 0); /* remove extra space from L and U */
                        cs_sprealloc(U, 0);
                        return N; /* success */
                }

                private static int cs_spsolve(cs G, cs B, int k, int[] xi, double[] x, int[] pinv, boolean lo) {
                        int j, J, p, q, px, top, n;
                        int[] Gp, Gi, Bp, Bi;
                        double[] Gx, Bx;
                        if (!(G != null && G.nz == -1) || !(B != null & B.nz == -1) || xi == null || x == null)
                                return (-1);
                        Gp = G.p;
                        Gi = G.i;
                        Gx = G.x;
                        n = G.n;
                        Bp = B.p;
                        Bi = B.i;
                        Bx = B.x;
                        top = cs_reach(G, B, k, xi, pinv); /* xi[top..n-1]=Reach(B(:,k)) */
                        for (p = top; p < n; p++)
                                x[xi[p]] = 0; /* clear x */
                        for (p = Bp[k]; p < Bp[k + 1]; p++)
                                x[Bi[p]] = Bx[p]; /* scatter B */
                        for (px = top; px < n; px++) {
                                j = xi[px]; /* x(j) is nonzero */
                                J = pinv != null ? (pinv[j]) : j; /* j maps to col J of G */
                                if (J < 0)
                                        continue; /* column J is empty */
                                x[j] /= Gx[lo ? (Gp[J]) : (Gp[J + 1] - 1)];/* x(j) /= G(j,j) */
                                p = lo ? (Gp[J] + 1) : (Gp[J]); /* lo: L(j,j) 1st entry */
                                q = lo ? (Gp[J + 1]) : (Gp[J + 1] - 1); /* up: U(j,j) last entry */
                                for (; p < q; p++) {
                                        x[Gi[p]] -= Gx[p] * x[j]; /* x(i) -= G(i,j) * x(j) */
                                }
                        }
                        return (top); /* return top of stack */
                }

                private static int cs_reach(cs G, cs B, int k, int[] xi, int[] pinv) {
                        int p, n, top;
                        int[] Bp, Bi, Gp;
                        if (!(G != null & G.nz == -1) || !(B != null && B.nz == -1) || xi == null)
                                return (-1); /* check inputs */
                        n = G.n;
                        Bp = B.p;
                        Bi = B.i;
                        Gp = G.p;
                        top = n;
                        for (p = Bp[k]; p < Bp[k + 1]; p++) {
                                if (!(Gp[Bi[p]] < 0)) /* start a dfs at unmarked node i */
                                {
                                        top = cs_dfs(Bi[p], G, top, xi, n, pinv);
                                }
                        }
                        for (p = top; p < n; p++)
                                Gp[xi[p]] = (-(Gp[xi[p]]) - 2); /* restore G */

                        return (top);
                }

                private static int cs_dfs(int jIN, cs G, int topIN, int[] xi, int pstack, int[] pinv) {
                        int top = topIN;
                        int j = jIN;
                        int i, p, p2, jnew, head = 0;
                        boolean done;
                        int[] Gp, Gi;
                        if (!(G != null && (G.nz == -1)) || xi == null)
                                return (-1); /* check inputs */
                        Gp = G.p;
                        Gi = G.i;
                        xi[0] = j; /* initialize the recursion stack */
                        while (head >= 0) {
                                j = xi[head]; /* get j from the top of the recursion stack */
                                jnew = pinv != null ? (pinv[j]) : j;
                                if (!(Gp[j] < 0)) {
                                        Gp[j] = (-(Gp[j]) - 2); /* mark node j as visited */
                                        xi[pstack + head] = (jnew < 0) ? 0 : (((Gp[jnew]) < 0) ? (-(Gp[jnew]) - 2) : (Gp[jnew]));
                                }
                                done = true; /* node j done if no unvisited neighbors */
                                p2 = (jnew < 0) ? 0 : (((Gp[jnew + 1]) < 0) ? (-(Gp[jnew + 1]) - 2) : (Gp[jnew + 1]));
                                for (p = xi[pstack + head]; p < p2; p++) /* examine all neighbors of j */
                                {
                                        i = Gi[p]; /* consider neighbor node i */
                                        if ((Gp[i] < 0))
                                                continue; /* skip visited node i */
                                        xi[pstack + head] = p; /*
                                                                * pause depth-first search of node
                                                                * j
                                                                */
                                        xi[++head] = i; /* start dfs at node i */
                                        done = false; /* node j is not done */
                                        break; /* break, to start dfs (i) */
                                }
                                if (done) /* depth-first search at node j is done */
                                {
                                        head--; /* remove j from the recursion stack */
                                        xi[--top] = j; /* and place in the output stack */
                                }
                        }
                        return (top);
                }

                private static css cs_sqr(int order, cs A) {
                        css S;
                        if (!(A != null && (A.nz == -1)))
                                return null; /* check inputs */
                        int n = A.n;
                        S = new css(); /* allocate result S */
                        S.q = cs_amd(order, A); /* fill-reducing ordering */
                        if (order != 0 && S.q == null)
                                return null;
                        S.unz = 4 * (A.p[n]) + n; /* for LU factorization only, */
                        S.lnz = S.unz; /* guess nnz(L) and nnz(U) */
                        return S; /* return result S */
                }

                private static cs cs_transpose(cs A, boolean values) {
                        int p, q, j, n, m;
                        int[] Cp, Ci, Ap, Ai, w;
                        double[] Cx, Ax;
                        cs C;
                        if (!(A != null && (A.nz == -1)))
                                return (null); /* check inputs */
                        m = A.m;
                        n = A.n;
                        Ap = A.p;
                        Ai = A.i;
                        Ax = A.x;
                        C = new cs(n, m, Ap[n], (values && Ax != null) ? 1 : 0, false); // allocate
                                                                                        // result
                        w = new int[m]; /* get workspace */
                        Cp = C.p;
                        Ci = C.i;
                        Cx = C.x;
                        for (p = 0; p < Ap[n]; p++)
                                w[Ai[p]]++; /* row counts */
                        cs_cumsum(Cp, w, m); /* row pointers */
                        for (j = 0; j < n; j++) {
                                for (p = Ap[j]; p < Ap[j + 1]; p++) {
                                        Ci[q = w[Ai[p]]++] = j; /* place A(i,j) as entry C(j,i) */
                                        if (Cx != null)
                                                Cx[q] = Ax[p];
                                }
                        }
                        return C; /* success; free w and return C */
                }

                private static int cs_tdfs(int j, int kIN, int head, int next, int[] post, int stack, int[] w) {
                        int k = kIN;
                        int i, p, top = 0;
                        if (post == null)
                                return (-1); /* check inputs */
                        w[stack + 0] = j; /* place j on the stack */
                        while (top >= 0) /* while (stack is not empty) */
                        {
                                p = w[stack + top]; /* p = top of stack */
                                i = w[head + p]; /* i = youngest child of p */
                                if (i == -1) {
                                        top--; /* p has no unordered children left */
                                        post[k++] = p; /* node p is the kth postordered node */
                                } else {
                                        w[head + p] = w[next + i]; /* remove i from children of p */
                                        w[stack + (++top)] = i; /* start dfs on child node i */
                                }
                        }
                        return (k);
                }

                private static int[] cs_amd(int order, cs A) {
                        cs C, A2, AT;
                        int[] Cp, Ci, last, W, len, P, ATp, ATi;
                        int d, dk, dext, lemax = 0, e, elenk, eln, i, j, k, k1, k2, k3, jlast, ln, dense, nzmax, mindeg = 0, nvi, nvj, nvk, mark,
                                        wnvi, cnz, nel = 0, p, p1, p2, p3, p4, pj, pk, pk1, pk2, pn, q, n, m, t, h, nv, next, head, elen, degree, w,
                                        hhead;
                        boolean ok;
                        /*
                         * --- Construct matrix C
                         * -----------------------------------------------
                         */
                        if (!(A != null && (A.nz == -1)) || order <= 0 || order > 3)
                                return (null); /* check */
                        AT = cs_transpose(A, false); /* compute A' */
                        if (AT == null)
                                return (null);
                        m = A.m;
                        n = A.n;
                        dense = (int) (((16) > (10 * Math.sqrt(n))) ? (16) : (10 * Math.sqrt(n))); /* find dense threshold */
                        dense = (((n - 2) < (dense)) ? (n - 2) : (dense));
                        if (order == 1 && n == m) {
                                C = cs_add(A, AT, 0, 0); /* C = A+A' */
                        } else if (order == 2) {
                                ATp = AT.p; /* drop dense columns from AT */
                                ATi = AT.i;
                                for (p2 = 0, j = 0; j < m; j++) {
                                        p = ATp[j]; /* column j of AT starts here */
                                        ATp[j] = p2; /* new column j starts here */
                                        if (ATp[j + 1] - p > dense)
                                                continue; /* skip dense col j */
                                        for (; p < ATp[j + 1]; p++)
                                                ATi[p2++] = ATi[p];
                                }
                                ATp[m] = p2; /* finalize AT */
                                A2 = cs_transpose(AT, false); /* A2 = AT' */
                                C = A2 != null ? cs_multiply(AT, A2) : null; /* C=A'*A with no dense rows */
                        } else {
                                C = cs_multiply(AT, A); /* C=A'*A */
                        }
                        if (C == null)
                                return (null);
                        cs_fkeep(C); /* drop diagonal entries */
                        Cp = C.p;
                        cnz = Cp[n];
                        P = new int[n + 1]; /* allocate result */
                        W = new int[8 * (n + 1)]; /* get workspace */
                        t = cnz + cnz / 5 + 2 * n; /* add elbow room to C */
                        if (!cs_sprealloc(C, t))
                                return null;
                        len = W;
                        nv = (n + 1);// on W
                        next = 2 * (n + 1);// on W
                        head = 3 * (n + 1);// on W
                        elen = 4 * (n + 1);// on W
                        degree = 5 * (n + 1);// on W
                        w = 6 * (n + 1);// on W
                        hhead = 7 * (n + 1);// on W
                        last = P; /* use P as workspace for last */
                        /*
                         * --- Initialize quotient graph
                         * ----------------------------------------
                         */
                        for (k = 0; k < n; k++)
                                len[k] = Cp[k + 1] - Cp[k];
                        len[n] = 0;
                        nzmax = C.nzmax;
                        Ci = C.i;
                        for (i = 0; i <= n; i++) {
                                W[head + i] = -1; /* degree list i is empty */
                                last[i] = -1;
                                W[next + i] = -1;
                                W[hhead + i] = -1; /* hash list i is empty */
                                W[nv + i] = 1; /* node i is just one node */
                                W[w + i] = 1; /* node i is alive */
                                W[elen + i] = 0; /* Ek of node i is empty */
                                W[degree + i] = len[i]; /* degree of node i */
                        }
                        mark = cs_wclear(0, 0, w, n, W); /* clear w */
                        W[elen + n] = -2; /* n is a dead element */
                        Cp[n] = -1; /* n is a root of assembly tree */
                        W[w + n] = 0; /* n is a dead element */
                        /*
                         * --- Initialize degree lists
                         * ------------------------------------------
                         */
                        for (i = 0; i < n; i++) {
                                d = W[degree + i];
                                if (d == 0) /* node i is empty */
                                {
                                        W[elen + i] = -2; /* element i is dead */
                                        nel++;
                                        Cp[i] = -1; /* i is a root of assembly tree */
                                        W[w + i] = 0;
                                } else if (d > dense) /* node i is dense */
                                {
                                        W[nv + i] = 0; /* absorb i into element n */
                                        W[elen + i] = -1; /* node i is dead */
                                        nel++;
                                        Cp[i] = (-(n) - 2);
                                        W[nv + n]++;
                                } else {
                                        if (W[head + d] != -1)
                                                last[W[head + d]] = i;
                                        W[next + i] = W[head + d]; /* put node i in degree list d */
                                        W[head + d] = i;
                                }
                        }
                        while (nel < n) /* while (selecting pivots) do */
                        {
                                /*
                                 * --- Select node of minimum approximate degree
                                 * --------------------
                                 */
                                for (k = -1; mindeg < n && (k = W[head + mindeg]) == -1; mindeg++)
                                        ;
                                if (W[next + k] != -1)
                                        last[W[next + k]] = -1;
                                W[head + mindeg] = W[next + k]; /* remove k from degree list */
                                elenk = W[elen + k]; /* elenk = |Ek| */
                                nvk = W[nv + k]; /* # of nodes k represents */
                                nel += nvk; /* W[nv+k] nodes of A eliminated */
                                /*
                                 * --- Garbage collection
                                 * -------------------------------------------
                                 */
                                if (elenk > 0 && cnz + mindeg >= nzmax) {
                                        for (j = 0; j < n; j++) {
                                                if ((p = Cp[j]) >= 0) /* j is a live node or element */
                                                {
                                                        Cp[j] = Ci[p]; /* save first entry of object */
                                                        Ci[p] = (-(j) - 2); /* first entry is now CS_FLIP(j) */
                                                }
                                        }
                                        for (q = 0, p = 0; p < cnz;) /* scan all of memory */
                                        {
                                                if ((j = (-(Ci[p++]) - 2)) >= 0) /* found object j */
                                                {
                                                        Ci[q] = Cp[j]; /* restore first entry of object */
                                                        Cp[j] = q++; /* new pointer to object j */
                                                        for (k3 = 0; k3 < len[j] - 1; k3++)
                                                                Ci[q++] = Ci[p++];
                                                }
                                        }
                                        cnz = q; /* Ci [cnz...nzmax-1] now free */
                                }
                                /*
                                 * --- Construct new element
                                 * ----------------------------------------
                                 */
                                dk = 0;
                                W[nv + k] = -nvk; /* flag k as in Lk */
                                p = Cp[k];
                                pk1 = (elenk == 0) ? p : cnz; /* do in place if W[elen+k] == 0 */
                                pk2 = pk1;
                                for (k1 = 1; k1 <= elenk + 1; k1++) {
                                        if (k1 > elenk) {
                                                e = k; /* search the nodes in k */
                                                pj = p; /* list of nodes starts at Ci[pj] */
                                                ln = len[k] - elenk; /* length of list of nodes in k */
                                        } else {
                                                e = Ci[p++]; /* search the nodes in e */
                                                pj = Cp[e];
                                                ln = len[e]; /* length of list of nodes in e */
                                        }
                                        for (k2 = 1; k2 <= ln; k2++) {
                                                i = Ci[pj++];
                                                if ((nvi = W[nv + i]) <= 0)
                                                        continue; /* node i dead, or seen */
                                                dk += nvi; /* W[degree+Lk] += size of node i */
                                                W[nv + i] = -nvi; /* negate W[nv+i] to denote i in Lk */
                                                Ci[pk2++] = i; /* place i in Lk */
                                                if (W[next + i] != -1)
                                                        last[W[next + i]] = last[i];
                                                if (last[i] != -1) /* remove i from degree list */
                                                {
                                                        W[next + last[i]] = W[next + i];
                                                } else {
                                                        W[head + W[degree + i]] = W[next + i];
                                                }
                                        }
                                        if (e != k) {
                                                Cp[e] = (-(k) - 2); /* absorb e into k */
                                                W[w + e] = 0; /* e is now a dead element */
                                        }
                                }
                                if (elenk != 0)
                                        cnz = pk2; /* Ci [cnz...nzmax] is free */
                                W[degree + k] = dk; /* external degree of k - |Lk\i| */
                                Cp[k] = pk1; /* element k is in Ci[pk1..pk2-1] */
                                len[k] = pk2 - pk1;
                                W[elen + k] = -2; /* k is now an element */
                                /*
                                 * --- Find set differences
                                 * -----------------------------------------
                                 */
                                mark = cs_wclear(mark, lemax, w, n, W); /* clear w if necessary */
                                for (pk = pk1; pk < pk2; pk++) /* scan 1: find |Le\Lk| */
                                {
                                        i = Ci[pk];
                                        if ((eln = W[elen + i]) <= 0)
                                                continue;/* skip if W[elen+i] empty */
                                        nvi = -W[nv + i]; /* nv [i] was negated */
                                        wnvi = mark - nvi;
                                        for (p = Cp[i]; p <= Cp[i] + eln - 1; p++) /* scan Ei */
                                        {
                                                e = Ci[p];
                                                if (W[w + e] >= mark) {
                                                        W[w + e] -= nvi; /* decrement |Le\Lk| */
                                                } else if (W[w + e] != 0) /* ensure e is a live element */
                                                {
                                                        W[w + e] = W[degree + e] + wnvi; /* 1st time e seen in scan 1 */
                                                }
                                        }
                                }
                                /*
                                 * --- Degree update
                                 * ------------------------------------------------
                                 */
                                for (pk = pk1; pk < pk2; pk++) /* scan2: degree update */
                                {
                                        i = Ci[pk]; /* consider node i in Lk */
                                        p1 = Cp[i];
                                        p2 = p1 + W[elen + i] - 1;
                                        pn = p1;
                                        for (h = 0, d = 0, p = p1; p <= p2; p++) /* scan Ei */
                                        {
                                                e = Ci[p];
                                                if (W[w + e] != 0) /* e is an unabsorbed element */
                                                {
                                                        dext = W[w + e] - mark; /* dext = |Le\Lk| */
                                                        if (dext > 0) {
                                                                d += dext; /* sum up the set differences */
                                                                Ci[pn++] = e; /* keep e in Ei */
                                                                h += e; /* compute the hash of node i */
                                                        } else {
                                                                Cp[e] = (-(k) - 2); /* aggressive absorb. e.k */
                                                                W[w + e] = 0; /* e is a dead element */
                                                        }
                                                }
                                        }
                                        W[elen + i] = pn - p1 + 1; /* W[elen+i] = |Ei| */
                                        p3 = pn;
                                        p4 = p1 + len[i];
                                        for (p = p2 + 1; p < p4; p++) /* prune edges in Ai */
                                        {
                                                j = Ci[p];
                                                if ((nvj = W[nv + j]) <= 0)
                                                        continue; /* node j dead or in Lk */
                                                d += nvj; /* degree(i) += |j| */
                                                Ci[pn++] = j; /* place j in node list of i */
                                                h += j; /* compute hash for node i */
                                        }
                                        if (d == 0) /* check for mass elimination */
                                        {
                                                Cp[i] = (-(k) - 2); /* absorb i into k */
                                                nvi = -W[nv + i];
                                                dk -= nvi; /* |Lk| -= |i| */
                                                nvk += nvi; /* |k| += W[nv+i] */
                                                nel += nvi;
                                                W[nv + i] = 0;
                                                W[elen + i] = -1; /* node i is dead */
                                        } else {
                                                W[degree + i] = (((W[degree + i]) < (d)) ? (W[degree + i]) : (d)); /* update degree(i) */
                                                Ci[pn] = Ci[p3]; /* move first node to end */
                                                Ci[p3] = Ci[p1]; /* move 1st el. to end of Ei */
                                                Ci[p1] = k; /* add k as 1st element in of Ei */
                                                len[i] = pn - p1 + 1; /* new len of adj. list of node i */
                                                h %= n; /* finalize hash of i */
                                                W[next + i] = W[hhead + h]; /* place i in hash bucket */
                                                W[hhead + h] = i;
                                                last[i] = h; /* save hash of i in last[i] */
                                        }
                                } /* scan2 is done */
                                W[degree + k] = dk; /* finalize |Lk| */
                                lemax = (((lemax) > (dk)) ? (lemax) : (dk));
                                mark = cs_wclear(mark + lemax, lemax, w, n, W); /* clear w */
                                /*
                                 * --- Supernode detection
                                 * ------------------------------------------
                                 */
                                for (pk = pk1; pk < pk2; pk++) {
                                        i = Ci[pk];
                                        if (W[nv + i] >= 0)
                                                continue; /* skip if i is dead */
                                        h = last[i]; /* scan hash bucket of node i */
                                        i = W[hhead + h];
                                        W[hhead + h] = -1; /* hash bucket will be empty */
                                        for (; i != -1 && W[next + i] != -1; i = W[next + i], mark++) {
                                                ln = len[i];
                                                eln = W[elen + i];
                                                for (p = Cp[i] + 1; p <= Cp[i] + ln - 1; p++)
                                                        W[w + Ci[p]] = mark;
                                                jlast = i;
                                                for (j = W[next + i]; j != -1;) /* compare i with all j */
                                                {
                                                        ok = (len[j] == ln) && (W[elen + j] == eln);
                                                        for (p = Cp[j] + 1; ok && p <= Cp[j] + ln - 1; p++) {
                                                                if (W[w + Ci[p]] != mark)
                                                                        ok = false; /* compare i and j */
                                                        }
                                                        if (ok) /* i and j are identical */
                                                        {
                                                                Cp[j] = (-(i) - 2); /* absorb j into i */
                                                                W[nv + i] += W[nv + j];
                                                                W[nv + j] = 0;
                                                                W[elen + j] = -1; /* node j is dead */
                                                                j = W[next + j]; /* delete j from hash bucket */
                                                                W[next + jlast] = j;
                                                        } else {
                                                                jlast = j; /* j and i are different */
                                                                j = W[next + j];
                                                        }
                                                }
                                        }
                                }
                                /*
                                 * --- Finalize new
                                 * element------------------------------------------
                                 */
                                for (p = pk1, pk = pk1; pk < pk2; pk++) /* finalize Lk */
                                {
                                        i = Ci[pk];
                                        if ((nvi = -W[nv + i]) <= 0)
                                                continue;/* skip if i is dead */
                                        W[nv + i] = nvi; /* restore W[nv+i] */
                                        d = W[degree + i] + dk - nvi; /* compute external degree(i) */
                                        d = (((d) < (n - nel - nvi)) ? (d) : (n - nel - nvi));
                                        if (W[head + d] != -1)
                                                last[W[head + d]] = i;
                                        W[next + i] = W[head + d]; /* put i back in degree list */
                                        last[i] = -1;
                                        W[head + d] = i;
                                        mindeg = (((mindeg) < (d)) ? (mindeg) : (d)); /* find new minimum degree */
                                        W[degree + i] = d;
                                        Ci[p++] = i; /* place i in Lk */
                                }
                                W[nv + k] = nvk; /* # nodes absorbed into k */
                                if ((len[k] = p - pk1) == 0) /* length of adj list of element k */
                                {
                                        Cp[k] = -1; /* k is a root of the tree */
                                        W[w + k] = 0; /* k is now a dead element */
                                }
                                if (elenk != 0)
                                        cnz = p; /* free unused space in Lk */
                        }
                        /*
                         * --- Postordering
                         * -----------------------------------------------------
                         */
                        for (i = 0; i < n; i++)
                                Cp[i] = (-(Cp[i]) - 2);/* fix assembly tree */
                        for (j = 0; j <= n; j++)
                                W[head + j] = -1;
                        for (j = n; j >= 0; j--) /* place unordered nodes in lists */
                        {
                                if (W[nv + j] > 0)
                                        continue; /* skip if j is an element */
                                W[next + j] = W[head + Cp[j]]; /* place j in list of its parent */
                                W[head + Cp[j]] = j;
                        }
                        for (e = n; e >= 0; e--) /* place elements in lists */
                        {
                                if (W[nv + e] <= 0)
                                        continue; /* skip unless e is an element */
                                if (Cp[e] != -1) {
                                        W[next + e] = W[head + Cp[e]]; /* place e in list of its parent */
                                        W[head + Cp[e]] = e;
                                }
                        }
                        for (k = 0, i = 0; i <= n; i++) /* postorder the assembly tree */
                        {
                                if (Cp[i] == -1)
                                        k = cs_tdfs(i, k, head, next, P, w, W);
                        }
                        return P;
                }

                private static int cs_fkeep(cs A) {
                        int j, p, nz = 0, n;
                        int[] Ap, Ai;
                        double[] Ax;
                        if (!(A != null && (A.nz == -1)))
                                return (-1); /* check inputs */
                        n = A.n;
                        Ap = A.p;
                        Ai = A.i;
                        Ax = A.x;
                        for (j = 0; j < n; j++) {
                                p = Ap[j]; /* get current location of col j */
                                Ap[j] = nz; /* record new location of col j */
                                for (; p < Ap[j + 1]; p++) {
                                        if (Ai[p] != j) {
                                                if (Ax != null)
                                                        Ax[nz] = Ax[p]; /* keep A(i,j) */
                                                Ai[nz++] = Ai[p];
                                        }
                                }
                        }
                        Ap[n] = nz; /* finalize A */
                        cs_sprealloc(A, 0); /* remove extra space from A */
                        return (nz);
                }

                private static int cs_wclear(int markIN, int lemax, int w, int n, int[] W) {
                        int mark = markIN;
                        int k;
                        if (mark < 2 || (mark + lemax < 0)) {
                                for (k = 0; k < n; k++)
                                        if (W[w + k] != 0)
                                                W[w + k] = 1;
                                mark = 2;
                        }
                        return (mark); /* at this point, w [0..n-1] < mark holds */
                }

                private static cs cs_add(cs A, cs B, int alpha, int beta) {
                        int p, j, nz = 0, anz, m, n, bnz;
                        int[] Cp, Ci, Bp, w;
                        boolean values;
                        double[] x, Bx, Cx;
                        cs C;
                        if (!(A != null && (A.nz == -1)) || !(B != null && (B.nz == -1)))
                                return (null); /* check inputs */
                        if (A.m != B.m || A.n != B.n)
                                return (null);
                        m = A.m;
                        anz = A.p[A.n];
                        n = B.n;
                        Bp = B.p;
                        Bx = B.x;
                        bnz = Bp[n];
                        w = new int[m]; /* get workspace */
                        values = (A.x != null) && (Bx != null);
                        x = values ? new double[m] : null; /* get workspace */
                        C = new cs(m, n, anz + bnz, values ? 1 : 0, false); // allocate
                                                                            // result
                        if (values && x == null)
                                return null;
                        Cp = C.p;
                        Ci = C.i;
                        Cx = C.x;
                        for (j = 0; j < n; j++) {
                                Cp[j] = nz; /* column j of C starts here */
                                nz = cs_scatter(A, j, alpha, w, x, j + 1, C, nz); /* alpha*A(:,j) */
                                nz = cs_scatter(B, j, beta, w, x, j + 1, C, nz); /* beta*B(:,j) */
                                if (values)
                                        for (p = Cp[j]; p < nz; p++)
                                                Cx[p] = x[Ci[p]];
                        }
                        Cp[n] = nz; /* finalize the last column of C */
                        cs_sprealloc(C, 0); /* remove extra space from C */
                        return C; /* success; free workspace, return C */
                }

        }

        /**
         * 
         * @param index_edges
         *                edges of the plateau
         * @param M
         *                number of edges of the plateau
         * @param index
         *                nodes of the plateau
         * @param boundary_values
         * @param numb_boundary
         */
        private void RandomWalker(Pixel[][] index_edges, int M, ArrayList<Pixel> index, float[][] boundary_values, int numb_boundary) {
                ArrayList<PseudoEdge> edgeL = new ArrayList<>();
                for (int j = 0; j < M; j++) {
                        edgeL.add(new PseudoEdge(index_edges[0][j], index_edges[1][j]));
                }
                boolean[] seeded_vertex = new boolean[index.size()];
                int[] indic_sparse = new int[index.size()];
                int[] nb_same_edges = new int[M];

                // Indexing the edges, and the seeds
                for (int i = 0; i < index.size(); i++)
                        index.get(i).indic_VP = gPixels[i];

                for (int j = 0; j < M; j++) {
                        Pixel v1 = edgeL.get(j).p1.indic_VP;
                        Pixel v2 = edgeL.get(j).p2.indic_VP;
                        // Pixel v1 = index_edges[0][j].indic_VP;
                        // Pixel v2 = index_edges[1][j].indic_VP;
                        if (v1.pointer < v2.pointer) {
                                edgeL.get(j).p1 = edgeL.get(j).p1.indic_VP;
                                edgeL.get(j).p2 = edgeL.get(j).p2.indic_VP;
                                indic_sparse[edgeL.get(j).p1.pointer]++;
                                indic_sparse[edgeL.get(j).p2.pointer]++;
                                // index_edges[0][j] = index_edges[0][j].indic_VP;
                                // index_edges[1][j] = index_edges[1][j].indic_VP;
                                // indic_sparse[index_edges[0][j].pointer]++;
                                // indic_sparse[index_edges[1][j].pointer]++;
                        } else {
                                edgeL.get(j).p2 = v1;
                                edgeL.get(j).p1 = v2;
                                indic_sparse[edgeL.get(j).p1.pointer]++;
                                indic_sparse[edgeL.get(j).p2.pointer]++;
                                // index_edges[1][j] = v1;
                                // index_edges[0][j] = v2;
                                // indic_sparse[index_edges[0][j].pointer]++;
                                // indic_sparse[index_edges[1][j].pointer]++;
                        }
                }
                /* TriEdges(index_edges, M, nb_same_edges); */
                {
                        Collections.sort(edgeL);
                        for (int m = 0; m < M; m++) {
                                int n = 0;
                                while ((m + n < M - 1) && edgeL.get(m + n).compareTo(edgeL.get(m + n + 1)) == 0) {
                                        n++;
                                }
                                nb_same_edges[m] = n;
                        }
                }

                for (int i = 0; i < numb_boundary; i++) {
                        gPixels[i].local_seed = gPixels[i].local_seed.indic_VP;
                        seeded_vertex[gPixels[i].local_seed.pointer] = true;
                }

                for (int j = 0; j < M; j++) {
                        index_edges[0][j] = edgeL.get(j).p1;
                        index_edges[1][j] = edgeL.get(j).p2;
                }

                // The system to solve is A x = -B X2
                // building matrix A : laplacian for unseeded nodes
                Matrix A2_m = new Matrix(index.size() - numb_boundary, index.size() - numb_boundary);
                fill_A(A2_m, index.size(), M, numb_boundary, index_edges, seeded_vertex, indic_sparse, nb_same_edges);
                cs A2 = jama2cs(A2_m, M * 2 + index.size());
                cs A = cs.cs_compress(A2);

                // building boundary matrix B
                Matrix B2_m = new Matrix(index.size() - numb_boundary, numb_boundary);
                fill_B(B2_m, index.size(), M, numb_boundary, index_edges, seeded_vertex, indic_sparse, nb_same_edges);
                cs B2 = jama2cs(B2_m, 2 * M + index.size());
                cs B = cs.cs_compress(B2);

                // building the right hand side of the system
                cs X = new cs(numb_boundary, 1, numb_boundary, 1, true);
                double[] b = new double[index.size() - numb_boundary];
                for (int l = 0; l < labels.size() - 1; l++) {
                        // building vector X
                        int rnz = 0;
                        for (int i = 0; i < numb_boundary; i++) {
                                X.x[rnz] = boundary_values[l][i];
                                X.p[rnz] = 0;
                                X.i[rnz] = i;
                                rnz++;
                        }
                        X.nz = rnz;
                        X.m = numb_boundary;
                        X.n = 1;

                        cs X2 = cs.cs_compress(X);
                        cs b_tmp = cs.cs_multiply(B, X2);

                        for (int i = 0; i < index.size() - numb_boundary; i++)
                                b[i] = 0;

                        for (int i = 0; i < b_tmp.nzmax; i++)
                                b[b_tmp.i[i]] = -b_tmp.x[i];
                        // solve Ax=b by LU decomposition, order = 1
                        A.cs_lusol(1, b, 1e-7);

                        int cpt = 0;
                        for (int k = 0; k < index.size(); k++) {
                                if (seeded_vertex[k] == false) {
                                        proba[l][index.get(k).pointer] = (float) b[cpt];
                                        cpt++;
                                }
                        }
                        // Enforce boundaries exactly
                        for (int k = 0; k < numb_boundary; k++)
                                proba[l][index.get(gPixels[k].local_seed.pointer).pointer] = boundary_values[l][k];
                }
        }

        private void fill_B(Matrix B, int N, int M, int numb_boundary, PWSHEDOP<T, L>.Pixel[][] index_edges, boolean[] seeded_vertex,
                        int[] indic_sparse, int[] nb_same_edges) {
                for (int k = 0; k < M; k++) {
                        if (seeded_vertex[index_edges[0][k].pointer] == true) {
                                B.set(indic_sparse[index_edges[1][k].pointer], indic_sparse[index_edges[0][k].pointer], -nb_same_edges[k] - 1);
                                k = k + nb_same_edges[k];
                        } else if (seeded_vertex[index_edges[1][k].pointer] == true) {
                                B.set(indic_sparse[index_edges[0][k].pointer], indic_sparse[index_edges[1][k].pointer], -nb_same_edges[k] - 1);
                                k = k + nb_same_edges[k];
                        }
                }

        }

        private cs jama2cs(Matrix A, int nzmax) {
                int n = A.getColumnDimension();
                int m = A.getRowDimension();
                cs output = new cs(m, n, nzmax, 1, true);
                int cur = 0;
                for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                                if (A.get(i, j) != 0) {
                                        output.i[cur] = i;
                                        output.p[cur] = j;
                                        output.x[cur] = A.get(i, j);
                                        cur++;
                                }
                        }
                }
                output.nz = cur;
                return output;
        }

        private void fill_A(Matrix A, int N, int M, int numb_boundary, PWSHEDOP<T, L>.Pixel[][] index_edges, boolean[] seeded_vertex,
                        int[] indic_sparse, int[] nb_same_edges) {
                int rnz = 0;
                // fill the diagonal
                for (int k = 0; k < N; k++) {
                        if (seeded_vertex[k] == false) {
                                A.set(rnz, rnz, indic_sparse[k]);
                                rnz++;
                        }
                }
                int rnzs = 0;
                int rnzu = 0;
                for (int k = 0; k < N; k++) {
                        if (seeded_vertex[k] == true) {
                                indic_sparse[k] = rnzs;
                                rnzs++;
                        } else {
                                indic_sparse[k] = rnzu;
                                rnzu++;
                        }
                }
                for (int k = 0; k < M; k++) {
                        if ((seeded_vertex[index_edges[0][k].pointer] == false) && (seeded_vertex[index_edges[1][k].pointer] == false)) {
                                A.set(indic_sparse[index_edges[0][k].pointer], indic_sparse[index_edges[1][k].pointer], -nb_same_edges[k] - 1);
                                rnz++;
                                A.set(indic_sparse[index_edges[1][k].pointer], indic_sparse[index_edges[0][k].pointer], -nb_same_edges[k] - 1);
                                rnz++;
                                k = k + nb_same_edges[k];
                        }
                }
        }

        private void merge_node(Pixel e1, Pixel e2) {
                if ((e1 != e2) && (proba[0][e1.pointer] < 0 || proba[0][e2.pointer] < 0)) {
                        // link re1 and re2;
                        // the Pixel with the smaller Rnk points to the other
                        // if both have the same rank increase the rnk of e2
                        if (e1.Rnk > e2.Rnk) {
                                e2.Fth = e1;
                        } else {
                                if (e1.Rnk == e2.Rnk) {
                                        e2.Rnk = e2.Rnk + 1;
                                }
                                e1.Fth = e2;
                        }

                        // which one has proba[0] < 0? Fill proba[_][ex] with proba[_][ey]
                        if (proba[0][e1.pointer] < 0) {
                                for (int k = 0; k < labels.size() - 1; k++) {
                                        proba[k][e1.pointer] = proba[k][e2.pointer];
                                }
                        } else {
                                for (int k = 0; k < labels.size() - 1; k++) {
                                        proba[k][e2.pointer] = proba[k][e1.pointer];
                                }
                        }
                }
        }

        private int[] boundary(int[] img) {
                int[] out = new int[img.length];
                for (int i = 2; i < width - 1; i++) {
                        for (int j = 2; j < height - 1; j++) {
                                if ((img[i - 1 + j * width] != img[i + j * width]) || (img[i + 1 + j * width] != img[i + j * width])
                                                || (img[i + (j - 1) * width] != img[i + j * width])
                                                || (img[i + (j + 1) * width] != img[i + j * width])) {
                                        out[i + j * width] = 255;
                                        out[i + 1 + j * width] = 255;
                                        out[i - 1 + j * width] = 255;
                                        out[i + (j - 1) * width] = 255;
                                        out[i + (j + 1) * width] = 255;
                                } else {
                                        out[i + j * width] = 0;
                                }
                        }
                }
                return out;
        }

        private void gageodilate_union_find() {
                // F = seeds_function
                // M = numberOfEdges
                // G = normal_weights
                // O = weights
                @SuppressWarnings("unchecked")
                ArrayList<PWSHEDOP<T, L>.Edge> seeds_func = (ArrayList<PWSHEDOP<T, L>.Edge>) edges.clone();
                weights = false;
                Collections.sort(seeds_func);
                Collections.reverse(seeds_func);
                for (Edge e : seeds_func) {
                        for (Edge n : e.neighbors) {
                                if (n != null && n.Mrk) {
                                        // element_link_geod_dilate(n, e);
                                        {
                                                Edge r = n.find();
                                                if (r != e) {
                                                        if ((r.normal_weight == e.normal_weight) || (e.normal_weight >= r.weight)) {
                                                                r.Fth = e;
                                                                e.weight = Math.max(r.weight, e.weight);
                                                        } else {
                                                                e.weight = 255;
                                                        }
                                                }
                                        }

                                }
                                e.Mrk = true;
                        }
                }
                Collections.reverse(seeds_func);
                for (Edge e : seeds_func) {
                        if (e.Fth == e) // p is root
                        {
                                if (e.weight == 255) {
                                        e.weight = e.normal_weight;
                                }
                        } else {
                                e.weight = e.Fth.weight;
                        }
                }
        }

        /**
         * 
         * @param input
         *                2D array
         * @return 1D version of it
         */
        private int[] toOneDim(int[][] input) {
                int out[] = new int[numOfPixels];
                for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                                out[y * width + x] = input[x][y];
                        }
                }
                return out;
        }

        /**
         * 
         * @param input
         *                1D array
         * @return 2D version of it
         */
        private int[][] toTwoDim(int[] input) {
                int out[][] = new int[width][height];
                for (int i = 0; i < numOfPixels; i++) {
                        out[i % width][i / width] = input[i];
                }
                return out;
        }

}
