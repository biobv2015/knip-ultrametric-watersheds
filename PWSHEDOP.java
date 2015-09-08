package org.knime.knip.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Stack;

import org.scijava.ItemIO;
import org.scijava.plugin.Menu;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import Jama.LUDecomposition;
import Jama.Matrix;
import net.imagej.ImgPlus;
import net.imagej.ops.Op;
import net.imagej.ops.OpService;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.labeling.Labeling;
import net.imglib2.labeling.LabelingType;
import net.imglib2.type.numeric.IntegerType;

@Plugin(menu = {@Menu(label = "DeveloperPlugins"),
                @Menu(label = "PowerWatershed")}, description = "TODO", headless = true, type = Op.class, name = "PWSHED")
public class PWSHEDOP<T extends IntegerType<T>, L extends Comparable<L>> implements Op {

        public static int SIZE_MAX_PLATEAU = 1000000;
        public static double epsilon = 0.000001;

        int[][] r_pixels;
        int[][] g_pixels;
        int[][] b_pixels;
        ArrayList<Edge<T, L>> edges;
        int numOfEdges;
        int numOfPixels;
        ArrayList<Pixel<T, L>> seedsL;
        int width;
        int height;
        ArrayList<L> labels;
        Pixel<T, L>[] gPixels;
        Pixel<T, L>[][] gPixelsT;
        float[][] proba;

        @SuppressWarnings("deprecation")
        @Parameter(type = ItemIO.OUTPUT)
        private Labeling<L> output;

        @Parameter
        private ImgPlus<T> image_path;

        @SuppressWarnings("deprecation")
        @Parameter
        private Labeling<L> seed_path;

        @Parameter
        private boolean color;

        @Parameter
        private OpService ops;

        @Override
        public void run() {

                final RandomAccess<T> imgRA = image_path.randomAccess();
                @SuppressWarnings("deprecation")
                final Cursor<LabelingType<L>> seedCursor = seed_path.cursor();
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
                //                if (color) {
                //                        r_pixels = new int[width][height];
                //                        g_pixels = new int[width][height];
                //                        b_pixels = new int[width][height];
                //                        for (int y = 0; y < height; y++) {
                //                                for (int x = 0; x < width; x++) {
                //                                        r_pixels[x][y] = (image[x][y] & (255 << 16)) >> 16;
                //                                        g_pixels[x][y] = (image[x][y] & (255 << 8)) >> 8;
                //                                        b_pixels[x][y] = image[x][y] & 255;
                //                                }
                //                        }
                //                }
                seedsL = new ArrayList<>();
                labels = new ArrayList<L>();
                /*
                 * Get the seeds from the input labeling. "labels" is the List of the labels, while "seedsL" stores the seeds.
                 */
                while (seedCursor.hasNext()) {
                        seedCursor.fwd();
                        List<L> labeling = seedCursor.get().getLabeling();
                        if (labeling.size() != 0) {
                                L label = labeling.get(0);
                                seedsL.add(new Pixel<T, L>(seedCursor.getIntPosition(0), seedCursor.getIntPosition(1), label, width));
                                if (!labels.contains(label)) {
                                        labels.add(label);
                                }
                        }
                }

                /*
                 * Create a "Pixel" for each pixel in the input, all are labeled as label_0
                 */
                gPixels = new Pixel[numOfPixels];
                gPixelsT = new Pixel[width][height];
                for (int i = 0; i < width; i++) {
                        for (int j = 0; j < height; j++) {
                                gPixelsT[i][j] = new Pixel<T, L>(i, j, labels.get(0), width);
                                gPixels[i + width * j] = gPixelsT[i][j];
                        }
                }

                /*
                 * Create the edges.
                 */
                edges = new ArrayList<>();
                Edge<T, L>[][] hor_edges = new Edge[width - 1][height];
                Edge<T, L>[][] ver_edges = new Edge[width][height - 1];
                for (int i = 0; i < height - 1; i++) {
                        for (int j = 0; j < width; j++) {
                                ver_edges[j][i] = new Edge<T, L>(gPixelsT[j][i], gPixelsT[j][i + 1], edges.size());
                                edges.add(ver_edges[j][i]);
                        }
                }
                for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width - 1; j++) {
                                hor_edges[j][i] = new Edge<T, L>(gPixelsT[j][i], gPixelsT[j + 1][i], edges.size());
                                edges.add(hor_edges[j][i]);
                        }
                }

                /*
                 * get the neighbor-information
                 */
                for (int i = 0; i < numOfEdges; i++) {
                        Edge<T, L> e = edges.get(i);
                        if (!e.isVertical()) {
                                if (e.getN1y() > 0) {
                                        e.neighbors[0] = ver_edges[e.getN2x()][e.getN1y() - 1];
                                        e.neighbors[1] = ver_edges[e.getN1x()][e.getN1y() - 1];
                                }
                                if (e.getN1x() > 0) {
                                        e.neighbors[2] = hor_edges[e.getN1x() - 1][e.getN1y()];
                                }
                                if (e.getN1y() < ver_edges[0].length - 1) {
                                        e.neighbors[3] = ver_edges[e.getN1x()][e.getN1y()];
                                        e.neighbors[4] = ver_edges[e.getN2x()][e.getN1y()];
                                }
                                if (e.getN1x() < hor_edges.length - 1) {
                                        e.neighbors[5] = hor_edges[e.getN2x()][e.getN1y()];
                                }
                        } else { // e.isVertical()
                                if (e.getN1y() > 0) {
                                        e.neighbors[0] = ver_edges[e.getN1x()][e.getN1y() - 1];
                                }
                                if (e.getN1x() > 0) {
                                        e.neighbors[1] = hor_edges[e.getN1x() - 1][e.getN1y()];
                                        e.neighbors[2] = hor_edges[e.getN2x() - 1][e.getN2y()];
                                }
                                if (e.getN2y() < ver_edges[0].length - 1) {
                                        e.neighbors[3] = ver_edges[e.getN2x()][e.getN2y()];
                                }
                                if (e.getN1x() < hor_edges.length - 1) {
                                        e.neighbors[4] = hor_edges[e.getN2x()][e.getN2y()];
                                        e.neighbors[5] = hor_edges[e.getN1x()][e.getN1y()];
                                }
                        }
                }

                /*
                 * Get the neighbors for the seed-pixels
                 */
                for (Pixel<T, L> p : seedsL) {
                        if (p.getX() < hor_edges.length) {
                                p.neighbors[0] = hor_edges[p.getX()][p.getY()];
                        }
                        if (p.getY() < ver_edges.length) {
                                p.neighbors[1] = ver_edges[p.getX()][p.getY()];
                        }
                        if (p.getX() > 0) {
                                p.neighbors[2] = hor_edges[p.getX() - 1][p.getY()];
                        }
                        if (p.getY() > 0) {
                                p.neighbors[3] = ver_edges[p.getX()][p.getY() - 1];
                        }
                }

                //                if (color) {
                //                        for (Edge<T, L> e : edges) {
                //                                int wr = Math.abs(r_pixels[e.n1x][e.n1y] - r_pixels[e.n2x][e.n2y]);
                //                                int wg = Math.abs(g_pixels[e.n1x][e.n1y] - g_pixels[e.n2x][e.n2y]);
                //                                int wb = Math.abs(b_pixels[e.n1x][e.n1y] - b_pixels[e.n2x][e.n2y]);
                //                                e.weight = 255 - wr;
                //                                if (255 - wg < e.weight) {
                //                                        e.weight = 255 - wg;
                //                                }
                //                                if (255 - wb < e.weight) {
                //                                        e.weight = 255 - wb;
                //                                }
                //                        }
                //                        for (Edge<T, L> e : edges) {
                //                                e.normal_weight = e.weight;
                //                        }
                //                } else {

                for (Edge<T, L> e : edges) {
                        imgRA.setPosition(e.getN1x(), 0);
                        imgRA.setPosition(e.getN1y(), 1);
                        int v1 = imgRA.get().getInteger();
                        imgRA.setPosition(e.getN2x(), 0);
                        imgRA.setPosition(e.getN2y(), 1);
                        int v2 = imgRA.get().getInteger();
                        //TODO calculation more general
                        e.normal_weight = 255 - Math.abs(v1 - v2);
                }

                //                }

                /*
                 * The edges connected to seeds get the weight set as their normal_weight
                 */
                for (Pixel<T, L> p : seedsL) {
                        for (Edge<T, L> e : p.neighbors) {
                                if (e != null) {
                                        e.weight = e.normal_weight;
                                }
                        }
                }

                ArrayList<Edge<T, L>> seeds_func = (ArrayList<Edge<T, L>>) edges.clone();
                Edge.weights = false;
                Collections.sort(seeds_func);
                Collections.reverse(seeds_func);
                for (Edge<T, L> e : seeds_func) {
                        for (Edge<T, L> n : e.neighbors) {
                                if (n != null && n.Mrk) {
                                        Edge<T, L> r = n.find();
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
                Collections.reverse(seeds_func);
                for (Edge<T, L> e : seeds_func) {
                        if (e.Fth == e) {
                                // p is root
                                if (e.weight == 255) {
                                        e.weight = e.normal_weight;
                                }
                        } else {
                                e.weight = e.Fth.weight;
                        }
                }
                PowerWatershed_q2();

                // writing results
                //TODO: Fix the confusion of labels
                output = seed_path.copy();
                @SuppressWarnings("deprecation")
                Cursor<LabelingType<L>> outCursor = output.cursor();
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
                        if (val > maxi) {
                                argmax = labels.size() - 1;
                        }
                        outCursor.get().setLabel(labels.get(argmax));
                        outCursor.fwd();
                }

        }

        private void PowerWatershed_q2() {
                proba = new float[labels.size() - 1][numOfPixels];
                for (int i = 0; i < labels.size() - 1; i++) {
                        Arrays.fill(proba[i], -1);
                }
                // proba[i][j] =1 <=> pixel[i] has label j+1
                for (Pixel<T, L> pix : seedsL) {
                        int pixPointer = pix.getPointer();
                        for (int j = 0; j < labels.size() - 1; j++) {
                                proba[j][pixPointer] = pix.label == labels.get(j + 1) ? 1 : 0;
                        }
                }
                float[][] local_labels = new float[labels.size() - 1][numOfPixels];
                @SuppressWarnings("unchecked")
                ArrayList<Edge<T, L>> sorted_weights = (ArrayList<Edge<T, L>>) edges.clone();
                Edge.weights = true;
                Collections.sort(sorted_weights);
                Collections.reverse(sorted_weights);

                /* beginning of main loop */
                for (Edge<T, L> e_max : sorted_weights) {
                        if (e_max.visited) {
                                continue;
                        }

                        // 1. Computing the edges of the plateau LCP linked to the edge
                        // e_max
                        Stack<Edge<T, L>> LIFO = new Stack<>();
                        HashSet<Edge<T, L>> visited = new HashSet<>();
                        ArrayList<Pixel<T, L>> Plateau = new ArrayList<>(); // vertices of a
                                                                            // plateau.
                        LIFO.add(e_max);
                        e_max.visited = true;
                        visited.add(e_max);
                        int wmax = e_max.weight;
                        ArrayList<Edge<T, L>> sorted_weights2 = new ArrayList<Edge<T, L>>();

                        // 2. putting the edges and vertices of the plateau into arrays
                        while (!LIFO.empty()) {
                                Edge<T, L> x = LIFO.pop();
                                Pixel<T, L> re1 = x.p1.find();
                                Pixel<T, L> re2 = x.p2.find();
                                if (proba[0][re1.getPointer()] < 0 || proba[0][re2.getPointer()] < 0) {
                                        if (!x.p1.visited) {
                                                Plateau.add(x.p1);
                                                x.p1.visited = true;
                                        }
                                        if (!x.p2.visited) {
                                                Plateau.add(x.p2);
                                                x.p2.visited = true;
                                        }
                                        sorted_weights2.add(x);
                                }

                                for (Edge<T, L> edge : x.neighbors) {
                                        if (edge != null) {
                                                if ((!visited.contains(edge)) && (edge.weight == wmax)) {
                                                        LIFO.add(edge);
                                                        visited.add(edge);
                                                        edge.visited = true;
                                                }
                                        }
                                }
                        }

                        for (Pixel<T, L> p : Plateau) {
                                p.visited = false;
                        }

                        // 3. If e_max belongs to a plateau
                        if (sorted_weights2.size() > 0) {
                                // 4. Evaluate if there are differents seeds on the plateau
                                boolean different_seeds = false;

                                for (int i = 0; i < labels.size() - 1; i++) {
                                        int p = 0;
                                        double val = -0.5;
                                        for (Pixel<T, L> j : Plateau) {
                                                int xr = j.find().getPointer();
                                                if (Math.abs(proba[i][xr] - val) > epsilon && proba[i][xr] >= 0) {
                                                        p++;
                                                        val = proba[i][xr];
                                                        if (p >= 2) {
                                                                different_seeds = true;
                                                                break;
                                                        }
                                                }
                                        }
                                        if (different_seeds) {
                                                break;
                                        }
                                }

                                if (different_seeds) {
                                        // 5. Sort the edges of the plateau according to their
                                        // normal weight
                                        Edge.weights = false;
                                        Collections.sort(sorted_weights2);
                                        Collections.reverse(sorted_weights2);

                                        // Merge nodes for edges of real max weight
                                        Plateau.clear();
                                        ArrayList<PseudoEdge<T, L>> edgesLCP = new ArrayList<>();
                                        for (Edge<T, L> Ne_max : sorted_weights2) {
                                                Pixel<T, L> re1 = Ne_max.p1.find();
                                                Pixel<T, L> re2 = Ne_max.p2.find();
                                                if (Ne_max.normal_weight != wmax) {
                                                        merge_node(re1, re2);
                                                } else {
                                                        if ((re1 != re2) && ((proba[0][re1.getPointer()] < 0 || proba[0][re2.getPointer()] < 0))) {
                                                                if (!re1.visited) {
                                                                        Plateau.add(re1);
                                                                        re1.visited = true;
                                                                }
                                                                if (!re2.visited) {
                                                                        Plateau.add(re2);
                                                                        re2.visited = true;
                                                                }
                                                                edgesLCP.add(new PseudoEdge<>(re1, re2));
                                                        }
                                                }
                                        }

                                        int k = 0;
                                        for (int i = 0; i < labels.size() - 1; i++) {
                                                k = 0;
                                                for (Pixel<T, L> xr : Plateau) {
                                                        if (proba[i][xr.getPointer()] >= 0) {
                                                                local_labels[i][k] = proba[i][xr.getPointer()];
                                                                gPixels[k].local_seed = xr;
                                                                k++;
                                                        }
                                                }
                                        }

                                        // 6. Execute Random Walker on plateaus
                                        if (Plateau.size() < SIZE_MAX_PLATEAU) {
                                                RandomWalker(edgesLCP, Plateau, local_labels, k);
                                        } else {
                                                System.out.printf("Plateau too big (%d vertices,%d edges), RW is not performed\n", Plateau.size(),
                                                                edgesLCP.size());
                                                for (PseudoEdge<T, L> pseudo : edgesLCP) {
                                                        merge_node(pseudo.p1.find(), pseudo.p2.find());
                                                }
                                        }

                                        for (Pixel<T, L> pix : Plateau) {
                                                pix.visited = false;
                                        }
                                } else {
                                        // if different seeds = false
                                        // 7. Merge nodes for edges of max weight
                                        for (Edge<T, L> edge : sorted_weights2) {
                                                merge_node(edge.p1.find(), edge.p2.find());
                                        }
                                }
                        }
                } // end main loop

                // building the final proba map (find the root vertex of each tree)
                for (Pixel<T, L> j : gPixels) {
                        Pixel<T, L> i = j.find();
                        for (int k = 0; k < labels.size() - 1; k++) {
                                proba[k][j.getPointer()] = proba[k][i.getPointer()];
                        }
                }
        }

        /**
         * 
         * @param index_edges
         *                edges of the plateau
         * @param index
         *                nodes of the plateau
         * @param boundary_values
         * @param numb_boundary
         */
        private void RandomWalker(ArrayList<PseudoEdge<T, L>> index_edges, ArrayList<Pixel<T, L>> index, float[][] boundary_values,
                        int numb_boundary) {
                ArrayList<PseudoEdge<T, L>> edgeL = new ArrayList<>();
                for (PseudoEdge<T, L> pseudo : index_edges) {
                        edgeL.add(new PseudoEdge<T, L>(pseudo.p1, pseudo.p2));
                }
                boolean[] seeded_vertex = new boolean[index.size()];
                int[] indic_sparse = new int[index.size()];
                int[] nb_same_edges = new int[index_edges.size()];

                // Indexing the edges, and the seeds
                for (int i = 0; i < index.size(); i++) {
                        index.get(i).indic_VP = gPixels[i];
                }

                for (PseudoEdge<T, L> pe : edgeL) {
                        Pixel<T, L> v1 = pe.p1.indic_VP;
                        Pixel<T, L> v2 = pe.p2.indic_VP;
                        if (v1.getPointer() < v2.getPointer()) {
                                pe.p1 = pe.p1.indic_VP;
                                pe.p2 = pe.p2.indic_VP;
                                indic_sparse[pe.p1.getPointer()]++;
                                indic_sparse[pe.p2.getPointer()]++;
                        } else {
                                pe.p2 = v1;
                                pe.p1 = v2;
                                indic_sparse[pe.p1.getPointer()]++;
                                indic_sparse[pe.p2.getPointer()]++;
                        }
                }
                Collections.sort(edgeL);
                for (int m = 0; m < index_edges.size(); m++) {
                        int n = 0;
                        while ((m + n < index_edges.size() - 1) && edgeL.get(m + n).compareTo(edgeL.get(m + n + 1)) == 0) {
                                n++;
                        }
                        nb_same_edges[m] = n;
                }

                for (int i = 0; i < numb_boundary; i++) {
                        gPixels[i].local_seed = gPixels[i].local_seed.indic_VP;
                        seeded_vertex[gPixels[i].local_seed.getPointer()] = true;
                }

                // The system to solve is A x = -B X2
                // building matrix A : laplacian for unseeded nodes
                Matrix A2_m = new Matrix(index.size() - numb_boundary, index.size() - numb_boundary);
                fill_A(A2_m, index.size(), numb_boundary, edgeL, seeded_vertex, indic_sparse, nb_same_edges);

                // building boundary matrix B
                Matrix B2_m = new Matrix(index.size() - numb_boundary, numb_boundary);
                fill_B(B2_m, edgeL, seeded_vertex, indic_sparse, nb_same_edges);
                LUDecomposition AXB = new LUDecomposition(A2_m);

                // building the right hand side of the system
                for (int l = 0; l < labels.size() - 1; l++) {
                        Matrix X = new Matrix(numb_boundary, 1);
                        double[] b = new double[index.size() - numb_boundary];
                        // building vector X
                        for (int i = 0; i < numb_boundary; i++) {
                                X.set(i, 0, boundary_values[l][i]);
                        }

                        Matrix b_tmp = B2_m.times(X);
                        for (int i = 0; i < index.size() - numb_boundary; i++) {
                                b[i] = 0;
                        }
                        for (int i = 0; i < b_tmp.getRowDimension(); i++) {
                                for (int j = 0; j < b_tmp.getColumnDimension(); j++) {
                                        if (b_tmp.get(i, j) != 0) {
                                                b[i] = -b_tmp.get(i, j);
                                        }
                                }
                        }
                        // solve Ax=b by LU decomposition, order = 1
                        //                        A.cs_lusol(1, b, 1e-7);
                        b = AXB.solve(new Matrix(b, b.length)).getColumnPackedCopy();

                        int cpt = 0;
                        for (int k = 0; k < index.size(); k++) {
                                if (!seeded_vertex[k]) {
                                        proba[l][index.get(k).getPointer()] = (float) b[cpt];
                                        cpt++;
                                }
                        }
                        // Enforce boundaries exactly
                        for (int k = 0; k < numb_boundary; k++) {
                                proba[l][index.get(gPixels[k].local_seed.getPointer()).getPointer()] = boundary_values[l][k];
                        }
                }
        }

        private void fill_B(Matrix B, ArrayList<PseudoEdge<T, L>> index_edges, boolean[] seeded_vertex, int[] indic_sparse, int[] nb_same_edges) {
                for (int k = 0; k < index_edges.size(); k++) {
                        int p1 = index_edges.get(k).p1.getPointer();
                        int p2 = index_edges.get(k).p2.getPointer();
                        if (seeded_vertex[p1] == true) {
                                B.set(indic_sparse[p2], indic_sparse[p1], -nb_same_edges[k] - 1);
                                k += nb_same_edges[k];
                        } else if (seeded_vertex[p2] == true) {
                                B.set(indic_sparse[p1], indic_sparse[p2], -nb_same_edges[k] - 1);
                                k += nb_same_edges[k];
                        }
                }

        }

        private void fill_A(Matrix A, int N, int numb_boundary, ArrayList<PseudoEdge<T, L>> index_edges, boolean[] seeded_vertex, int[] indic_sparse,
                        int[] nb_same_edges) {
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
                        if (seeded_vertex[k]) {
                                indic_sparse[k] = rnzs;
                                rnzs++;
                        } else {
                                indic_sparse[k] = rnzu;
                                rnzu++;
                        }
                }
                for (int k = 0; k < index_edges.size(); k++) {
                        int p1 = index_edges.get(k).p1.getPointer();
                        int p2 = index_edges.get(k).p2.getPointer();
                        if ((seeded_vertex[p1] == false) && (seeded_vertex[p2] == false)) {
                                A.set(indic_sparse[p1], indic_sparse[p2], -nb_same_edges[k] - 1);
                                A.set(indic_sparse[p2], indic_sparse[p1], -nb_same_edges[k] - 1);
                                k += nb_same_edges[k];
                        }
                }
        }

        /**
         * merges two pixels as in the else-clause of the algorithm by Camille
         * Couprie
         * 
         * @param p1
         *                first pixel
         * @param p2
         *                second pixel
         */
        private void merge_node(Pixel<T, L> p1, Pixel<T, L> p2) {
                //merge if e1!=e2 and one of them has no probability yet
                if ((p1 != p2) && (proba[0][p1.getPointer()] < 0 || proba[0][p2.getPointer()] < 0)) {
                        // link re1 and re2;
                        // the Pixel with the smaller Rnk points to the other
                        // if both have the same rank increase the rnk of e2
                        if (p1.Rnk > p2.Rnk) {
                                p2.Fth = p1;
                        } else {
                                if (p1.Rnk == p2.Rnk) {
                                        p2.Rnk = p2.Rnk + 1;
                                }
                                p1.Fth = p2;
                        }

                        // which one has proba[0] < 0? Fill proba[_][ex] with proba[_][ey]
                        if (proba[0][p1.getPointer()] < 0) {
                                for (int k = 0; k < labels.size() - 1; k++) {
                                        proba[k][p1.getPointer()] = proba[k][p2.getPointer()];
                                }
                        } else {
                                for (int k = 0; k < labels.size() - 1; k++) {
                                        proba[k][p2.getPointer()] = proba[k][p1.getPointer()];
                                }
                        }
                }
        }

}
