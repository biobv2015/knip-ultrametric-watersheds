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

        //        int[][] r_pixels;
        //        int[][] g_pixels;
        //        int[][] b_pixels;
        ArrayList<Edge<T, L>> edges;
        int width;
        int height;
        ArrayList<L> labels;
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

                ArrayList<Pixel<T, L>> seedsL = new ArrayList<>();
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
                Pixel<T, L>[] gPixels = new Pixel[width * height];
                Pixel<T, L>[][] gPixelsT = new Pixel[width][height];
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
                for (Edge<T, L> e : edges) {
                        if (!e.isVertical()) {
                                if (e.p1.getY() > 0) {
                                        e.neighbors[0] = ver_edges[e.p2.getX()][e.p1.getY() - 1];
                                        e.neighbors[1] = ver_edges[e.p1.getX()][e.p1.getY() - 1];
                                }
                                if (e.p1.getX() > 0) {
                                        e.neighbors[2] = hor_edges[e.p1.getX() - 1][e.p1.getY()];
                                }
                                if (e.p1.getY() < ver_edges[0].length - 1) {
                                        e.neighbors[3] = ver_edges[e.p1.getX()][e.p1.getY()];
                                        e.neighbors[4] = ver_edges[e.p2.getX()][e.p1.getY()];
                                }
                                if (e.p1.getX() < hor_edges.length - 1) {
                                        e.neighbors[5] = hor_edges[e.p2.getX()][e.p1.getY()];
                                }
                        } else { // e.isVertical()
                                if (e.p1.getY() > 0) {
                                        e.neighbors[0] = ver_edges[e.p1.getX()][e.p1.getY() - 1];
                                }
                                if (e.p1.getX() > 0) {
                                        e.neighbors[1] = hor_edges[e.p1.getX() - 1][e.p1.getY()];
                                        e.neighbors[2] = hor_edges[e.p2.getX() - 1][e.p2.getY()];
                                }
                                if (e.p2.getY() < ver_edges[0].length - 1) {
                                        e.neighbors[3] = ver_edges[e.p2.getX()][e.p2.getY()];
                                }
                                if (e.p1.getX() < hor_edges.length - 1) {
                                        e.neighbors[4] = hor_edges[e.p2.getX()][e.p2.getY()];
                                        e.neighbors[5] = hor_edges[e.p1.getX()][e.p1.getY()];
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
                        imgRA.setPosition(e.p1.getX(), 0);
                        imgRA.setPosition(e.p1.getY(), 1);
                        int v1 = imgRA.get().getInteger();
                        imgRA.setPosition(e.p2.getX(), 0);
                        imgRA.setPosition(e.p2.getY(), 1);
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

                Edge.weights = false;
                Collections.sort(edges);
                Collections.reverse(edges);
                //seeds_func are the edges, heaviest first
                for (Edge<T, L> e : edges) {
                        //go through the neighbors
                        for (Edge<T, L> n : e.neighbors) {
                                //if the neighbor has already been visited (same or higher normal_weight)
                                if (n != null && n.Mrk) {
                                        //get the root
                                        Edge<T, L> r = n.find();
                                        //if the root is not the current edge
                                        if (r != e) {
                                                //if the edges have the same normal_weight OR this edge has a higher normal_weight than the other root
                                                if ((e.normal_weight == r.normal_weight) || (e.normal_weight >= r.weight)) {
                                                        //set this as the root of the other root
                                                        r.Fth = e;
                                                        //and give it the weight of the old root if it's heavier than this
                                                        e.weight = Math.max(r.weight, e.weight);
                                                } else {
                                                        //if they have different normal_weights AND the root is less heavy than the normal_weight
                                                        e.weight = 255;
                                                }
                                        }
                                }
                        }
                        e.Mrk = true;
                }
                //set weight to normal_weight of roots
                Collections.reverse(edges);
                for (Edge<T, L> e : edges) {
                        if (e.Fth == e) {
                                // p is root
                                if (e.weight == 255) {
                                        e.weight = e.normal_weight;
                                }
                        } else {
                                e.weight = e.Fth.weight;
                        }
                }

                proba = new float[labels.size() - 1][width * height];
                for (int i = 0; i < labels.size() - 1; i++) {
                        Arrays.fill(proba[i], -1);
                }
                // proba[i][j] =1 <=> pixel[i] has label j
                for (Pixel<T, L> pix : seedsL) {
                        int pixPointer = pix.getPointer();
                        for (int j = 0; j < labels.size() - 1; j++) {
                                proba[j][pixPointer] = pix.label == labels.get(j) ? 1 : 0;
                        }
                }

                PowerWatershed_q2();

                // building the final proba map (find the root vertex of each tree)
                for (Pixel<T, L> j : gPixels) {
                        Pixel<T, L> i = j.find();
                        if (i != j) {
                                for (int k = 0; k < labels.size() - 1; k++) {
                                        proba[k][j.getPointer()] = proba[k][i.getPointer()];
                                }
                        }
                }

                // writing results
                //TODO: Fix the confusion of labels
                output = seed_path.copy();
                @SuppressWarnings("deprecation")
                Cursor<LabelingType<L>> outCursor = output.cursor();
                for (int j = 0; j < width * height; j++) {
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
                Edge.weights = true;
                Collections.sort(edges);
                Collections.reverse(edges);

                /* beginning of main loop */
                for (Edge<T, L> e_max : edges) {
                        if (e_max.visited) {
                                continue;
                        }

                        // 1. Computing the edges of the plateau LCP linked to the edge
                        // e_max
                        Stack<Edge<T, L>> LIFO = new Stack<>();
                        HashSet<Edge<T, L>> visited = new HashSet<>();
                        LIFO.add(e_max);
                        e_max.visited = true;
                        visited.add(e_max);
                        ArrayList<Edge<T, L>> sorted_weights = new ArrayList<Edge<T, L>>();

                        // 2. putting the edges and vertices of the plateau into arrays
                        while (!LIFO.empty()) {
                                Edge<T, L> x = LIFO.pop();
                                Pixel<T, L> re1 = x.p1.find();
                                Pixel<T, L> re2 = x.p2.find();
                                if (proba[0][re1.getPointer()] < 0 || proba[0][re2.getPointer()] < 0) {
                                        sorted_weights.add(x);
                                }

                                for (Edge<T, L> edge : x.neighbors) {
                                        if (edge != null) {
                                                if ((!visited.contains(edge)) && (edge.weight == e_max.weight)) {
                                                        LIFO.add(edge);
                                                        visited.add(edge);
                                                        edge.visited = true;
                                                }
                                        }
                                }
                        }

                        // 3. If e_max belongs to a plateau
                        if (sorted_weights.size() > 0) {
                                // 4. Evaluate if there are differents seeds on the plateau
                                boolean different_seeds = false;

                                for (int i = 0; i < labels.size() - 1; i++) {
                                        int p = 0;
                                        double val = -0.5;
                                        for (Edge<T, L> x : sorted_weights) {
                                                int xr = x.p1.find().getPointer();
                                                if (Math.abs(proba[i][xr] - val) > epsilon && proba[i][xr] >= 0) {
                                                        p++;
                                                        val = proba[i][xr];
                                                }
                                                xr = x.p2.find().getPointer();
                                                if (Math.abs(proba[i][xr] - val) > epsilon && proba[i][xr] >= 0) {
                                                        p++;
                                                        val = proba[i][xr];
                                                }
                                                if (p >= 2) {
                                                        different_seeds = true;
                                                        break;
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
                                        Collections.sort(sorted_weights);
                                        Collections.reverse(sorted_weights);

                                        // Merge nodes for edges of real max weight
                                        ArrayList<Pixel<T, L>> pixelsLCP = new ArrayList<>(); // vertices of a plateau.
                                        ArrayList<PseudoEdge<T, L>> edgesLCP = new ArrayList<>();
                                        for (Edge<T, L> e : sorted_weights) {
                                                Pixel<T, L> re1 = e.p1.find();
                                                Pixel<T, L> re2 = e.p2.find();
                                                if (e.normal_weight != e_max.weight) {
                                                        merge_node(re1, re2);
                                                } else if ((re1 != re2) && (proba[0][re1.getPointer()] < 0 || proba[0][re2.getPointer()] < 0)) {
                                                        if (!pixelsLCP.contains(re1)) {
                                                                pixelsLCP.add(re1);
                                                        }
                                                        if (!pixelsLCP.contains(re2)) {
                                                                pixelsLCP.add(re2);
                                                        }
                                                        edgesLCP.add(new PseudoEdge<>(re1, re2));

                                                }
                                        }

                                        // 6. Execute Random Walker on plateaus
                                        if (pixelsLCP.size() < SIZE_MAX_PLATEAU) {
                                                RandomWalker(edgesLCP, pixelsLCP);
                                        } else {
                                                System.out.printf("Plateau too big (%d vertices,%d edges), RW is not performed\n", pixelsLCP.size(),
                                                                edgesLCP.size());
                                                for (PseudoEdge<T, L> pseudo : edgesLCP) {
                                                        merge_node(pseudo.p1.find(), pseudo.p2.find());
                                                }
                                        }
                                } else {
                                        // if different seeds = false
                                        // 7. Merge nodes for edges of max weight
                                        for (Edge<T, L> edge : sorted_weights) {
                                                merge_node(edge.p1.find(), edge.p2.find());
                                        }
                                }
                        }
                } // end main loop
        }

        /**
         * 
         * @param edgesLCP
         *                edges of the plateau
         * @param pixelsLCP
         *                nodes of the plateau
         */
        private void RandomWalker(ArrayList<PseudoEdge<T, L>> edgesLCP, ArrayList<Pixel<T, L>> pixelsLCP) {

                boolean[] seeded_vertex = new boolean[pixelsLCP.size()];
                int[] indic_sparse = new int[pixelsLCP.size()];
                int[] nb_same_edges = new int[edgesLCP.size()];

                // Indexing the edges, and the seeds
                for (int i = 0; i < pixelsLCP.size(); i++) {
                        pixelsLCP.get(i).indic_VP = i;
                }

                for (PseudoEdge<T, L> pseudo : edgesLCP) {
                        if (pseudo.p1.indic_VP > pseudo.p2.indic_VP) {
                                Pixel<T, L> tmp = pseudo.p1;
                                pseudo.p1 = pseudo.p2;
                                pseudo.p2 = tmp;
                        }
                        indic_sparse[pseudo.p1.indic_VP]++;
                        indic_sparse[pseudo.p2.indic_VP]++;
                }
                Collections.sort(edgesLCP);
                for (int m = 0; m < edgesLCP.size(); m++) {
                        int n = 0;
                        while ((m + n < edgesLCP.size() - 1) && edgesLCP.get(m + n).compareTo(edgesLCP.get(m + n + 1)) == 0) {
                                n++;
                        }
                        nb_same_edges[m] = n;
                }

                ArrayList<ArrayList<Float>> local_labels = new ArrayList<>();
                for (int i = 0; i < labels.size() - 1; i++) {
                        ArrayList<Float> curLocLabel = new ArrayList<>();
                        for (Pixel<T, L> p : pixelsLCP) {
                                if (proba[i][p.getPointer()] >= 0) {
                                        pixelsLCP.get(curLocLabel.size()).local_seed = p;
                                        curLocLabel.add(new Float(proba[i][p.getPointer()]));
                                }
                        }
                        local_labels.add(curLocLabel);
                }
                int numOfLocalLabels = local_labels.get(local_labels.size() - 1).size();

                for (int i = 0; i < numOfLocalLabels; i++) {
                        seeded_vertex[pixelsLCP.get(i).local_seed.indic_VP] = true;
                }

                // The system to solve is A x = -B X2
                // building matrix A : laplacian for unseeded nodes
                Matrix A2_m = new Matrix(pixelsLCP.size() - numOfLocalLabels, pixelsLCP.size() - numOfLocalLabels);
                fill_A(A2_m, pixelsLCP.size(), numOfLocalLabels, edgesLCP, seeded_vertex, indic_sparse, nb_same_edges);

                // building boundary matrix B
                Matrix B2_m = new Matrix(pixelsLCP.size() - numOfLocalLabels, numOfLocalLabels);
                fill_B(B2_m, edgesLCP, seeded_vertex, indic_sparse, nb_same_edges);
                LUDecomposition AXB = new LUDecomposition(A2_m);

                // building the right hand side of the system
                for (int l = 0; l < labels.size() - 1; l++) {
                        Matrix X = new Matrix(numOfLocalLabels, 1);
                        double[] b = new double[pixelsLCP.size() - numOfLocalLabels];
                        // building vector X
                        for (int i = 0; i < numOfLocalLabels; i++) {
                                X.set(i, 0, local_labels.get(l).get(i).floatValue());
                        }

                        Matrix b_tmp = B2_m.times(X);

                        Arrays.fill(b, 0);

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
                        for (int k = 0; k < pixelsLCP.size(); k++) {
                                if (!seeded_vertex[k]) {
                                        proba[l][pixelsLCP.get(k).getPointer()] = (float) b[cpt];
                                        cpt++;
                                }
                        }
                        // Enforce boundaries exactly
                        for (int k = 0; k < numOfLocalLabels; k++) {
                                proba[l][pixelsLCP.get(pixelsLCP.get(k).local_seed.indic_VP).getPointer()] = local_labels.get(l).get(k).floatValue();
                        }
                }
        }

        private void fill_B(Matrix B, ArrayList<PseudoEdge<T, L>> edgesLCP, boolean[] seeded_vertex, int[] indic_sparse, int[] nb_same_edges) {
                for (int k = 0; k < edgesLCP.size(); k++) {
                        int p1 = edgesLCP.get(k).p1.indic_VP;
                        int p2 = edgesLCP.get(k).p2.indic_VP;
                        if (seeded_vertex[p1] == true) {
                                B.set(indic_sparse[p2], indic_sparse[p1], -nb_same_edges[k] - 1);
                                k += nb_same_edges[k];
                        } else if (seeded_vertex[p2] == true) {
                                B.set(indic_sparse[p1], indic_sparse[p2], -nb_same_edges[k] - 1);
                                k += nb_same_edges[k];
                        }
                }

        }

        private void fill_A(Matrix A, int N, int numOfLocalLabels, ArrayList<PseudoEdge<T, L>> edgesLCP, boolean[] seeded_vertex, int[] indic_sparse,
                        int[] nb_same_edges) {
                // fill the diagonal
                int rnz = 0;
                for (int k = 0; k < N; k++) {
                        if (!seeded_vertex[k]) {
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
                for (int k = 0; k < edgesLCP.size(); k++) {
                        int p1 = edgesLCP.get(k).p1.indic_VP;
                        int p2 = edgesLCP.get(k).p2.indic_VP;
                        if (!seeded_vertex[p1] && !seeded_vertex[p2]) {
                                A.set(indic_sparse[p1], indic_sparse[p2], -nb_same_edges[k] - 1);
                                A.set(indic_sparse[p2], indic_sparse[p1], -nb_same_edges[k] - 1);
                        }
                        k += nb_same_edges[k];
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
