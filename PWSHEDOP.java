package org.knime.knip.example;

import java.util.ArrayList;
import java.util.Collections;
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

        @Parameter(type = ItemIO.OUTPUT)
        private Labeling<L> output;

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
                while (seedCursor.hasNext()) {
                        seedCursor.fwd();
                        if (seedCursor.get().getLabeling().size() != 0) {
                                L label = seedCursor.get().getLabeling().get(0);
                                seedsL.add(new Pixel<T, L>(seedCursor.getIntPosition(0), seedCursor.getIntPosition(1), label, width));
                                if (!labels.contains(label)) {
                                        labels.add(label);
                                }
                        }
                }
                gPixels = new Pixel[numOfPixels];
                gPixelsT = new Pixel[width][height];
                for (int i = 0; i < width; i++) {
                        for (int j = 0; j < height; j++) {
                                gPixelsT[i][j] = new Pixel<T, L>(i, j, labels.get(0), width);
                                gPixels[i + width * j] = gPixelsT[i][j];
                        }
                }
                edges = new ArrayList<>();
                Edge<T, L>[][] hor_edges = new Edge[width - 1][height];
                Edge<T, L>[][] ver_edges = new Edge[width][height - 1];

                for (int i = 0; i < height - 1; i++) {
                        for (int j = 0; j < width; j++) {
                                ver_edges[j][i] = new Edge<T, L>(j, i, j, i + 1, edges.size());
                                ver_edges[j][i].p1 = gPixelsT[j][i];
                                ver_edges[j][i].p2 = gPixelsT[j][i + 1];
                                edges.add(ver_edges[j][i]);
                        }
                }
                for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width - 1; j++) {
                                hor_edges[j][i] = new Edge<T, L>(j, i, j + 1, i, edges.size());
                                hor_edges[j][i].p1 = gPixelsT[j][i];
                                hor_edges[j][i].p2 = gPixelsT[j + 1][i];
                                edges.add(hor_edges[j][i]);
                        }
                }
                for (int i = 0; i < numOfEdges; i++) {
                        Edge<T, L> e = edges.get(i);
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
                for (Pixel<T, L> p : seedsL) {
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
                        imgRA.setPosition(e.n1x, 0);
                        imgRA.setPosition(e.n1y, 1);
                        int v1 = imgRA.get().getInteger();
                        imgRA.setPosition(e.n2x, 0);
                        imgRA.setPosition(e.n2y, 1);
                        int v2 = imgRA.get().getInteger();
                        e.normal_weight = 255 - Math.abs(v1 - v2);
                }
                //                }
                int[] seeds_function = new int[numOfEdges];
                for (Pixel<T, L> p : seedsL) {
                        for (Edge<T, L> e : p.neighbors) {
                                if (e != null) {
                                        seeds_function[e.number] = e.normal_weight;
                                }
                        }
                }
                for (Edge<T, L> e : edges) {
                        e.weight = seeds_function[e.number];
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
                                e.Mrk = true;
                        }
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
                        for (int j = 0; j < numOfPixels; j++) {
                                proba[i][j] = -1;
                        }
                }
                Pixel<T, L>[][] edgesLCP = new Pixel[2][numOfEdges];
                // proba[i][j] =1 <=> pixel[i] has label j+1
                for (Pixel<T, L> pix : seedsL) {
                        for (int j = 0; j < labels.size() - 1; j++) {
                                if (pix.label == labels.get(j + 1)) {
                                        proba[j][pix.pointer] = 1;
                                } else {
                                        proba[j][pix.pointer] = 0;
                                }
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
                        Stack<Edge<T, L>> LCP = new Stack<>();
                        ArrayList<Pixel<T, L>> Plateau = new ArrayList<>(); // vertices of a
                                                                            // plateau.
                        LIFO.add(e_max);
                        e_max.visitedPlateau = true;
                        e_max.visited = true;
                        LCP.add(e_max);
                        int nb_edges = 0;
                        int wmax = e_max.weight;
                        Edge<T, L> x;
                        ArrayList<Edge<T, L>> sorted_weights2 = new ArrayList<Edge<T, L>>();

                        // 2. putting the edges and vertices of the plateau into arrays
                        while (!LIFO.empty()) {
                                x = LIFO.pop();
                                Pixel<T, L> re1 = x.p1.find();
                                Pixel<T, L> re2 = x.p2.find();
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

                                for (Edge<T, L> edge : x.neighbors) {
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

                        for (Pixel<T, L> p : Plateau) {
                                p.visited = false;
                        }
                        for (Edge<T, L> e : LCP) {
                                e.visitedPlateau = false;
                        }

                        // 3. If e_max belongs to a plateau
                        if (nb_edges > 0) {
                                // 4. Evaluate if there are differents seeds on the plateau
                                boolean different_seeds = false;

                                for (int i = 0; i < labels.size() - 1; i++) {
                                        int p = 0;
                                        double val = -0.5;
                                        for (Pixel<T, L> j : Plateau) {
                                                int xr = j.find().pointer;
                                                if (Math.abs(proba[i][xr] - val) > epsilon && proba[i][xr] >= 0) {
                                                        p++;
                                                        val = proba[i][xr];
                                                        if (p == 2) {
                                                                break;
                                                        }
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
                                        Edge.weights = false;
                                        Collections.sort(sorted_weights2);
                                        Collections.reverse(sorted_weights2);

                                        // Merge nodes for edges of real max weight
                                        Plateau.clear();
                                        int Nnb_edges = 0;
                                        for (Edge<T, L> Ne_max : sorted_weights2) {
                                                Pixel<T, L> re1 = Ne_max.p1.find();
                                                Pixel<T, L> re2 = Ne_max.p2.find();
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
                                                for (Pixel<T, L> xr : Plateau) {
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
                                                        Pixel<T, L> e1 = edgesLCP[0][j];
                                                        Pixel<T, L> e2 = edgesLCP[1][j];
                                                        merge_node(e1.find(), e2.find());
                                                }
                                        }

                                        for (Pixel<T, L> pix : Plateau) {
                                                pix.visited = false;
                                        }
                                } else {
                                        // if different seeds = false
                                        // 7. Merge nodes for edges of max weight
                                        for (int j = 0; j < nb_edges; j++) {
                                                Pixel<T, L> e1 = edgesLCP[0][j];
                                                Pixel<T, L> e2 = edgesLCP[1][j];
                                                merge_node(e1.find(), e2.find());
                                        }
                                }
                        }
                        LCP.clear();
                } // end main loop

                // building the final proba map (find the root vertex of each tree)
                for (Pixel<T, L> j : gPixels) {
                        Pixel<T, L> i = j;
                        Pixel<T, L> xr = i.Fth;
                        while (i.Fth != i) {
                                i = xr;
                                xr = i.Fth;
                        }
                        for (int k = 0; k < labels.size() - 1; k++) {
                                proba[k][j.pointer] = proba[k][i.pointer];
                        }
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
        private void RandomWalker(Pixel<T, L>[][] index_edges, int M, ArrayList<Pixel<T, L>> index, float[][] boundary_values, int numb_boundary) {
                ArrayList<PseudoEdge<T, L>> edgeL = new ArrayList<>();
                for (int j = 0; j < M; j++) {
                        edgeL.add(new PseudoEdge<T, L>(index_edges[0][j], index_edges[1][j]));
                }
                boolean[] seeded_vertex = new boolean[index.size()];
                int[] indic_sparse = new int[index.size()];
                int[] nb_same_edges = new int[M];

                // Indexing the edges, and the seeds
                for (int i = 0; i < index.size(); i++) {
                        index.get(i).indic_VP = gPixels[i];
                }

                for (int j = 0; j < M; j++) {
                        Pixel<T, L> v1 = edgeL.get(j).p1.indic_VP;
                        Pixel<T, L> v2 = edgeL.get(j).p2.indic_VP;
                        if (v1.pointer < v2.pointer) {
                                edgeL.get(j).p1 = edgeL.get(j).p1.indic_VP;
                                edgeL.get(j).p2 = edgeL.get(j).p2.indic_VP;
                                indic_sparse[edgeL.get(j).p1.pointer]++;
                                indic_sparse[edgeL.get(j).p2.pointer]++;
                        } else {
                                edgeL.get(j).p2 = v1;
                                edgeL.get(j).p1 = v2;
                                indic_sparse[edgeL.get(j).p1.pointer]++;
                                indic_sparse[edgeL.get(j).p2.pointer]++;
                        }
                }
                Collections.sort(edgeL);
                for (int m = 0; m < M; m++) {
                        int n = 0;
                        while ((m + n < M - 1) && edgeL.get(m + n).compareTo(edgeL.get(m + n + 1)) == 0) {
                                n++;
                        }
                        nb_same_edges[m] = n;
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

                // building boundary matrix B
                Matrix B2_m = new Matrix(index.size() - numb_boundary, numb_boundary);
                fill_B(B2_m, index.size(), M, numb_boundary, index_edges, seeded_vertex, indic_sparse, nb_same_edges);
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
                                if (seeded_vertex[k] == false) {
                                        proba[l][index.get(k).pointer] = (float) b[cpt];
                                        cpt++;
                                }
                        }
                        // Enforce boundaries exactly
                        for (int k = 0; k < numb_boundary; k++) {
                                proba[l][index.get(gPixels[k].local_seed.pointer).pointer] = boundary_values[l][k];
                        }
                }
        }

        private void fill_B(Matrix B, int N, int M, int numb_boundary, Pixel<T, L>[][] index_edges, boolean[] seeded_vertex, int[] indic_sparse,
                        int[] nb_same_edges) {
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

        private void fill_A(Matrix A, int N, int M, int numb_boundary, Pixel<T, L>[][] index_edges, boolean[] seeded_vertex, int[] indic_sparse,
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

        private void merge_node(Pixel<T, L> e1, Pixel<T, L> e2) {
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

}
