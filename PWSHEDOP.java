package net.imagej.ops.labeling.watershed;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Stack;

import org.scijava.ItemIO;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import Jama.LUDecomposition;
import Jama.Matrix;
import net.imagej.ops.AbstractOp;
import net.imagej.ops.Op;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.roi.labeling.LabelingType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

@Plugin(type = Op.class)
public class PWSHEDOP<T extends RealType<T>, L extends Comparable<L>> extends AbstractOp {

    private static int SIZE_MAX_PLATEAU = 1000000;
    private static double EPSILON = 0.000001;

    @Parameter(type = ItemIO.INPUT)
    private RandomAccessibleInterval<T> image;

    @Parameter(type = ItemIO.INPUT)
    private RandomAccessibleInterval<LabelingType<L>> seeds;

    @Parameter(type = ItemIO.BOTH)
    private RandomAccessibleInterval<LabelingType<L>> output;

    int width;
    int height;
    int depth;
    float[][] proba;

    @Override
    public void run() {

        // save the width for easy access
        width = (int) image.dimension(0);
        Pixel.width = width;
        // save the height for easy access
        height = (int) image.dimension(1);
        Pixel.height = height;
        // save the depth for easy access
        depth = (int) image.dimension(2);

        @SuppressWarnings("deprecation")
        final Cursor<LabelingType<L>> seedCursor = Views.iterable(seeds).localizingCursor();
        ArrayList<Pixel<T, L>> seedsL = new ArrayList<>();
        ArrayList<L> labels = new ArrayList<L>();

        /*
         * Create a "Pixel" for each pixel in the input Get the seeds from the
         * input labeling. "labels" is the List of the labels, while "seedsL"
         * stores the seeds. Create the edges.
         */
        Pixel<T, L>[][][] gPixelsT = new Pixel[width][height][depth];
        ArrayList<Edge<T, L>> edges = new ArrayList<>();
        Edge<T, L>[][][] hor_edges = new Edge[width - 1][height][depth];
        Edge<T, L>[][][] ver_edges = new Edge[width][height - 1][depth];
        Edge<T, L>[][][] dep_edges = new Edge[width][height][depth - 1];
        final Cursor<T> imageCursor = Views.iterable(image).localizingCursor();
        double[][] lastSlice = new double[width][height];
        for (int k = 0; k < depth; k++) {
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    if (!seedCursor.hasNext()) {
                        System.out.println("fuck");
                    }
                    seedCursor.fwd();
                    LabelingType<L> labeling = seedCursor.get();
                    if (labeling.size() != 0) {
                        L label = labeling.iterator().next();
                        gPixelsT[i][j][k] = new Pixel<T, L>(i, j, k, label);
                        seedsL.add(gPixelsT[i][j][k]);
                        if (!labels.contains(label)) {
                            labels.add(label);
                        }
                    } else {
                        gPixelsT[i][j][k] = new Pixel<T, L>(i, j, k, null);
                    }

                    if (!imageCursor.hasNext()) {
                        System.out.println("fuck");
                    }
                    imageCursor.fwd();
                    double currentPixel = imageCursor.get().getRealDouble();
                    if (j > 0) {
                        double normal_weight = 255 - Math.abs(lastSlice[i][j - 1] - currentPixel);
                        ver_edges[i][j - 1][k] = new Edge<T, L>(gPixelsT[i][j - 1][k], gPixelsT[i][j][k],
                                normal_weight);
                        edges.add(ver_edges[i][j - 1][k]);
                    }
                    if (k > 0) {
                        double normal_weight = 255 - Math.abs(lastSlice[i][j] - currentPixel);
                        dep_edges[i][j][k - 1] = new Edge<T, L>(gPixelsT[i][j][k - 1], gPixelsT[i][j][k],
                                normal_weight);
                        edges.add(dep_edges[i][j][k - 1]);
                    }
                    lastSlice[i][j] = currentPixel;
                    if (i > 0) {
                        double normal_weight = 255 - Math.abs(lastSlice[i - 1][j] - lastSlice[i][j]);
                        hor_edges[i - 1][j][k] = new Edge<T, L>(gPixelsT[i - 1][j][k], gPixelsT[i][j][k],
                                normal_weight);
                        edges.add(hor_edges[i - 1][j][k]);
                    }
                }
            }
        }

        /*
         * get the neighbor-information
         */
        for (Edge<T, L> e : edges) {
            if (!e.isVertical()) {
                if (!e.isDepth()) {
                    if (e.p1.getY() > 0) {
                        e.neighbors[0] = ver_edges[e.p2.getX()][e.p1.getY() - 1][e.p1.getZ()];
                        e.neighbors[1] = ver_edges[e.p1.getX()][e.p1.getY() - 1][e.p1.getZ()];
                    }
                    if (e.p1.getX() > 0) {
                        e.neighbors[2] = hor_edges[e.p1.getX() - 1][e.p1.getY()][e.p1.getZ()];
                    }
                    if (e.p1.getY() < ver_edges[0].length) {
                        e.neighbors[3] = ver_edges[e.p1.getX()][e.p1.getY()][e.p1.getZ()];
                        e.neighbors[4] = ver_edges[e.p2.getX()][e.p1.getY()][e.p1.getZ()];
                    }
                    if (e.p2.getX() < hor_edges.length) {
                        e.neighbors[5] = hor_edges[e.p2.getX()][e.p1.getY()][e.p1.getZ()];
                    }
                    if (e.p1.getZ() > 0) {
                        e.neighbors[6] = dep_edges[e.p1.getX()][e.p1.getY()][e.p1.getZ() - 1];
                        e.neighbors[7] = dep_edges[e.p2.getX()][e.p2.getY()][e.p2.getZ() - 1];
                    }
                    if (e.p1.getZ() < dep_edges[0][0].length) {
                        e.neighbors[8] = dep_edges[e.p1.getX()][e.p1.getY()][e.p1.getZ()];
                        e.neighbors[9] = dep_edges[e.p2.getX()][e.p2.getY()][e.p2.getZ()];
                    }
                } else {
                    // e.isDepth()
                    if (e.p1.getX() > 0) {
                        e.neighbors[0] = hor_edges[e.p1.getX() - 1][e.p1.getY()][e.p1.getZ()];
                        e.neighbors[1] = hor_edges[e.p2.getX() - 1][e.p2.getY()][e.p2.getZ()];
                    }
                    if (e.p1.getY() > 0) {
                        e.neighbors[2] = ver_edges[e.p1.getX()][e.p1.getY() - 1][e.p1.getZ()];
                        e.neighbors[3] = ver_edges[e.p2.getX()][e.p2.getY() - 1][e.p2.getZ()];
                    }
                    if (e.p1.getZ() > 0) {
                        e.neighbors[4] = dep_edges[e.p1.getX()][e.p1.getY()][e.p1.getZ() - 1];
                    }
                    if (e.p1.getX() < hor_edges.length) {
                        e.neighbors[5] = hor_edges[e.p1.getX()][e.p1.getY()][e.p1.getZ()];
                        e.neighbors[6] = hor_edges[e.p2.getX()][e.p2.getY()][e.p2.getZ()];
                    }
                    if (e.p2.getY() < ver_edges[0].length) {
                        e.neighbors[7] = ver_edges[e.p1.getX()][e.p1.getY()][e.p1.getZ()];
                        e.neighbors[8] = ver_edges[e.p2.getX()][e.p2.getY()][e.p2.getZ()];
                    }
                    if (e.p2.getZ() < dep_edges[0][0].length) {
                        e.neighbors[9] = dep_edges[e.p2.getX()][e.p2.getY()][e.p2.getZ()];
                    }
                }
            } else { // e.isVertical()
                if (e.p1.getY() > 0) {
                    e.neighbors[0] = ver_edges[e.p1.getX()][e.p1.getY() - 1][e.p1.getZ()];
                }
                if (e.p1.getX() > 0) {
                    e.neighbors[1] = hor_edges[e.p1.getX() - 1][e.p1.getY()][e.p1.getZ()];
                    e.neighbors[2] = hor_edges[e.p2.getX() - 1][e.p2.getY()][e.p1.getZ()];
                }
                if (e.p2.getY() < ver_edges[0].length) {
                    e.neighbors[3] = ver_edges[e.p2.getX()][e.p2.getY()][e.p1.getZ()];
                }
                if (e.p1.getX() < hor_edges.length) {
                    e.neighbors[4] = hor_edges[e.p2.getX()][e.p2.getY()][e.p1.getZ()];
                    e.neighbors[5] = hor_edges[e.p1.getX()][e.p1.getY()][e.p1.getZ()];
                }
                if (e.p1.getZ() > 0) {
                    e.neighbors[6] = dep_edges[e.p1.getX()][e.p1.getY()][e.p1.getZ() - 1];
                    e.neighbors[7] = dep_edges[e.p2.getX()][e.p2.getY()][e.p2.getZ() - 1];
                }
                if (e.p1.getZ() < dep_edges[0][0].length) {
                    e.neighbors[8] = dep_edges[e.p1.getX()][e.p1.getY()][e.p1.getZ()];
                    e.neighbors[9] = dep_edges[e.p2.getX()][e.p2.getY()][e.p2.getZ()];
                }
            }
        }

        /*
         * The edges connected to seeds get the weight set as their
         * normal_weight
         */
        for (Pixel<T, L> p : seedsL) {
            if (p.getX() < hor_edges.length) {
                hor_edges[p.getX()][p.getY()][p.getZ()].weight = hor_edges[p.getX()][p.getY()][p.getZ()].normal_weight;
            }
            if (p.getY() < ver_edges[0].length) {
                ver_edges[p.getX()][p.getY()][p.getZ()].weight = ver_edges[p.getX()][p.getY()][p.getZ()].normal_weight;
            }
            if (p.getZ() < dep_edges[0][0].length) {
                ver_edges[p.getX()][p.getY()][p.getZ()].weight = ver_edges[p.getX()][p.getY()][p.getZ()].normal_weight;
            }
            if (p.getX() > 0) {
                hor_edges[p.getX() - 1][p.getY()][p.getZ()].weight = hor_edges[p.getX() - 1][p.getY()][p
                        .getZ()].normal_weight;
            }
            if (p.getY() > 0) {
                ver_edges[p.getX()][p.getY() - 1][p.getZ()].weight = ver_edges[p.getX()][p.getY() - 1][p
                        .getZ()].normal_weight;
            }
            if (p.getZ() > 0) {
                dep_edges[p.getX()][p.getY()][p.getZ() - 1].weight = dep_edges[p.getX()][p.getY()][p.getZ()
                        - 1].normal_weight;
            }
        }

        Edge.weights = false;
        Collections.sort(edges);
        Collections.reverse(edges);
        // heaviest first
        for (Edge<T, L> e : edges) {
            // go through the neighbors
            for (Edge<T, L> n : e.neighbors) {
                // if the neighbor has already been visited (same or higher
                // normal_weight)
                if (n != null && n.Mrk) {
                    // get the root
                    Edge<T, L> r = n.find();
                    // if the root is not the current edge
                    if (r != e) {
                        // if the edges have the same normal_weight OR this edge
                        // has a higher normal_weight than the other root
                        if ((e.normal_weight == r.normal_weight) || (e.normal_weight >= r.weight)) {
                            // set this as the root of the other root
                            r.Fth = e;
                            // and give it the weight of the old root if it's
                            // heavier than this
                            e.weight = Math.max(r.weight, e.weight);
                        } else {
                            // if they have different normal_weights AND the
                            // root is less heavy than the normal_weight
                            e.weight = 255;
                        }
                    }
                }
            }
            e.Mrk = true;
        }
        // set weight to normal_weight of roots (via backtracing)
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

        proba = new float[labels.size() - 1][width * height * depth];
        for (float[] labelProb : proba) {
            Arrays.fill(labelProb, -1);
        }
        // proba[i][j] =1 <=> pixel[i] has label j
        for (Pixel<T, L> pix : seedsL) {
            int pixPointer = pix.getPointer();
            for (int j = 0; j < labels.size() - 1; j++) {
                proba[j][pixPointer] = pix.label == labels.get(j) ? 1 : 0;
            }
        }

        Edge.weights = true;
        Collections.sort(edges);
        Collections.reverse(edges);

        for (Edge<T, L> e_max : edges) {
            if (e_max.visited) {
                continue;
            }
            PowerWatershed(e_max);
        }

        // building the final proba map (find the root vertex of each tree)
        for (Pixel<T, L>[][] gPixelsT2 : gPixelsT) {
            for (Pixel<T, L>[] gPixels : gPixelsT2) {
                for (Pixel<T, L> j : gPixels) {
                    Pixel<T, L> i = j.find();
                    if (i != j) {
                        for (float[] labelProb : proba) {
                            labelProb[j.getPointer()] = labelProb[i.getPointer()];
                        }
                    }
                }
            }
        }

        @SuppressWarnings("deprecation")
        Cursor<LabelingType<L>> outCursor = Views.iterable(output).localizingCursor();
        for (int j = 0; j < width * height * depth; j++) {
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
            outCursor.get().clear();
            outCursor.get().add(labels.get(argmax));
            outCursor.fwd();
        }

    }

    private void PowerWatershed(Edge<T, L> e_max) {
        // 1. Computing the edges of the plateau LCP linked to the edge e_max
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

            for (int i = 0; i < proba.length; i++) {
                int p = 0;
                double val = -0.5;
                for (Edge<T, L> x : sorted_weights) {
                    int xr = x.p1.find().getPointer();
                    if (Math.abs(proba[i][xr] - val) > EPSILON && proba[i][xr] >= 0) {
                        p++;
                        val = proba[i][xr];
                    }
                    xr = x.p2.find().getPointer();
                    if (Math.abs(proba[i][xr] - val) > EPSILON && proba[i][xr] >= 0) {
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
                ArrayList<Pixel<T, L>> pixelsLCP = new ArrayList<>(); // vertices
                                                                      // of a
                                                                      // plateau.
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
    }

    /**
     * 
     * @param edgesLCP
     *            edges of the plateau
     * @param pixelsLCP
     *            nodes of the plateau
     */
    private void RandomWalker(ArrayList<PseudoEdge<T, L>> edgesLCP, ArrayList<Pixel<T, L>> pixelsLCP) {

        int[] indic_sparse = new int[pixelsLCP.size()];
        int[] numOfSameEdges = new int[edgesLCP.size()];
        ArrayList<Pixel<T, L>> local_seeds = new ArrayList<>();

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
            while ((m < edgesLCP.size() - 1) && edgesLCP.get(m).compareTo(edgesLCP.get(m + 1)) == 0) {
                edgesLCP.remove(m + 1);
                numOfSameEdges[m]++;
            }
        }

        ArrayList<ArrayList<Float>> localLabelsList = new ArrayList<>();
        for (int i = 0; i < proba.length; i++) {
            ArrayList<Float> curLocLabel = new ArrayList<>();
            for (Pixel<T, L> p : pixelsLCP) {
                if (proba[i][p.getPointer()] >= 0) {
                    if (local_seeds.size() <= curLocLabel.size()) {
                        local_seeds.add(p);
                    }
                    curLocLabel.add(new Float(proba[i][p.getPointer()]));
                }
            }
            localLabelsList.add(curLocLabel);
        }
        float[][] local_labels = new float[localLabelsList.size()][localLabelsList.get(0).size()];
        for (int i = 0; i < localLabelsList.size(); i++) {
            for (int j = 0; j < localLabelsList.get(i).size(); j++) {
                local_labels[i][j] = localLabelsList.get(i).get(j).floatValue();
            }
        }
        int numOfSeededNodes = local_labels[0].length;
        int numOfUnseededNodes = pixelsLCP.size() - numOfSeededNodes;

        // The system to solve is A x = -B X2
        // building matrix A : laplacian for unseeded nodes
        // building boundary matrix B
        Matrix A = new Matrix(numOfUnseededNodes, numOfUnseededNodes);
        Matrix B = new Matrix(numOfUnseededNodes, numOfSeededNodes);

        // fill the diagonal
        int rnz = 0;
        for (Pixel<T, L> p : pixelsLCP) {
            if (!local_seeds.contains(p)) {
                A.set(rnz, rnz, indic_sparse[p.indic_VP]);
                rnz++;
            }
        }
        // A(i,i)=n means there are n edges on this plateau that are incident to
        // node i
        int rnzs = 0;
        int rnzu = 0;
        for (Pixel<T, L> p : pixelsLCP) {
            if (local_seeds.contains(p)) {
                indic_sparse[p.indic_VP] = rnzs;
                rnzs++;
            } else {
                indic_sparse[p.indic_VP] = rnzu;
                rnzu++;
            }
        }

        for (int k = 0; k < edgesLCP.size(); k++) {
            PseudoEdge<T, L> e = edgesLCP.get(k);
            int p1 = e.p1.indic_VP;
            int p2 = e.p2.indic_VP;
            if (!local_seeds.contains(e.p1) && !local_seeds.contains(e.p2)) {
                A.set(indic_sparse[p1], indic_sparse[p2], -numOfSameEdges[k] - 1);
                A.set(indic_sparse[p2], indic_sparse[p1], -numOfSameEdges[k] - 1);
            } else if (local_seeds.contains(e.p1)) {
                B.set(indic_sparse[p2], indic_sparse[p1], -numOfSameEdges[k] - 1);
            } else if (local_seeds.contains(e.p2)) {
                B.set(indic_sparse[p1], indic_sparse[p2], -numOfSameEdges[k] - 1);
            }
        }
        // A(i,j)=n means that node ith and jth unseeded nodes are connected by
        // -n-1 edges
        // B(i,j)=n means that the ith unseeded and the jth seeded node are
        // connected by -n-1 edges

        LUDecomposition AXB = new LUDecomposition(A);

        // building the right hand side of the system
        for (int l = 0; l < proba.length; l++) {
            Matrix X = new Matrix(numOfSeededNodes, 1);
            // building vector X
            for (int i = 0; i < numOfSeededNodes; i++) {
                X.set(i, 0, local_labels[l][i]);
            }

            Matrix BX = B.times(X);

            double[] b = new double[numOfUnseededNodes];

            for (int i = 0; i < numOfUnseededNodes; i++) {
                for (int j = 0; j < BX.getColumnDimension(); j++) {
                    if (BX.get(i, j) != 0) {
                        b[i] = -BX.get(i, j);
                    }
                }
            }
            // solve Ax=b by LU decomposition, order = 1

            b = AXB.solve(new Matrix(b, numOfUnseededNodes)).getColumnPackedCopy();

            int cpt = 0;
            for (Pixel<T, L> p : pixelsLCP) {
                if (!local_seeds.contains(p)) {
                    proba[l][p.getPointer()] = (float) b[cpt];
                    cpt++;
                }
            }
            // Enforce boundaries exactly
            for (int k = 0; k < numOfSeededNodes; k++) {
                proba[l][local_seeds.get(k).getPointer()] = local_labels[l][k];
            }
        }
    }

    /**
     * merges two pixels as in the else-clause of the algorithm by Camille
     * Couprie
     * 
     * @param p1
     *            first pixel
     * @param p2
     *            second pixel
     */
    private void merge_node(Pixel<T, L> p1, Pixel<T, L> p2) {
        // merge if p1!=p2 and one of them has no probability yet
        if ((p1 != p2) && (proba[0][p1.getPointer()] < 0 || proba[0][p2.getPointer()] < 0)) {
            // link p1 and p2;
            // the Pixel with the smaller Rnk points to the other
            // if both have the same rank increase the rnk of p2
            if (p1.Rnk > p2.Rnk) {
                p2.Fth = p1;
            } else {
                if (p1.Rnk == p2.Rnk) {
                    p2.Rnk++;
                }
                p1.Fth = p2;
            }

            // which one has proba[0] < 0? Fill proba[_][ex] with proba[_][ey]
            if (proba[0][p1.getPointer()] < 0) {
                for (float[] labelProb : proba) {
                    labelProb[p1.getPointer()] = labelProb[p2.getPointer()];
                }
            } else {
                for (float[] labelProb : proba) {
                    labelProb[p2.getPointer()] = labelProb[p1.getPointer()];
                }
            }
        }
    }

}
