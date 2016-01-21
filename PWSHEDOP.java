package net.imagej.ops.labeling.watershed;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.scijava.ItemIO;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import net.imagej.ops.AbstractOp;
import net.imagej.ops.Op;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.roi.labeling.LabelingType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

@Plugin(type = Op.class)
public class PowerWatershedOp<T extends RealType<T>, L extends Comparable<L>> extends AbstractOp {

    private static class Edge implements Comparable<Edge> {
        final double normal_weight;
        double weight = 0;
        Set<Edge> neighbors;
        boolean visited;
        final long p1;
        final long p2;
        Edge Fth = this;
        boolean Mrk;

        static boolean weights;

        Edge(long p1, long p2, double normal_weight) {
            this.p1 = p1;
            this.p2 = p2;
            this.normal_weight = normal_weight;
        }

        Edge find() {
            if (Fth != this) {
                Fth = Fth.find();
            }
            return Fth;
        }

        @Override
        public int compareTo(Edge e) {
            int result = 0;
            if (weights) {
                if (weight < e.weight) {
                    result = 1;
                } else if (weight > e.weight) {
                    result = -1;
                } else {
                    result = 0;
                }
            } else {
                if (normal_weight < e.normal_weight) {
                    result = 1;
                } else if (normal_weight > e.normal_weight) {
                    result = -1;
                } else {
                    result = 0;
                }
            }
            return result;
        }
    }

    private class PseudoEdge implements Comparable<PseudoEdge> {
        long p1;
        long p2;

        PseudoEdge(long p, long q) {
            p1 = p;
            p2 = q;
        }

        @Override
        public int compareTo(PseudoEdge p) {
            if (indic_VP[(int) p1] < indic_VP[(int) p.p1]) {
                return -1;
            } else if (indic_VP[(int) p1] > indic_VP[(int) p.p1]) {
                return 1;
            } else {
                if (indic_VP[(int) p2] < indic_VP[(int) p.p2]) {
                    return -1;
                } else if (indic_VP[(int) p2] > indic_VP[(int) p.p2]) {
                    return 1;
                } else {
                    return 0;
                }
            }
        }
    }

    private static int SIZE_MAX_PLATEAU = 10000;
    private static double EPSILON = 0.000001;

    @Parameter(type = ItemIO.INPUT)
    private RandomAccessibleInterval<T> image;

    @Parameter(type = ItemIO.INPUT)
    private RandomAccessibleInterval<LabelingType<L>> seeds;

    @Parameter(type = ItemIO.BOTH)
    private RandomAccessibleInterval<LabelingType<L>> output;

    float[][] proba;
    int[] rnk;
    int[] indic_VP;
    HashMap<Integer, L> int2label;
    HashMap<L, Integer> label2int;
    long[] Fth;
    int[] pixLabel;

    @Override
    public void run() {

        checkInput();

        long dimensions[] = new long[image.numDimensions()];
        image.dimensions(dimensions);

        long[][] edgeDimensions = new long[dimensions.length][dimensions.length];
        for (int i = 0; i < dimensions.length; i++) {
            edgeDimensions[i] = dimensions.clone();
            edgeDimensions[i][i] -= 1;
        }

        final T maxVal = Views.iterable(image).firstElement().createVariable();
        maxVal.setReal(maxVal.getMaxValue());
        double max = 100000;// 0000;

        final Cursor<LabelingType<L>> seedCursor = Views.iterable(seeds).localizingCursor();
        Set<Long> seedsL = new HashSet<Long>();
        int2label = new HashMap<>();
        label2int = new HashMap<>();

        /*
         * Create a "Pixel" for each pixel in the input Get the seeds from the
         * input labeling. "labels" is the List of the labels, while "seedsL"
         * stores the seeds. Create the edges.
         */
        long numOfPixels = 1;
        for (long d : dimensions) {
            numOfPixels *= d;
        }
        Fth = new long[(int) numOfPixels];
        rnk = new int[(int) numOfPixels];
        indic_VP = new int[(int) numOfPixels];
        pixLabel = new int[(int) numOfPixels];

        long[] edgeOffset = new long[dimensions.length];
        edgeOffset[0] = 0;
        for (int i = 1; i < edgeOffset.length; i++) {
            edgeOffset[i] = edgeOffset[i - 1] + numOfPixels - numOfPixels / dimensions[i - 1];
        }

        long numOfEdges = 0;
        for (int i = 0; i < dimensions.length; i++) {
            numOfEdges += numOfPixels - numOfPixels / dimensions[i];
        }
        Edge[] allEdges = new Edge[(int) numOfEdges];

        final Cursor<T> imageCursor = Views.iterable(image).localizingCursor();
        double[] lastSlice = new double[(int) (numOfPixels / dimensions[dimensions.length - 1])];

        for (int pointer = 0; pointer < numOfPixels; pointer++) {
            Fth[pointer] = pointer;
            LabelingType<L> labeling = seedCursor.next();
            if (labeling.size() != 0) {
                L label = labeling.iterator().next();
                if (!label2int.containsKey(label)) {
                    pixLabel[pointer] = seedsL.size();
                    label2int.put(label, pixLabel[pointer]);
                    int2label.put(pixLabel[pointer], label);
                } else {
                    pixLabel[pointer] = label2int.get(label);
                }
                seedsL.add((long) pointer);
            }
            double currentPixel = imageCursor.next().getRealDouble();
            int[] coords = toCoordinates(pointer, dimensions);
            for (int i = 0; i < dimensions.length; i++) {
                if (coords[i] > 0) {
                    coords[i] -= 1;
                    int tmp = coords[coords.length - 1];
                    coords[coords.length - 1] = 0;
                    double normal_weight = max - Math.abs(lastSlice[toPointer(coords, dimensions)] - currentPixel);
                    coords[coords.length - 1] = tmp;
                    Edge e = new Edge(toPointer(coords, dimensions), pointer, normal_weight);
                    if (seedsL.contains(e.p1) || seedsL.contains(e.p2)) {
                        // The edges connected to seeds get the weight set as
                        // their normal_weight
                        e.weight = normal_weight;
                    }
                    allEdges[(int) (edgeOffset[i] + toPointer(coords, edgeDimensions[i]))] = e;
                    coords[i] += 1;
                }
            }
            int tmp = coords[coords.length - 1];
            coords[coords.length - 1] = 0;
            lastSlice[toPointer(coords, dimensions)] = currentPixel;
            coords[coords.length - 1] = tmp;
        }
        ArrayList<Edge> edges = new ArrayList<Edge>(Arrays.asList(allEdges));

        Edge.weights = false;
        Collections.sort(edges);
        // heaviest first
        for (Edge e : edges) {
            /*
             * get the neighbor-information
             */
            int[] coord1 = toCoordinates((int) e.p1, dimensions);
            int[] coord2 = toCoordinates((int) e.p2, dimensions);
            e.neighbors = new HashSet<>();
            for (int i = 0; i < dimensions.length; i++) {
                if (coord1[i] > 0) {
                    coord1[i] -= 1;
                    Edge neighbor = allEdges[(int) (edgeOffset[i] + toPointer(coord1, edgeDimensions[i]))];
                    coord1[i] += 1;
                    if (e != neighbor) {
                        e.neighbors.add(neighbor);
                    }
                }
                if (coord1[i] < edgeDimensions[i][i]) {
                    Edge neighbor = allEdges[(int) (edgeOffset[i] + toPointer(coord1, edgeDimensions[i]))];
                    if (e != neighbor) {
                        e.neighbors.add(neighbor);
                    }
                }
                if (coord2[i] > 0) {
                    coord2[i] -= 1;
                    Edge neighbor = allEdges[(int) (edgeOffset[i] + toPointer(coord2, edgeDimensions[i]))];
                    coord2[i] += 1;
                    if (e != neighbor) {
                        e.neighbors.add(neighbor);
                    }
                }
                if (coord2[i] < edgeDimensions[i][i]) {
                    Edge neighbor = allEdges[(int) (edgeOffset[i] + toPointer(coord2, edgeDimensions[i]))];
                    if (e != neighbor) {
                        e.neighbors.add(neighbor);
                    }
                }
            }
            // go through the neighbors
            for (Edge n : e.neighbors) {
                // if the neighbor has already been visited (same or higher
                // normal_weight)
                if (n.Mrk) {
                    // get the root
                    Edge r = n.find();
                    // if the root is not the current edge
                    if (r != e) {
                        // if the edges have the same normal_weight OR this edge
                        // has a higher normal_weight than the other root
                        if ((e.normal_weight == r.normal_weight) || (e.normal_weight >= r.weight)) {
                            // set this as the root of the other root
                            r.Fth = e;
                            // and give it the weight of the old root if it's
                            // heavier than this
                            if (r.weight > e.weight) {
                                e.weight = r.weight;
                            }
                        } else {
                            // if they have different normal_weights AND the
                            // root is less heavy than the normal_weight
                            e.weight = max;
                        }
                    }
                }
            }
            e.Mrk = true;
        }
        // set weight to normal_weight of roots (via backtracing)
        for (int i = edges.size() - 1; i >= 0; i--) {
            Edge e = edges.get(i);
            if (e.Fth == e) {
                // p is root
                if (e.weight == max) {
                    e.weight = e.normal_weight;
                }
            } else {
                e.weight = e.Fth.weight;
            }
        }

        proba = new float[int2label.size() - 1][(int) numOfPixels];
        for (float[] labelProb : proba) {
            Arrays.fill(labelProb, -1);
        }
        // proba[i][j] =1 <=> pixel[i] has label j
        for (long pix : seedsL) {
            for (int j = 0; j < int2label.size() - 1; j++) {
                proba[j][(int) pix] = int2label.get(pixLabel[(int) pix]) == int2label.get(j) ? 1 : 0;
            }
        }

        Edge.weights = true;
        Collections.sort(edges);
        for (Edge e_max : edges) {
            if (!e_max.visited) {
                PowerWatershed(e_max);
            }
        }

        // building the final proba map (find the root vertex of each tree)
        for (int j = 0; j < numOfPixels; j++) {
            long i = find(j);
            if (i != j) {
                for (float[] labelProb : proba) {
                    labelProb[(int) j] = labelProb[(int) i];
                }
            }
        }

        Cursor<LabelingType<L>> outCursor = Views.iterable(output).localizingCursor();
        for (int j = 0; j < numOfPixels; j++) {
            outCursor.fwd();
            double maxi = 0;
            int argmax = 0;
            double val = 1;
            for (int k = 0; k < int2label.size() - 1; k++) {
                if (proba[k][j] > maxi) {
                    maxi = proba[k][j];
                    argmax = k;
                }
                val = val - proba[k][j];

            }
            if (val > maxi) {
                argmax = int2label.size() - 1;
            }
            outCursor.get().clear();
            outCursor.get().add(int2label.get(argmax));
        }

    }

    private void PowerWatershed(Edge e_max) {
        // 1. Computing the edges of the plateau LCP linked to the edge e_max
        Stack<Edge> LIFO = new Stack<>();
        HashSet<Edge> visited = new HashSet<>();
        LIFO.add(e_max);
        e_max.visited = true;
        visited.add(e_max);
        ArrayList<Edge> sorted_weights = new ArrayList<Edge>();

        // 2. putting the edges and vertices of the plateau into arrays
        while (!LIFO.empty()) {
            Edge x = LIFO.pop();
            if (proba[0][(int) find(x.p1)] < 0 || proba[0][(int) find(x.p2)] < 0) {
                sorted_weights.add(x);
            }

            for (Edge neighbor : x.neighbors) {
                if ((neighbor.weight == e_max.weight) && visited.add(neighbor)) {
                    LIFO.add(neighbor);
                    neighbor.visited = true;
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
                for (Edge x : sorted_weights) {
                    int xr = (int) find(x.p1);
                    if (Math.abs(proba[i][xr] - val) > EPSILON && proba[i][xr] >= 0) {
                        p++;
                        val = proba[i][xr];
                    }
                    xr = (int) find(x.p2);
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

                // Merge nodes for edges of real max weight
                ArrayList<Long> pixelsLCP = new ArrayList<>(); // vertices
                                                               // of a
                                                               // plateau.
                ArrayList<PseudoEdge> edgesLCP = new ArrayList<>();
                for (Edge e : sorted_weights) {
                    long re1 = find(e.p1);
                    long re2 = find(e.p2);
                    if (e.normal_weight != e_max.weight) {
                        merge_node(re1, re2);
                    } else if ((re1 != re2) && (proba[0][(int) re1] < 0 || proba[0][(int) re2] < 0)) {
                        if (!pixelsLCP.contains(re1)) {
                            pixelsLCP.add(re1);
                        }
                        if (!pixelsLCP.contains(re2)) {
                            pixelsLCP.add(re2);
                        }
                        edgesLCP.add(new PseudoEdge(re1, re2));

                    }
                }

                // 6. Execute Random Walker on plateaus
                if (pixelsLCP.size() < SIZE_MAX_PLATEAU) {
                    RandomWalker(edgesLCP, pixelsLCP);
                } else {
                    System.out.printf("Plateau too big (%d vertices,%d edges), RW is not performed\n", pixelsLCP.size(),
                            edgesLCP.size());
                    for (PseudoEdge pseudo : edgesLCP) {
                        merge_node(find(pseudo.p1), find(pseudo.p2));
                    }
                }
            } else {
                // if different seeds = false
                // 7. Merge nodes for edges of max weight
                for (Edge edge : sorted_weights) {
                    merge_node(find(edge.p1), find(edge.p2));
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
    private void RandomWalker(ArrayList<PseudoEdge> edgesLCP, ArrayList<Long> pixelsLCP) {

        int[] indic_sparse = new int[pixelsLCP.size()];
        int[] numOfSameEdges = new int[edgesLCP.size()];
        ArrayList<Long> local_seeds = new ArrayList<>();

        // Indexing the edges, and the seeds
        for (int i = 0; i < pixelsLCP.size(); i++) {
            indic_VP[pixelsLCP.get(i).intValue()] = i;
        }

        for (PseudoEdge pseudo : edgesLCP) {
            if (indic_VP[(int) pseudo.p1] > indic_VP[(int) pseudo.p2]) {
                long tmp = pseudo.p1;
                pseudo.p1 = pseudo.p2;
                pseudo.p2 = tmp;
            }
            indic_sparse[indic_VP[(int) pseudo.p1]]++;
            indic_sparse[indic_VP[(int) pseudo.p2]]++;
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
            for (long p : pixelsLCP) {
                if (proba[i][(int) p] >= 0) {
                    if (local_seeds.size() <= curLocLabel.size()) {
                        local_seeds.add(p);
                    }
                    curLocLabel.add(new Float(proba[i][(int) p]));
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
        Array2DRowRealMatrix A = new Array2DRowRealMatrix(numOfUnseededNodes, numOfUnseededNodes);
        Array2DRowRealMatrix B = new Array2DRowRealMatrix(numOfUnseededNodes, numOfSeededNodes);

        // fill the diagonal
        int rnz = 0;
        for (long p : pixelsLCP) {
            if (!local_seeds.contains(p)) {
                A.setEntry(rnz, rnz, indic_sparse[indic_VP[(int) p]]);
                rnz++;
            }
        }
        // A(i,i)=n means there are n edges on this plateau that are incident to
        // node i
        int rnzs = 0;
        int rnzu = 0;
        for (long p : pixelsLCP) {
            if (local_seeds.contains(p)) {
                indic_sparse[indic_VP[(int) p]] = rnzs;
                rnzs++;
            } else {
                indic_sparse[indic_VP[(int) p]] = rnzu;
                rnzu++;
            }
        }

        for (int k = 0; k < edgesLCP.size(); k++) {
            PseudoEdge e = edgesLCP.get(k);
            int p1 = indic_VP[(int) e.p1];
            int p2 = indic_VP[(int) e.p2];
            if (!local_seeds.contains(e.p1) && !local_seeds.contains(e.p2)) {
                A.setEntry(indic_sparse[p1], indic_sparse[p2], -numOfSameEdges[k] - 1);
                A.setEntry(indic_sparse[p2], indic_sparse[p1], -numOfSameEdges[k] - 1);
            } else if (local_seeds.contains(e.p1)) {
                B.setEntry(indic_sparse[p2], indic_sparse[p1], -numOfSameEdges[k] - 1);
            } else if (local_seeds.contains(e.p2)) {
                B.setEntry(indic_sparse[p1], indic_sparse[p2], -numOfSameEdges[k] - 1);
            }
        }
        // A(i,j)=n means that node ith and jth unseeded nodes are connected by
        // -n-1 edges
        // B(i,j)=n means that the ith unseeded and the jth seeded node are
        // connected by -n-1 edges

        LUDecomposition AXB = new LUDecomposition(A);

        // building the right hand side of the system
        for (int l = 0; l < proba.length; l++) {
            Array2DRowRealMatrix X = new Array2DRowRealMatrix(numOfSeededNodes, 1);
            // building vector X
            for (int i = 0; i < numOfSeededNodes; i++) {
                X.setEntry(i, 0, local_labels[l][i]);
            }

            Array2DRowRealMatrix BX = B.multiply(X);

            double[] b = new double[numOfUnseededNodes];

            for (int i = 0; i < numOfUnseededNodes; i++) {
                for (int j = 0; j < BX.getColumnDimension(); j++) {
                    if (BX.getEntry(i, j) != 0) {
                        b[i] = -BX.getEntry(i, j);
                    }
                }
            }
            // solve Ax=b by LU decomposition, order = 1

            b = AXB.getSolver().solve(new Array2DRowRealMatrix(b)).getColumnVector(0).toArray();

            int cpt = 0;
            for (long p : pixelsLCP) {
                if (!local_seeds.contains(p)) {
                    proba[l][(int) p] = (float) b[cpt];
                    cpt++;
                }
            }
            // Enforce boundaries exactly
            for (int k = 0; k < numOfSeededNodes; k++) {
                proba[l][local_seeds.get(k).intValue()] = local_labels[l][k];
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
    private void merge_node(long p1, long p2) {
        // merge if p1!=p2 and one of them has no probability yet
        if ((p1 != p2) && (proba[0][(int) p1] < 0 || proba[0][(int) p2] < 0)) {
            // link p1 and p2;
            // the Pixel with the smaller Rnk points to the other
            // if both have the same rank increase the rnk of p2
            if (rnk[(int) p1] > rnk[(int) p2]) {
                Fth[(int) p2] = p1;
            } else {
                if (rnk[(int) p1] == rnk[(int) p2]) {
                    rnk[(int) p2]++;
                }
                Fth[(int) p1] = p2;
            }

            // which one has proba[0] < 0? Fill proba[_][ex] with proba[_][ey]
            if (proba[0][(int) p1] < 0) {
                for (float[] labelProb : proba) {
                    labelProb[(int) p1] = labelProb[(int) p2];
                }
            } else {
                for (float[] labelProb : proba) {
                    labelProb[(int) p2] = labelProb[(int) p1];
                }
            }
        }
    }

    private long find(long f) {
        if (Fth[(int) f] != f) {
            return find(Fth[(int) f]);
        }
        return f;
    }

    /**
     * Check if input is valid
     *
     * @param image
     * @param seeds
     * @param output
     */
    private void checkInput() {
        if (seeds.numDimensions() != image.numDimensions()) {
            throw new IllegalArgumentException(String.format(
                    "The dimensionality of the seed labeling (%dD) does not match that of the intensity image (%dD)",
                    seeds.numDimensions(), image.numDimensions()));
        }
        if (seeds.numDimensions() != output.numDimensions()) {
            throw new IllegalArgumentException(String.format(
                    "The dimensionality of the seed labeling (%dD) does not match that of the output labeling (%dD)",
                    seeds.numDimensions(), output.numDimensions()));
        }
        for (int i = 0; i < seeds.numDimensions(); i++) {
            if (seeds.dimension(i) != image.dimension(i) || seeds.dimension(i) != output.dimension(i)) {
                throw new IllegalArgumentException("only images with identical size are supported right now");
            }
        }
    }

    private int[] toCoordinates(int pointer, long[] dim) {
        int[] coord = new int[dim.length];
        int index = pointer;
        for (int i = 0; i < dim.length; i++) {
            coord[i] = (int) (index % dim[i]);
            index /= dim[i];
        }
        return coord;
    }

    private int toPointer(int[] coord, long[] dim) {
        int pointer = coord[0];
        int mult = (int) dim[0];
        for (int i = 1; i < coord.length; i++) {
            pointer += coord[i] * mult;
            mult *= dim.length > i ? (int) dim[i] : 1;
        }
        return pointer;
    }

}
