package net.imagej.ops.labeling.watershed;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
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

    private static int SIZE_MAX_PLATEAU = 10000;
    private static double EPSILON = 0.000001;

    @Parameter(type = ItemIO.INPUT)
    private RandomAccessibleInterval<T> image;

    @Parameter(type = ItemIO.INPUT)
    private RandomAccessibleInterval<LabelingType<L>> seeds;

    @Parameter(type = ItemIO.BOTH)
    private RandomAccessibleInterval<LabelingType<L>> output;

    float[][] proba;

    @Override
    public void run() {

        checkInput();

        long dimensions[] = new long[image.numDimensions()];
        image.dimensions(dimensions);
        Edge.dimensions = dimensions;
        Pixel.dimensions = dimensions;
        long[] position = new long[dimensions.length];

        final T maxVal = Views.iterable(image).firstElement().createVariable();
        maxVal.setReal(maxVal.getMaxValue());
        double max = 1000000000;

        final Cursor<LabelingType<L>> seedCursor = Views.iterable(seeds).localizingCursor();
        ArrayList<Pixel<T, L>> seedsL = new ArrayList<>();
        ArrayList<L> labels = new ArrayList<L>();

        /*
         * Create a "Pixel" for each pixel in the input Get the seeds from the
         * input labeling. "labels" is the List of the labels, while "seedsL"
         * stores the seeds. Create the edges.
         */
        long numOfPixels = 1;
        for (long d : dimensions) {
            numOfPixels *= d;
        }
        Pixel<T, L>[] gPixelsT = new Pixel[(int) numOfPixels];
        ArrayList<Edge<T, L>> edges = new ArrayList<>();

        long[] edgeDimesions = new long[dimensions.length];
        for (int i = 0; i < edgeDimesions.length; i++) {
            edgeDimesions[i] = numOfPixels - numOfPixels / dimensions[i];
        }

        long numOfEdges = 0;
        for (long d : edgeDimesions) {
            numOfEdges += d;
        }
        Edge<T, L>[] allEdges = new Edge[(int) numOfEdges];

        final Cursor<T> imageCursor = Views.iterable(image).localizingCursor();
        double[] lastSlice = new double[(int) (dimensions[0] * dimensions[1])];

        for (int pointer = 0; pointer < gPixelsT.length; pointer++) {
            LabelingType<L> labeling = seedCursor.next();
            if (labeling.size() != 0) {
                L label = labeling.iterator().next();
                gPixelsT[pointer] = new Pixel<T, L>(pointer, label);
                seedsL.add(gPixelsT[pointer]);
                if (!labels.contains(label)) {
                    labels.add(label);
                }
            } else {
                gPixelsT[pointer] = new Pixel<T, L>(pointer, null);
            }
            double currentPixel = imageCursor.next().getRealDouble();
            if (pointer % (dimensions[0] * dimensions[1]) >= dimensions[0]) {
                double normal_weight = max - Math.abs(
                        lastSlice[(int) (pointer % (dimensions[0] * dimensions[1]) - dimensions[0])] - currentPixel);
                // Vertical Edge
                allEdges[(int) (edgeDimesions[0] + pointer
                        - dimensions[0]
                                * (1 + Math.floor(pointer / (dimensions[0] * dimensions[1]))))] = new Edge<T, L>(
                                        gPixelsT[(int) (pointer - dimensions[0])], gPixelsT[pointer], normal_weight);
                edges.add(allEdges[(int) (edgeDimesions[0] + pointer
                        - dimensions[0] * (1 + Math.floor(pointer / (dimensions[0] * dimensions[1]))))]);
            }
            if (pointer >= (dimensions[0] * dimensions[1])) {
                double normal_weight = max
                        - Math.abs(lastSlice[(int) (pointer % (dimensions[0] * dimensions[1]))] - currentPixel);
                // depth edge
                allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + pointer
                        - (dimensions[0] * dimensions[1]))] = new Edge<T, L>(
                                gPixelsT[(int) (pointer - (dimensions[0] * dimensions[1]))], gPixelsT[pointer],
                                normal_weight);
                edges.add(allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + pointer
                        - (dimensions[0] * dimensions[1]))]);
            }
            lastSlice[(int) (pointer % (dimensions[0] * dimensions[1]))] = currentPixel;
            if (pointer % dimensions[0] > 0) {
                double normal_weight = max - Math.abs(lastSlice[(int) (pointer % (dimensions[0] * dimensions[1]) - 1)]
                        - lastSlice[(int) (pointer % (dimensions[0] * dimensions[1]))]);
                // horizontal edge
                allEdges[(int) (pointer - 1 - Math.floor((pointer % (dimensions[0] * dimensions[1])) / dimensions[0])
                        - dimensions[1] * Math.floor(pointer / (dimensions[0] * dimensions[1])))] = new Edge<T, L>(
                                gPixelsT[pointer - 1], gPixelsT[pointer], normal_weight);
                edges.add(allEdges[(int) (pointer - 1
                        - Math.floor((pointer % (dimensions[0] * dimensions[1])) / dimensions[0])
                        - dimensions[1] * Math.floor(pointer / (dimensions[0] * dimensions[1])))]);
            }
        }

        /*
         * get the neighbor-information
         */
        for (Edge<T, L> e : edges) {
            long z1 = Math.floorDiv(e.p1.getPointer(), (dimensions[0] * dimensions[1]));
            if (!e.isVertical()) {
                if (!e.isDepth()) {
                    if (Math.floorDiv((e.p1.getPointer() % (dimensions[0] * dimensions[1])), dimensions[0]) > 0) {
                        e.neighbors[0] = allEdges[(int) (edgeDimesions[0] + e.p2.getPointer() - dimensions[0]
                                - Math.floorDiv(e.p2.getPointer(), (dimensions[0] * dimensions[1])) * dimensions[0])];
                        e.neighbors[1] = allEdges[(int) (edgeDimesions[0] + e.p1.getPointer() - dimensions[0]
                                - Math.floorDiv(e.p1.getPointer(), (dimensions[0] * dimensions[1])) * dimensions[0])];
                    }
                    if (e.p1.getPointer() % dimensions[0] > 0) {
                        e.neighbors[2] = allEdges[(int) (e.p1.getPointer() - 1
                                - Math.floorDiv(e.p1.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                                - z1 * dimensions[1])];
                    }
                    if (Math.floorDiv((e.p1.getPointer() % (dimensions[0] * dimensions[1])),
                            dimensions[0]) < dimensions[1] - 1) {
                        e.neighbors[3] = allEdges[(int) (edgeDimesions[0] + e.p1.getPointer()
                                - dimensions[0] * Math.floorDiv(e.p1.getPointer(), (dimensions[0] * dimensions[1])))];
                        e.neighbors[4] = allEdges[(int) (edgeDimesions[0] + e.p2.getPointer()
                                - dimensions[0] * Math.floorDiv(e.p2.getPointer(), (dimensions[0] * dimensions[1])))];
                    }
                    if (e.p2.getPointer() % dimensions[0] < dimensions[0] - 1) {
                        e.neighbors[5] = allEdges[(int) (e.p2.getPointer()
                                - Math.floorDiv(e.p2.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                                - Math.floorDiv(e.p2.getPointer(), (dimensions[0] * dimensions[1])) * dimensions[1])];
                    }
                    if (z1 > 0) {
                        e.neighbors[6] = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + e.p1.getPointer()
                                - (dimensions[0] * dimensions[1]))];
                        e.neighbors[7] = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + e.p2.getPointer()
                                - (dimensions[0] * dimensions[1]))];
                    }
                    if (z1 < (dimensions.length > 2 ? (int) dimensions[2] : 1) - 1) {
                        e.neighbors[8] = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + e.p1.getPointer())];
                        e.neighbors[9] = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + e.p2.getPointer())];
                    }
                } else {
                    // e.isDepth()
                    if (e.p1.getPointer() % dimensions[0] > 0) {
                        e.neighbors[0] = allEdges[(int) (e.p1.getPointer() - 1
                                - Math.floorDiv(e.p1.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                                - z1 * dimensions[1])];
                        e.neighbors[1] = allEdges[(int) (e.p2.getPointer() - 1
                                - Math.floorDiv(e.p2.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                                - Math.floorDiv(e.p2.getPointer(), (dimensions[0] * dimensions[1])) * dimensions[1])];
                    }
                    if (Math.floorDiv((e.p1.getPointer() % (dimensions[0] * dimensions[1])), dimensions[0]) > 0) {
                        e.neighbors[2] = allEdges[(int) (edgeDimesions[0] + e.p1.getPointer() - dimensions[0]
                                - Math.floorDiv(e.p1.getPointer(), (dimensions[0] * dimensions[1])) * dimensions[0])];
                        e.neighbors[3] = allEdges[(int) (edgeDimesions[0] + e.p2.getPointer() - dimensions[0]
                                - Math.floorDiv(e.p2.getPointer(), (dimensions[0] * dimensions[1])) * dimensions[0])];
                    }
                    if (z1 > 0) {
                        e.neighbors[4] = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + e.p1.getPointer()
                                - (dimensions[0] * dimensions[1]))];
                    }
                    if (e.p1.getPointer() % dimensions[0] < dimensions[0] - 1) {
                        e.neighbors[5] = allEdges[(int) (e.p1.getPointer()
                                - Math.floorDiv(e.p1.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                                - z1 * dimensions[1])];
                        e.neighbors[6] = allEdges[(int) (e.p2.getPointer()
                                - Math.floorDiv(e.p2.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                                - Math.floorDiv(e.p2.getPointer(), (dimensions[0] * dimensions[1])) * dimensions[1])];
                    }
                    if (Math.floorDiv((e.p1.getPointer() % (dimensions[0] * dimensions[1])),
                            dimensions[0]) < dimensions[1] - 1) {
                        e.neighbors[7] = allEdges[(int) (edgeDimesions[0] + e.p1.getPointer()
                                - dimensions[0] * Math.floorDiv(e.p1.getPointer(), (dimensions[0] * dimensions[1])))];
                        e.neighbors[8] = allEdges[(int) (edgeDimesions[0] + e.p2.getPointer()
                                - dimensions[0] * Math.floorDiv(e.p2.getPointer(), (dimensions[0] * dimensions[1])))];
                    }
                    if (Math.floorDiv(e.p2.getPointer(),
                            (dimensions[0] * dimensions[1])) < (dimensions.length > 2 ? (int) dimensions[2] : 1) - 1) {
                        e.neighbors[9] = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + e.p2.getPointer())];
                    }
                }
            } else { // e.isVertical()
                if (Math.floorDiv((e.p1.getPointer() % (dimensions[0] * dimensions[1])), dimensions[0]) > 0) {
                    e.neighbors[0] = allEdges[(int) (edgeDimesions[0] + e.p1.getPointer() - dimensions[0]
                            - Math.floorDiv(e.p1.getPointer(), (dimensions[0] * dimensions[1])) * dimensions[0])];
                }
                if (e.p1.getPointer() % dimensions[0] > 0) {
                    e.neighbors[1] = allEdges[(int) (e.p1.getPointer() - 1
                            - Math.floorDiv(e.p1.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                            - z1 * dimensions[1])];
                    e.neighbors[2] = allEdges[(int) (e.p2.getPointer() - 1
                            - Math.floorDiv(e.p2.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                            - Math.floorDiv(e.p2.getPointer(), (dimensions[0] * dimensions[1])) * dimensions[1])];
                }
                if (Math.floorDiv((e.p2.getPointer() % (dimensions[0] * dimensions[1])), dimensions[0]) < dimensions[1]
                        - 1) {
                    e.neighbors[3] = allEdges[(int) (edgeDimesions[0] + e.p2.getPointer()
                            - dimensions[0] * Math.floorDiv(e.p2.getPointer(), (dimensions[0] * dimensions[1])))];
                }
                if (e.p1.getPointer() % dimensions[0] < dimensions[0] - 1) {
                    e.neighbors[4] = allEdges[(int) (e.p2.getPointer()
                            - Math.floorDiv(e.p2.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                            - Math.floorDiv(e.p2.getPointer(), (dimensions[0] * dimensions[1])) * dimensions[1])];
                    e.neighbors[5] = allEdges[(int) (e.p1.getPointer()
                            - Math.floorDiv(e.p1.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                            - z1 * dimensions[1])];
                }
                if (z1 > 0) {
                    e.neighbors[6] = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + e.p1.getPointer()
                            - (dimensions[0] * dimensions[1]))];
                    e.neighbors[7] = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + e.p2.getPointer()
                            - (dimensions[0] * dimensions[1]))];
                }
                if (z1 < (dimensions.length > 2 ? (int) dimensions[2] : 1) - 1) {
                    e.neighbors[8] = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + e.p1.getPointer())];
                    e.neighbors[9] = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + e.p2.getPointer())];
                }
            }
        }

        /*
         * The edges connected to seeds get the weight set as their
         * normal_weight
         */
        for (Pixel<T, L> p : seedsL) {
            if (p.getPointer() % dimensions[0] < dimensions[0] - 1) {
                // to the right
                allEdges[(int) (p.getPointer()
                        - Math.floorDiv(p.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                        - Math.floorDiv(p.getPointer(), (dimensions[0] * dimensions[1]))
                                * dimensions[1])].weight = allEdges[(int) (p.getPointer()
                                        - Math.floorDiv(p.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                                        - Math.floorDiv(p.getPointer(), (dimensions[0] * dimensions[1]))
                                                * dimensions[1])].normal_weight;
            }
            if (Math.floorDiv((p.getPointer() % (dimensions[0] * dimensions[1])), dimensions[0]) < dimensions[1] - 1) {
                // to the bottom
                allEdges[(int) (edgeDimesions[0] + p.getPointer()
                        - dimensions[0] * Math.floorDiv(p.getPointer(),
                                (dimensions[0] * dimensions[1])))].weight = allEdges[(int) (edgeDimesions[0]
                                        + p.getPointer() - dimensions[0] * Math.floorDiv(p.getPointer(),
                                                (dimensions[0] * dimensions[1])))].normal_weight;
            }
            if (Math.floorDiv(p.getPointer(),
                    (dimensions[0] * dimensions[1])) < (dimensions.length > 2 ? (int) dimensions[2] : 1) - 1) {
                // to the back
                allEdges[(int) (edgeDimesions[0] + edgeDimesions[1]
                        + p.getPointer())].weight = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1]
                                + p.getPointer())].normal_weight;
            }
            if (p.getPointer() % dimensions[0] > 0) {
                // to the left
                allEdges[(int) (p.getPointer() - 1
                        - Math.floorDiv(p.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                        - Math.floorDiv(p.getPointer(), (dimensions[0] * dimensions[1]))
                                * dimensions[1])].weight = allEdges[(int) (p.getPointer() - 1
                                        - Math.floorDiv(p.getPointer() % (dimensions[0] * dimensions[1]), dimensions[0])
                                        - Math.floorDiv(p.getPointer(), (dimensions[0] * dimensions[1]))
                                                * dimensions[1])].normal_weight;
            }
            if (Math.floorDiv((p.getPointer() % (dimensions[0] * dimensions[1])), dimensions[0]) > 0) {
                // to the top
                allEdges[(int) (edgeDimesions[0] + p.getPointer() - dimensions[0]
                        - dimensions[0] * Math.floorDiv(p.getPointer(),
                                (dimensions[0] * dimensions[1])))].weight = allEdges[(int) (edgeDimesions[0]
                                        + p.getPointer() - dimensions[0] - dimensions[0] * Math.floorDiv(p.getPointer(),
                                                (dimensions[0] * dimensions[1])))].normal_weight;
            }
            if (Math.floorDiv(p.getPointer(), (dimensions[0] * dimensions[1])) > 0) {
                // to the front
                allEdges[(int) (edgeDimesions[0] + edgeDimesions[1] + p.getPointer()
                        - (dimensions[0] * dimensions[1]))].weight = allEdges[(int) (edgeDimesions[0] + edgeDimesions[1]
                                + p.getPointer() - (dimensions[0] * dimensions[1]))].normal_weight;
            }
        }

        Edge.weights = false;
        Edge.ascending = false;
        Collections.sort(edges);
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
                            e.weight = max;
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
                if (e.weight == max) {
                    e.weight = e.normal_weight;
                }
            } else {
                e.weight = e.Fth.weight;
            }
        }

        proba = new float[labels.size() - 1][(int) numOfPixels];
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
        Edge.ascending = false;
        Collections.sort(edges);

        for (Edge<T, L> e_max : edges) {
            if (e_max.visited) {
                continue;
            }
            PowerWatershed(e_max);
        }

        // building the final proba map (find the root vertex of each tree)
        for (Pixel<T, L> j : gPixelsT) {
            Pixel<T, L> i = j.find();
            if (i != j) {
                for (float[] labelProb : proba) {
                    labelProb[j.getPointer()] = labelProb[i.getPointer()];
                }
            }
        }

        Cursor<LabelingType<L>> outCursor = Views.iterable(output).localizingCursor();
        for (int j = 0; j < numOfPixels; j++) {
            outCursor.fwd();
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
                Edge.ascending = false;
                Collections.sort(sorted_weights);

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
        Array2DRowRealMatrix A = new Array2DRowRealMatrix(numOfUnseededNodes, numOfUnseededNodes);
        Array2DRowRealMatrix B = new Array2DRowRealMatrix(numOfUnseededNodes, numOfSeededNodes);

        // fill the diagonal
        int rnz = 0;
        for (Pixel<T, L> p : pixelsLCP) {
            if (!local_seeds.contains(p)) {
                A.setEntry(rnz, rnz, indic_sparse[p.indic_VP]);
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
        if (seeds.numDimensions() > 3) {
            throw new IllegalArgumentException("more than 3 dimensions not yet supported");
        } else {
            if (seeds.numDimensions() > 2) {
                if (seeds.dimension(2) != image.dimension(2) || seeds.dimension(2) != output.dimension(2)) {
                    throw new IllegalArgumentException("only images with identical size are supported right now");
                }
            }
            if (seeds.numDimensions() > 1) {
                if (seeds.dimension(1) != image.dimension(1) || seeds.dimension(1) != output.dimension(1)) {
                    throw new IllegalArgumentException("only images with identical size are supported right now");
                }
            }
            if (seeds.numDimensions() > 0) {
                if (seeds.dimension(0) != image.dimension(0) || seeds.dimension(0) != output.dimension(0)) {
                    throw new IllegalArgumentException("only images with identical size are supported right now");
                }
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
