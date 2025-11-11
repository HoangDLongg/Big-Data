import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;

public class KMapper extends Mapper<LongWritable, Text, IntWritable, PointWritable> {

    private double[][] centroids;
    private String sep;
    private boolean skipHeaderHeuristic;
    private final IntWritable outKey = new IntWritable();

    @Override
    protected void setup(Context ctx) {
        sep = ctx.getConfiguration().get("input.sep", ",");
        skipHeaderHeuristic = ctx.getConfiguration().getBoolean("skip.header.heuristic", true);
        String ser = ctx.getConfiguration().get("centroids");
        centroids = parseCentroids(ser);
    }

    private static double[][] parseCentroids(String ser) {
        // "v1,v2,...|v1,v2,..."
        String[] parts = ser.split("\\|");
        double[][] res = new double[parts.length][];
        for (int i = 0; i < parts.length; i++) {
            String[] p = parts[i].split(",");
            res[i] = new double[p.length];
            for (int j = 0; j < p.length; j++) res[i][j] = Double.parseDouble(p[j]);
        }
        return res;
    }

    private static boolean isHeaderLine(String s) {
        if (s == null || s.isEmpty()) return false;
        char c0 = s.charAt(0);
        return (c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z');
    }

    private static double[] parseVector(String line, String sep) {
        String[] p = line.split(sep, -1);
        double[] v = new double[p.length];
        for (int i = 0; i < p.length; i++) v[i] = Double.parseDouble(p[i]);
        return v;
    }

    private static int argmin(double[] v, double[][] cents) {
        int best = 0;
        double bestd = Double.MAX_VALUE;
        for (int k = 0; k < cents.length; k++) {
            double d = 0;
            for (int j = 0; j < v.length; j++) {
                double diff = v[j] - cents[k][j];
                d += diff * diff;
            }
            if (d < bestd) { bestd = d; best = k; }
        }
        return best;
    }

    @Override
    protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
        String line = value.toString().trim();
        if (line.isEmpty()) return;
        if (skipHeaderHeuristic && isHeaderLine(line)) return;

        try {
            double[] v = parseVector(line, sep);
            int cid = argmin(v, centroids);
            outKey.set(cid);
            ctx.write(outKey, PointWritable.fromPoint(v));
        } catch (Exception ignore) {
            // bỏ qua các dòng lỗi parse
        }
    }
}
