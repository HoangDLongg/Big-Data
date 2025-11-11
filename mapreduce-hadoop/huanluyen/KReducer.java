import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;

public class KReducer extends Reducer<IntWritable, PointWritable, Writable, Text> {

    private String mode; // "centroids" or "json"

    @Override
    protected void setup(Context ctx) {
        mode = ctx.getConfiguration().get("output.mode", "centroids");
    }

    @Override
    protected void reduce(IntWritable key, Iterable<PointWritable> values, Context ctx)
            throws IOException, InterruptedException {

        double[] sum = null;
        long count = 0L;

        for (PointWritable pw : values) {
            if (sum == null) sum = new double[pw.sum.length];
            for (int i = 0; i < sum.length; i++) sum[i] += pw.sum[i];
            count += pw.count;
        }

        if (sum == null || count == 0) return;

        double[] mean = new double[sum.length];
        for (int i = 0; i < sum.length; i++) mean[i] = sum[i] / (double) count;

        if ("json".equals(mode)) {
            // xuất JSON một centroid/line
            // giả định thứ tự cột: longitude,latitude,elevation,max_temperature,min_temperature,precipitation,wind,relative_humidity
            String json = String.format(
                "{\"cluster\":%d,\"longitude\":%s,\"latitude\":%s,\"elevation\":%s," +
                "\"max_temperature\":%s,\"min_temperature\":%s,\"precipitation\":%s," +
                "\"wind\":%s,\"relative_humidity\":%s,\"count\":%d}",
                key.get(),
                fmt(mean, 0), fmt(mean, 1), fmt(mean, 2),
                fmt(mean, 3), fmt(mean, 4), fmt(mean, 5),
                fmt(mean, 6), fmt(mean, 7), count
            );
            ctx.write(NullWritable.get(), new Text(json));
        } else {
            // mode "centroids": "cid \t v1,v2,..."
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < mean.length; i++) {
                if (i > 0) sb.append(",");
                sb.append(mean[i]);
            }
            ctx.write(key, new Text(sb.toString()));
        }
    }

    private static String fmt(double[] a, int i) {
        if (i < 0 || i >= a.length) return "0";
        return Double.toString(a[i]);
    }
}
