import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;

public class KCombiner extends Reducer<IntWritable, PointWritable, IntWritable, PointWritable> {
    @Override
    protected void reduce(IntWritable key, Iterable<PointWritable> values, Context ctx)
            throws IOException, InterruptedException {
        PointWritable acc = null;
        for (PointWritable pw : values) {
            if (acc == null) {
                acc = new PointWritable(new double[pw.sum.length], 0L);
            }
            acc.addInPlace(pw);
        }
        if (acc != null) ctx.write(key, acc);
    }
}
