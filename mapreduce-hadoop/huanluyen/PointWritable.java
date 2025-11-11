import java.io.*;
import org.apache.hadoop.io.Writable;

public class PointWritable implements Writable {
    public double[] sum;   // tổng các chiều
    public long count;     // số điểm

    public PointWritable() {}

    public PointWritable(double[] v, long c) {
        this.sum = v;
        this.count = c;
    }

    public static PointWritable fromPoint(double[] v) {
        double[] s = new double[v.length];
        System.arraycopy(v, 0, s, 0, v.length);
        return new PointWritable(s, 1L);
    }

    public void addInPlace(PointWritable other) {
        if (this.sum == null) {
            this.sum = new double[other.sum.length];
            this.count = 0L;
        }
        for (int i = 0; i < this.sum.length; i++) this.sum[i] += other.sum[i];
        this.count += other.count;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(sum == null ? 0 : sum.length);
        if (sum != null) {
            for (double v : sum) out.writeDouble(v);
        }
        out.writeLong(count);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        int n = in.readInt();
        sum = new double[n];
        for (int i = 0; i < n; i++) sum[i] = in.readDouble();
        count = in.readLong();
    }
}
