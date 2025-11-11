import java.io.*;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class Main {

    private static void parseLegacyDArgsIntoConf(String[] appArgs, Configuration conf) {
        Map<String,String> map = new HashMap<>();
        for (int i = 0; i < appArgs.length; i++) {
            String a = appArgs[i];
            if (a.startsWith("-D") && a.contains("=")) {
                int eq = a.indexOf('=');
                map.put(a.substring(2, eq).trim(), a.substring(eq + 1).trim());
            } else if (a.startsWith("-D")) {
                String k = a.substring(2).trim();
                if (i+1 < appArgs.length && !appArgs[i+1].startsWith("-")) { map.put(k, appArgs[++i]); }
            } else if (a.matches("^-(in|out|k|thresh|NumReduceTask|maxloop|lines|result|input\\.sep)$")) {
                String k = a.substring(1);
                if (i+1 < appArgs.length && !appArgs[i+1].startsWith("-")) {
                    String v = appArgs[++i];
                    if ("maxloop".equals(k)) k = "max.iter";
                    map.put(k, v);
                }
            }
        }
        for (Map.Entry<String,String> e : map.entrySet()) conf.set(e.getKey(), e.getValue());
    }

    private static Path pickFirstCsvFile(Path in, FileSystem fs) throws IOException {
        if (fs.isFile(in)) return in;
        Queue<Path> q = new ArrayDeque<>();
        q.add(in);
        while (!q.isEmpty()) {
            Path cur = q.poll();
            FileStatus[] sts = fs.listStatus(cur);
            if (sts == null) continue;
            for (FileStatus s : sts) {
                if (s.isFile()) {
                    String name = s.getPath().getName().toLowerCase();
                    if (name.endsWith(".csv") || name.endsWith(".tsv") || name.endsWith(".txt")) return s.getPath();
                }
            }
            for (FileStatus s : sts) if (s.isDirectory()) q.add(s.getPath());
        }
        throw new FileNotFoundException("No CSV-like file found under: " + in);
    }

    private static List<double[]> readFirstKPoints(Path inPath, int k, String sep, FileSystem fs) throws IOException {
        List<double[]> pts = new ArrayList<>(k);
        try (FSDataInputStream in = fs.open(inPath);
             BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
            String line;
            while ((line = br.readLine()) != null && pts.size() < k) {
                line = line.trim();
                if (line.isEmpty()) continue;
                char c0 = line.charAt(0);
                if ((c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z')) continue;
                String[] p = line.split(sep, -1);
                double[] v = new double[p.length];
                for (int i = 0; i < p.length; i++) v[i] = Double.parseDouble(p[i]);
                pts.add(v);
            }
        }
        if (pts.size() < k) throw new IOException("Not enough points to initialize k centroids");
        return pts;
    }

    private static List<double[]> readRandomKAmongNLines(Path inPath, int k, long N, String sep, FileSystem fs) throws IOException {
        List<double[]> reservoir = new ArrayList<>(k);
        Random rnd = new Random(42);
        long seen = 0;
        try (FSDataInputStream in = fs.open(inPath);
             BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
            String line;
            while ((line = br.readLine()) != null && seen < N) {
                line = line.trim();
                if (line.isEmpty()) continue;
                char c0 = line.charAt(0);
                if ((c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z')) continue;
                String[] p = line.split(sep, -1);
                double[] v = new double[p.length];
                boolean ok = true;
                for (int i = 0; i < p.length; i++) {
                    try { v[i] = Double.parseDouble(p[i]); } catch (Exception ex) { ok = false; break; }
                }
                if (!ok) continue;
                seen++;
                if (reservoir.size() < k) reservoir.add(v);
                else {
                    long j = (long)Math.floor(rnd.nextDouble() * seen);
                    if (j < k) reservoir.set((int)j, v);
                }
            }
        }
        if (reservoir.size() < k) throw new IOException("Not enough numeric lines to initialize k centroids (need " + k + ", got " + reservoir.size() + ")");
        return reservoir;
    }

    private static String serialize(List<double[]> cents) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < cents.size(); i++) {
            if (i > 0) sb.append("|");
            double[] v = cents.get(i);
            for (int j = 0; j < v.length; j++) {
                if (j > 0) sb.append(",");
                sb.append(v[j]);
            }
        }
        return sb.toString();
    }

    private static List<double[]> parseCentroidsFromOutput(Path outDir, FileSystem fs) throws IOException {
        Path part = new Path(outDir, "part-r-00000");
        List<double[]> res = new ArrayList<>();
        try (FSDataInputStream in = fs.open(part);
             BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] kv = line.split("\t", 2);
                if (kv.length < 2) continue;
                String[] p = kv[1].split(",");
                double[] v = new double[p.length];
                for (int i = 0; i < p.length; i++) v[i] = Double.parseDouble(p[i]);
                res.add(v);
            }
        }
        return res;
    }

    private static double maxShift(List<double[]> a, List<double[]> b) {
        double m = 0;
        for (int i = 0; i < a.size(); i++) {
            double s = 0;
            for (int j = 0; j < a.get(i).length; j++) {
                double d = a.get(i)[j] - b.get(i)[j];
                s += d*d;
            }
            m = Math.max(m, Math.sqrt(s));
        }
        return m;
    }

    private static Job makeJob(Configuration conf, Path in, Path out, String mode) throws Exception {
        conf.set("output.mode", mode);
        Job job = Job.getInstance(conf, "kmeans-" + mode);
        job.setJarByClass(Main.class);

        // Input format: gộp small files
        job.setInputFormatClass(CombineTextInputFormat.class);
        // Tùy cluster: 128MB/256MB/512MB
        CombineTextInputFormat.setMaxInputSplitSize(job, 128L * 1024 * 1024);

        job.setMapperClass(KMapper.class);
        job.setCombinerClass(KCombiner.class); // ✅ COMBINER
        job.setReducerClass(KReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(PointWritable.class);

        if ("json".equals(mode)) {
            job.setOutputKeyClass(NullWritable.class);
            job.setOutputValueClass(Text.class);
        } else {
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(Text.class);
        }

        FileInputFormat.setInputPaths(job, in);
        FileOutputFormat.setOutputPath(job, out);
        return job;
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] appArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        parseLegacyDArgsIntoConf(appArgs, conf);

        String inArg  = conf.get("in");
        String outArg = conf.get("out");
        if (inArg == null || outArg == null) {
            System.err.println("Usage: hadoop jar <jar> Main -Din <hdfs_input_path(file_or_dir)> -Dout <hdfs_output_dir> "
                + "[-Dk 6] [-Dmaxloop 20] [-Dthresh 0.001] [-DNumReduceTask 1] [-Dlines 10000] [-Dinput.sep ,]");
            System.exit(1);
        }

        Path inPathArg  = new Path(inArg);
        Path outRoot    = new Path(outArg);
        int    k        = Integer.parseInt(conf.get("k", "6"));
        int    maxIt    = Integer.parseInt(conf.get("max.iter", conf.get("maxloop", "20")));
        double thr      = Double.parseDouble(conf.get("thresh", "0.001"));
        int    red      = Integer.parseInt(conf.get("NumReduceTask", "1"));
        String sep      = conf.get("input.sep", ",");

        conf.setBoolean("skip.header.heuristic", true);
        conf.set("feature.names", conf.get("feature.names",
            "longitude,latitude,elevation,max_temperature,min_temperature,precipitation,wind,relative_humidity"));

        FileSystem fs = FileSystem.get(inPathArg.toUri(), conf);

        // Dùng folder làm input cho Job, nhưng lấy 1 file CSV thật làm seed
        Path inputForJob = inPathArg;
        Path seedFile    = pickFirstCsvFile(inPathArg, fs);

        // Khởi tạo centroid
        List<double[]> cents;
        String linesStr = conf.get("lines");
        if (linesStr != null) {
            long N = Long.parseLong(linesStr);
            cents = readRandomKAmongNLines(seedFile, k, N, sep, fs);
        } else {
            cents = readFirstKPoints(seedFile, k, sep, fs);
        }
        System.out.println("[INFO] Seed file for centroid init: " + seedFile);

        Path iterOut = null;
        for (int it = 1; it <= maxIt; it++) {
            if (iterOut != null && fs.exists(iterOut)) fs.delete(iterOut, true);
            iterOut = new Path(outRoot, String.format("out-%02d", it));

            Configuration roundConf = new Configuration(conf);
            roundConf.set("centroids", serialize(cents));
            Job job = makeJob(roundConf, inputForJob, iterOut, "centroids");
            job.setNumReduceTasks(red);

            if (!job.waitForCompletion(true))
                throw new RuntimeException("Job failed at iter " + it);

            List<double[]> newCents = parseCentroidsFromOutput(iterOut, fs);
            if (newCents.size() != cents.size())
                throw new RuntimeException("Centroid size mismatch at iter " + it);

            double shift = maxShift(cents, newCents);
            System.out.println("Iter " + it + " maxShift=" + shift);
            cents = newCents;

            if (shift <= thr || it == maxIt) {
                Path finalOut = new Path(outRoot, "final");
                if (fs.exists(finalOut)) fs.delete(finalOut, true);

                Configuration finalConf = new Configuration(conf);
                finalConf.set("centroids", serialize(cents));
                Job jobFinal = makeJob(finalConf, inputForJob, finalOut, "json");
                jobFinal.setNumReduceTasks(red);

                if (!jobFinal.waitForCompletion(true))
                    throw new RuntimeException("Final JSON job failed");

                System.out.println("DONE. Final JSON at: " + finalOut);
                break;
            }
        }
    }
}
