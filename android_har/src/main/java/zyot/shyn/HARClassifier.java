package zyot.shyn;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class HARClassifier {
    private static final int[] INPUT_SIZE = {1, 100, 12};
    private static final int OUTPUT_SIZE = 7;
    public static final int N_SAMPLES = 100;

//    private static final String TF_MODEL_FILE = "har_tflite_retrain.tflite";
    private static final String TF_MODEL_FILE = "har_tflite_09716.tflite";

    private Interpreter interpreter;

    public HARClassifier(final Context context) {
        try {
            interpreter = new Interpreter(loadModelFile(context.getAssets(), TF_MODEL_FILE), new Interpreter.Options());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public float[] predict(float[][][] input) {
        float[][] result = new float[1][OUTPUT_SIZE];
        interpreter.run(input, result);
        return result[0];
    }

    public ArrayList<Float> predictHumanActivityProbs(List<Float> ax, List<Float> ay, List<Float> az,
                                                      List<Float> lx, List<Float> ly, List<Float> lz,
                                                      List<Float> gx, List<Float> gy, List<Float> gz) {
        ArrayList<Float> results = null;

        if (ax.size() >= N_SAMPLES && ay.size() >= N_SAMPLES && az.size() >= N_SAMPLES
                && lx.size() >= N_SAMPLES && ly.size() >= N_SAMPLES && lz.size() >= N_SAMPLES
                && gx.size() >= N_SAMPLES && gy.size() >= N_SAMPLES && gz.size() >= N_SAMPLES
        ) {
            List<Float> ma = new ArrayList<>();
            List<Float> ml = new ArrayList<>();
            List<Float> mg = new ArrayList<>();
            samplingData(ax, ay, az, lx, ly, lz, gx, gy, gz);

            double maValue;
            double mgValue;
            double mlValue;

            for (int i = 0; i < N_SAMPLES; i++) {
                maValue = Math.sqrt(Math.pow(ax.get(i), 2) + Math.pow(ay.get(i), 2) + Math.pow(az.get(i), 2));
                mlValue = Math.sqrt(Math.pow(lx.get(i), 2) + Math.pow(ly.get(i), 2) + Math.pow(lz.get(i), 2));
                mgValue = Math.sqrt(Math.pow(gx.get(i), 2) + Math.pow(gy.get(i), 2) + Math.pow(gz.get(i), 2));

                ma.add((float) maValue);
                ml.add((float) mlValue);
                mg.add((float) mgValue);
            }

            float[][][] input = reshapeInput(ax, ay, az, lx, ly, lz, gx, gy, gz, ma, ml, mg);
            float[] resultsArr = predict(input);
            results = new ArrayList<>();
            for (int i = 0; i < OUTPUT_SIZE; i++)
                results.add(resultsArr[i]);

            ax.clear();
            ay.clear();
            az.clear();
            lx.clear();
            ly.clear();
            lz.clear();
            gx.clear();
            gy.clear();
            gz.clear();
            ma.clear();
            ml.clear();
            mg.clear();
        }
        return results;
    }

    public int predictHumanActivityIndex(List<Float> ax, List<Float> ay, List<Float> az,
                                    List<Float> lx, List<Float> ly, List<Float> lz,
                                    List<Float> gx, List<Float> gy, List<Float> gz) {
        return getIndexOfActHavingMaxProb(predictHumanActivityProbs(ax, ay, az, lx, ly, lz, gx, gy, gz));
    }

    public ActivityPrediction predictHumanActivity(List<Float> ax, List<Float> ay, List<Float> az,
                                                   List<Float> lx, List<Float> ly, List<Float> lz,
                                                   List<Float> gx, List<Float> gy, List<Float> gz) {
        ArrayList<Float> results = predictHumanActivityProbs(ax, ay, az, lx, ly, lz, gx, gy, gz);
        int index = -1;
        float max = 0;
        if (results != null) {
            for (int i = 0; i < results.size(); i++) {
                if (results.get(i) > max) {
                    index = i;
                    max = results.get(i);
                }
            }
        }
        return new ActivityPrediction(max, index);
    }

    public float[][][] reshapeInput(List<Float> ax, List<Float> ay, List<Float> az,
                                    List<Float> lx, List<Float> ly, List<Float> lz,
                                    List<Float> gx, List<Float> gy, List<Float> gz,
                                    List<Float> ma, List<Float> ml, List<Float> mg) {
        float[][][] input = new float[INPUT_SIZE[0]][INPUT_SIZE[1]][INPUT_SIZE[2]];
        for (int i = 0; i < N_SAMPLES; i++) {
            input[0][i][0] = ax.get(i);
            input[0][i][1] = ay.get(i);
            input[0][i][2] = az.get(i);
            input[0][i][3] = lx.get(i);
            input[0][i][4] = ly.get(i);
            input[0][i][5] = lz.get(i);
            input[0][i][6] = gx.get(i);
            input[0][i][7] = gy.get(i);
            input[0][i][8] = gz.get(i);
            input[0][i][9] = ma.get(i);
            input[0][i][10] = ml.get(i);
            input[0][i][11] = mg.get(i);
        }
        return input;
    }

    public static int getIndexOfActHavingMaxProb(ArrayList<Float> arr) {
        int index = -1;
        if (arr != null) {
            float max = -1;
            for (int i = 0; i < arr.size(); i++) {
                if (arr.get(i) > max) {
                    index = i;
                    max = arr.get(i);
                }
            }
        }
        return index;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void samplingData(List<Float> ax, List<Float> ay, List<Float> az,
                             List<Float> lx, List<Float> ly, List<Float> lz,
                             List<Float> gx, List<Float> gy, List<Float> gz) {
        if (ax.size() >= N_SAMPLES && ay.size() >= N_SAMPLES && az.size() >= N_SAMPLES
                && lx.size() >= N_SAMPLES && ly.size() >= N_SAMPLES && lz.size() >= N_SAMPLES
                && gx.size() >= N_SAMPLES && gy.size() >= N_SAMPLES && gz.size() >= N_SAMPLES
        ) {
            ax.subList(N_SAMPLES, ax.size()).clear();
            ay.subList(N_SAMPLES, ay.size()).clear();
            az.subList(N_SAMPLES, az.size()).clear();
            lx.subList(N_SAMPLES, lx.size()).clear();
            ly.subList(N_SAMPLES, ly.size()).clear();
            lz.subList(N_SAMPLES, lz.size()).clear();
            gx.subList(N_SAMPLES, gx.size()).clear();
            gy.subList(N_SAMPLES, gy.size()).clear();
            gz.subList(N_SAMPLES, gz.size()).clear();
        }
    }
}
