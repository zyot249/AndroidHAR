package zyot.shyn;

import android.content.Context;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.List;

public class HARClassifier {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE = "file:///android_asset/frozen_HAR_model.pb";
    private static final String INPUT_NODE = "LSTM_1_input";
    private static final String[] OUTPUT_NODES = {"Dense_2/Softmax"};
    private static final String OUTPUT_NODE = "Dense_2/Softmax";
    private static final long[] INPUT_SIZE = {1, 100, 12};
    private static final int OUTPUT_SIZE = 7;
    public static final int N_SAMPLES = 100;

    private TensorFlowInferenceInterface inferenceInterface;

    public HARClassifier(final Context context) {
        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), MODEL_FILE);
    }

    public float[] predictProbabilities(float[] data) {
        float[] result = new float[OUTPUT_SIZE];
        inferenceInterface.feed(INPUT_NODE, data, INPUT_SIZE);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE, result);

        //Biking   Downstairs	 Jogging	  Sitting	Standing	Upstairs	Walking
        return result;
    }

    public ArrayList<Float> predictHumanActivityProbs(List<Float> ax, List<Float> ay, List<Float> az,
                                                      List<Float> lx, List<Float> ly, List<Float> lz,
                                                      List<Float> gx, List<Float> gy, List<Float> gz) {
        ArrayList<Float> results = null;

        if (ax.size() >= N_SAMPLES && ay.size() >= N_SAMPLES && az.size() >= N_SAMPLES
                && lx.size() >= N_SAMPLES && ly.size() >= N_SAMPLES && lz.size() >= N_SAMPLES
                && gx.size() >= N_SAMPLES && gy.size() >= N_SAMPLES && gz.size() >= N_SAMPLES
        ) {
            List<Float> data = new ArrayList<>();
            List<Float> ma = new ArrayList<>();
            List<Float> ml = new ArrayList<>();
            List<Float> mg = new ArrayList<>();

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

            data.addAll(ax.subList(0, N_SAMPLES));
            data.addAll(ay.subList(0, N_SAMPLES));
            data.addAll(az.subList(0, N_SAMPLES));

            data.addAll(lx.subList(0, N_SAMPLES));
            data.addAll(ly.subList(0, N_SAMPLES));
            data.addAll(lz.subList(0, N_SAMPLES));

            data.addAll(gx.subList(0, N_SAMPLES));
            data.addAll(gy.subList(0, N_SAMPLES));
            data.addAll(gz.subList(0, N_SAMPLES));

            data.addAll(ma.subList(0, N_SAMPLES));
            data.addAll(ml.subList(0, N_SAMPLES));
            data.addAll(mg.subList(0, N_SAMPLES));

            float[] resultsArr = predictProbabilities(toFloatArray(data));
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

    public int predictHumanActivity(List<Float> ax, List<Float> ay, List<Float> az,
                                    List<Float> lx, List<Float> ly, List<Float> lz,
                                    List<Float> gx, List<Float> gy, List<Float> gz) {
        return getIndexOfActHavingMaxProb(predictHumanActivityProbs(ax, ay, az, lx, ly, lz, gx, gy, gz));
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

    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }
}
