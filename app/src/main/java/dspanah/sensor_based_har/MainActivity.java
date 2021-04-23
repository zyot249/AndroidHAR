package dspanah.sensor_based_har;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.widget.TableRow;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;
import androidx.core.content.res.ResourcesCompat;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

import zyot.shyn.HARClassifier;
import zyot.shyn.HumanActivity;


public class MainActivity extends AppCompatActivity implements SensorEventListener, TextToSpeech.OnInitListener {


    private static int prevIdx = -1;

    private static List<Float> ax;
    private static List<Float> ay;
    private static List<Float> az;

    private static List<Float> lx;
    private static List<Float> ly;
    private static List<Float> lz;

    private static List<Float> gx;
    private static List<Float> gy;
    private static List<Float> gz;

//    private static List<Float> ma;
//    private static List<Float> ml;
//    private static List<Float> mg;

    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mGyroscope;
    private Sensor mLinearAcceleration;


    private TextView bikingTextView;
    private TextView downstairsTextView;
    private TextView joggingTextView;
    private TextView sittingTextView;
    private TextView standingTextView;
    private TextView upstairsTextView;
    private TextView walkingTextView;

    private TableRow bikingTableRow;
    private TableRow downstairsTableRow;
    private TableRow joggingTableRow;
    private TableRow sittingTableRow;
    private TableRow standingTableRow;
    private TableRow upstairsTableRow;
    private TableRow walkingTableRow;

    private TextToSpeech textToSpeech;
    private ArrayList<Float> results;
    private HARClassifier classifier;

    private String[] labels = {"Biking", "Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"};

    private long firstTimeStartState = 0;
    private long notiTimestamp = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ax = new ArrayList<>(); ay = new ArrayList<>(); az = new ArrayList<>();
        lx = new ArrayList<>(); ly = new ArrayList<>(); lz = new ArrayList<>();
        gx = new ArrayList<>(); gy = new ArrayList<>(); gz = new ArrayList<>();
//        ma = new ArrayList<>(); ml = new ArrayList<>(); mg = new ArrayList<>();

        bikingTextView = (TextView) findViewById(R.id.biking_prob);
        downstairsTextView = (TextView) findViewById(R.id.downstairs_prob);
        joggingTextView = (TextView) findViewById(R.id.jogging_prob);
        sittingTextView = (TextView) findViewById(R.id.sitting_prob);
        standingTextView = (TextView) findViewById(R.id.standing_prob);
        upstairsTextView = (TextView) findViewById(R.id.upstairs_prob);
        walkingTextView = (TextView) findViewById(R.id.walking_prob);

        bikingTableRow = (TableRow) findViewById(R.id.biking_row);
        downstairsTableRow = (TableRow) findViewById(R.id.downstairs_row);
        joggingTableRow = (TableRow) findViewById(R.id.jogging_row);
        sittingTableRow = (TableRow) findViewById(R.id.sitting_row);
        standingTableRow = (TableRow) findViewById(R.id.standing_row);
        upstairsTableRow = (TableRow) findViewById(R.id.upstairs_row);
        walkingTableRow = (TableRow) findViewById(R.id.walking_row);

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mSensorManager.registerListener(this, mAccelerometer , SensorManager.SENSOR_DELAY_FASTEST);

        mLinearAcceleration = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mSensorManager.registerListener(this, mLinearAcceleration , SensorManager.SENSOR_DELAY_FASTEST);

        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mSensorManager.registerListener(this, mGyroscope , SensorManager.SENSOR_DELAY_FASTEST);

        classifier = new HARClassifier(getApplicationContext());

        textToSpeech = new TextToSpeech(this, this);
        textToSpeech.setLanguage(Locale.US);

        initActionNoti();
    }

    private void initActionNoti() {
        createNotificationChannel();

        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                long now = System.currentTimeMillis();
                if (now - firstTimeStartState >= 3000 && (prevIdx == 3 || prevIdx == 4) && now - notiTimestamp >= 10000) {
                    showNoti("Warning", "You need do exercises");
                    notiTimestamp = now;
                }
            }
        }, 1000, 3000);
    }

    @Override
    public void onInit(int status) {
        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (results == null || results.size() == 0) {
                    return;
                }
                float max = -1;
                int idx = -1;
                for (int i = 0; i < results.size(); i++) {
                    if (results.get(i) > max) {
                        idx = i;
                        max = results.get(i);
                    }
                }

                if(max > 0.50 && idx != prevIdx) {
                    textToSpeech.speak(labels[idx], TextToSpeech.QUEUE_ADD, null,
                            Integer.toString(new Random().nextInt()));
                    prevIdx = idx;
                    firstTimeStartState = System.currentTimeMillis();
                }
            }
        }, 1000, 3000);
    }

    protected void onResume() {
        super.onResume();
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_FASTEST);
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION), SensorManager.SENSOR_DELAY_FASTEST);
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    public void onDestroy() {
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        super.onDestroy();
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        activityPrediction();

        Sensor sensor = event.sensor;
        if (sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            ax.add(event.values[0]);
            ay.add(event.values[1]);
            az.add(event.values[2]);

        } else if (sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            lx.add(event.values[0]);
            ly.add(event.values[1]);
            lz.add(event.values[2]);

        } else if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            gx.add(event.values[0]);
            gy.add(event.values[1]);
            gz.add(event.values[2]);

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    private void activityPrediction() {
        results = classifier.predictHumanActivityProbs(ax, ay, az, lx, ly, lz, gx, gy, gz);
        if (results != null) {
//            float max = -1;
//            int idx = -1;
//            for (int i = 0; i < results.size(); i++) {
//                if (results.get(i) > max) {
//                    idx = i;
//                    max = results.get(i);
//                }
//            }

            setProbabilities(results);
            setRowsColor(HARClassifier.getIndexOfActHavingMaxProb(results));
        }
    }

    private void setProbabilities(ArrayList<Float> results) {
        bikingTextView.setText(Float.toString(round(results.get(HumanActivity.BIKING.getIndex()), 2)));
        downstairsTextView.setText(Float.toString(round(results.get(HumanActivity.DOWNSTAIRS.getIndex()), 2)));
        joggingTextView.setText(Float.toString(round(results.get(HumanActivity.JOGGING.getIndex()), 2)));
        sittingTextView.setText(Float.toString(round(results.get(HumanActivity.SITTING.getIndex()), 2)));
        standingTextView.setText(Float.toString(round(results.get(HumanActivity.STANDING.getIndex()), 2)));
        upstairsTextView.setText(Float.toString(round(results.get(HumanActivity.UPSTAIRS.getIndex()), 2)));
        walkingTextView.setText(Float.toString(round(results.get(HumanActivity.WALKING.getIndex()), 2)));
    }

    private void setRowsColor(int idx) {
        bikingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        downstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        joggingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        sittingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        standingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        upstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));
        walkingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorTransparent, null));

        if(idx == 0)
            //bikingTextView.setBackgroundColor(Color.parseColor("#33B5E5"));
            bikingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 1)
            downstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 2)
            joggingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 3)
            sittingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 4)
            standingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 5)
            upstairsTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
        else if (idx == 6)
            walkingTableRow.setBackgroundColor(ResourcesCompat.getColor(getResources(), R.color.colorBlue, null));
    }


    private static float round(float d, int decimalPlace) {
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }

    private SensorManager getSensorManager() {
        return (SensorManager) getSystemService(SENSOR_SERVICE);
    }


    final String CHANNEL_ID = "123456";

    private void createNotificationChannel() {
        // Create the NotificationChannel, but only on API 26+ because
        // the NotificationChannel class is new and not in the support library
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            CharSequence name = "channel_name";
            String description = "channel_description";
            int importance = NotificationManager.IMPORTANCE_DEFAULT;
            NotificationChannel channel = new NotificationChannel(CHANNEL_ID, name, importance);
            channel.setDescription(description);
            // Register the channel with the system; you can't change the importance
            // or other notification behaviors after this
            NotificationManager notificationManager = getSystemService(NotificationManager.class);
            notificationManager.createNotificationChannel(channel);
        }
    }


    private void showNoti(String textTitle, String textContent) {
        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_launcher_foreground)
                .setContentTitle(textTitle)
                .setContentText(textContent)
                .setPriority(NotificationCompat.PRIORITY_DEFAULT);

        NotificationManagerCompat notificationManager = NotificationManagerCompat.from(this);

        int notificationId = (int)System.currentTimeMillis();
        // notificationId is a unique int for each notification that you must define
        notificationManager.notify(notificationId, builder.build());
    }
}
