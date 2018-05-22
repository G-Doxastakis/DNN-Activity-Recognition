package gr.gdoxastakis.m112activityrecognition;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    final int SEQUENCE_LENGTH = 200;
    final float g = 9.81f;
    private SensorManager mSensorManager;
    private Sensor mSensor;
    private TensorFlowInferenceInterface inferenceInterface;
    float[] activitiesProb;
    float data[] = new float[3*SEQUENCE_LENGTH];
    int data_counter = 0;
    ImageView[] imageViews= {null,null,null,null,null,null};
    Handler mHandler = new Handler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        FloatingActionButton fab = findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                        .setAction("Action", null).show();
            }
        });

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "tensorflow_lite_CNN_Phone_acc.pb");

        /* Full Model
        imageViews[0] = findViewById(R.id.bike);
        imageViews[1] = findViewById(R.id.sit);
        imageViews[2] = findViewById(R.id.stairsdown);
        imageViews[3] = findViewById(R.id.stairsup);
        imageViews[4] = findViewById(R.id.stand);
        imageViews[5] = findViewById(R.id.walk);
        */

        /* Reduced Model */
        imageViews[0] = findViewById(R.id.bike);
        imageViews[1] = findViewById(R.id.stand);
        imageViews[2] = findViewById(R.id.walk);


        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mSensorManager.registerListener(this, mSensor, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
    @Override
    public final void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Do something here if sensor accuracy changes.
    }

    @Override
    public final void onSensorChanged(SensorEvent event) {
        data[data_counter++] = event.values[0]/g;
        data[data_counter++] = event.values[1]/g;
        data[data_counter++] = event.values[2]/g;
        if(data_counter==3*SEQUENCE_LENGTH){
            data_counter = 0;
            mHandler.post(new Runnable() {
                @Override
                public void run() {
                    activitiesProb = predictMotion(data,SEQUENCE_LENGTH);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            updateUI(activitiesProb);
                        }
                    });
                }
            });

        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mSensor, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
    }

    private float[] predictMotion(float[] input, int seqLen){
        float[] output = new float[3];

        inferenceInterface.feed("conv1d_1_input", input, 1, seqLen, 3);
        inferenceInterface.run(new String[]{"dense_2/Softmax"});
        inferenceInterface.fetch("dense_2/Softmax", output);

        return output;
    }

    private void updateUI(float[] acProb){
        for (int i=0;i<3;i++){
            imageViews[i].setAlpha(acProb[i]);
        }
    }
}
