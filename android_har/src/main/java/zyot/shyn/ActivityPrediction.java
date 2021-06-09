package zyot.shyn;

public class ActivityPrediction {
    private float probability;
    private int activityIdx;

    public ActivityPrediction(float probability, int activityIdx) {
        this.probability = probability;
        this.activityIdx = activityIdx;
    }

    public float getProbability() {
        return probability;
    }

    public HumanActivity getActivityIdx() {
        return HumanActivity.getHumanActivity(activityIdx);
    }
}
