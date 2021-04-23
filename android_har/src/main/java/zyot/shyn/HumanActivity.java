package zyot.shyn;

public enum HumanActivity {
    UNKNOWN(-1),
    BIKING(0),
    DOWNSTAIRS(1),
    JOGGING(2),
    SITTING(3),
    STANDING(4),
    UPSTAIRS(5),
    WALKING(6);

    int index;

    HumanActivity(int index) {
        this.index = index;
    }

    public int getIndex() {
        return index;
    }

    public static HumanActivity getHumanActivity(int index) {
        switch (index) {
            case 0:
                return BIKING;
            case 1:
                return DOWNSTAIRS;
            case 2:
                return JOGGING;
            case 3:
                return SITTING;
            case 4:
                return STANDING;
            case 5:
                return UPSTAIRS;
            case 6:
                return WALKING;
            default:
                return UNKNOWN;
        }
    }
}
