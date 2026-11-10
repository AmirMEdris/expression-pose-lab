"""Conservative display coverage checks, not a pose correctness classifier."""
import math

MAJOR_JOINTS = {
    11:'left shoulder',12:'right shoulder',13:'left elbow',14:'right elbow',
    15:'left wrist',16:'right wrist',23:'left hip',24:'right hip',
    25:'left knee',26:'right knee',27:'left ankle',28:'right ankle',
}


def assess_pose(person, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError('Landmark threshold must be between 0 and 1')
    if len(person) != 33:
        return {'displayable':False,'reason':'invalid landmark count','missing_joints':[]}
    missing = []
    for index, name in MAJOR_JOINTS.items():
        point = person[index]
        values = [point.get(k,float('nan')) for k in ('x','y','z','visibility','presence')]
        if (not all(math.isfinite(v) for v in values)
            or not 0 <= point['x'] < 1 or not 0 <= point['y'] < 1
            or point['visibility'] < threshold or point['presence'] < threshold):
            missing.append(name)
    return {'displayable':not missing,
            'reason':'full-body coverage' if not missing else 'incomplete or low-visibility body',
            'missing_joints':missing}
