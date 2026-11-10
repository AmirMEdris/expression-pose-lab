import pytest
from pose_quality import assess_pose


def complete_pose():
    return [dict(x=.5,y=.5,z=0.,visibility=.99,presence=.99) for _ in range(33)]


def test_complete_body_can_be_displayed():
    assert assess_pose(complete_pose())['displayable']


@pytest.mark.parametrize('field,value', [('y',1.2),('x',-0.1),('visibility',.1),('presence',.1),('x',float('nan')),('z',float('inf'))])
def test_incomplete_or_invalid_major_joint_withholds_entire_pose(field,value):
    person=complete_pose()
    person[27][field]=value
    result=assess_pose(person)
    assert not result['displayable']
    assert 'left ankle' in result['missing_joints']


def test_coverage_does_not_require_small_face_or_finger_points():
    person=complete_pose()
    person[0]['visibility']=0
    assert assess_pose(person)['displayable']


def test_invalid_landmark_count_is_not_a_pose():
    assert not assess_pose([])['displayable']


def test_threshold_is_configurable_and_validated():
    person=complete_pose()
    person[15]['visibility']=.6
    assert assess_pose(person,.5)['displayable']
    assert not assess_pose(person,.7)['displayable']
    with pytest.raises(ValueError):
        assess_pose(person,1.1)
