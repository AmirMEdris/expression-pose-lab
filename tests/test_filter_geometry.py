from filter_geometry import spatial_step,merge_support,support_box


def test_baseline_support_sizes_and_boundary_clipping():
    g=(1,0,0);size=48
    sizes=[]
    for kernel in (5,5):
        g=spatial_step(g,kernel,1,1,'same',size);sizes.append(g[2]-g[1]+1)
    g=spatial_step(g,2,2,1,'valid',size);size=24
    for kernel in (5,5):
        g=spatial_step(g,kernel,1,1,'same',size);sizes.append(g[2]-g[1]+1)
    g=spatial_step(g,2,2,1,'valid',size);size=12
    for kernel in (3,3):
        g=spatial_step(g,kernel,1,1,'same',size);sizes.append(g[2]-g[1]+1)
    assert sizes==[5,9,18,26,36,44]
    box=support_box((g,g),0,0)
    assert box[0]==0 and box[1]==0 and box[2]<=48 and box[3]<=48


def test_residual_union_includes_shortcut_and_asymmetric_same_padding():
    main=spatial_step((1,0,0),3,2,1,'same',48)
    shortcut=spatial_step((1,0,0),1,2,1,'same',48)
    assert main==(2,0,2) and shortcut==(2,0,0)
    assert merge_support([main,shortcut])==(2,0,2)
    assert support_box(((1,-2,2),(1,-2,2)),24,24)==[22,22,27,27]
