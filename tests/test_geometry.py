from bhairav.geometry import bbox_iou, point_in_polygon

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_point_inside():
    assert point_in_polygon(5.0, 5.0, SQUARE)


def test_point_outside():
    assert not point_in_polygon(12.0, 5.0, SQUARE)
    assert not point_in_polygon(5.0, -2.0, SQUARE)


def test_point_in_concave_polygon():
    concave = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 5.0), (0.0, 10.0)]
    assert point_in_polygon(2.0, 6.0, concave)    # inside polygon body
    assert not point_in_polygon(5.0, 8.0, concave)   # inside the cut-out notch


def test_bbox_iou_identical():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_bbox_iou_disjoint():
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_bbox_iou_half_overlap():
    # boxes [0,10] and [5,15] overlap by half of each area
    assert abs(bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
