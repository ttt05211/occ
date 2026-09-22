import numpy as np

from causal_se2_occ.data.features import world_points_to_t0
from causal_se2_occ.geometry.grid import OccupancyGrid, pose_matrix
from causal_se2_occ.geometry.render import (
    RasterizedRigidComponent,
    compose_hard_a1,
    rasterize_rigid_component,
)
from causal_se2_occ.geometry.se2 import (
    apply_box_centered_rigid_xy,
    apply_source_centered_rigid_xy,
    source_center_se2_target,
)
from causal_se2_occ.inference import t0_xy_to_world_preserve_source_z
from causal_se2_occ.protocol import DYNAMIC_CLASS_IDS, YAW_ENABLED_CLASS_IDS


def test_source_center_equivalent():
    source_center = np.array([3.2, -1.7])
    a0 = np.array([2.0, -2.0])
    ah = np.array([3.5, 0.3])
    yaw = 0.4
    target = source_center_se2_target(source_center, a0, ah, yaw)
    points = np.array([[2.1, -2.2], [3.2, -1.7], [4.0, -1.0]])

    expected = apply_box_centered_rigid_xy(points, a0, ah, yaw)
    actual = apply_source_centered_rigid_xy(
        points,
        source_center,
        target.source_displacement_xy_m,
        target.yaw_rad,
    )
    np.testing.assert_allclose(expected, actual, atol=2e-7, rtol=0)


def test_pedestrian_rule():
    assert 7 in DYNAMIC_CLASS_IDS
    assert 7 not in YAW_ENABLED_CLASS_IDS


def test_renderer_and_a1():
    grid = OccupancyGrid(
        x_min=0,
        y_min=0,
        z_min=0,
        voxel_size=(1.0, 1.0, 1.0),
        shape_xyz=(8, 8, 4),
    )
    identity = np.eye(4)
    source = np.array([[2, 2, 0], [3, 2, 0], [2, 2, 1]])
    rendered = rasterize_rigid_component(
        source,
        4,
        identity,
        identity,
        source_center_world=np.array([3.0, 2.5, 0.5]),
        target_center_world=np.array([4.0, 3.5, 0.5]),
        grid=grid,
    )
    assert rendered.source_voxel_count == 3

    grid2 = OccupancyGrid(
        x_min=0,
        y_min=0,
        z_min=0,
        voxel_size=(1.0, 1.0, 1.0),
        shape_xyz=(5, 5, 1),
    )
    anchor = np.full(grid2.shape_xyz, 17, dtype=np.uint8)
    anchor[1, 1, 0] = 4
    anchor[2, 2, 0] = 11
    baseline = [
        RasterizedRigidComponent(
            4,
            np.array([[1, 1, 0], [2, 2, 0]]),
            2,
        )
    ]
    replacements = [
        RasterizedRigidComponent(4, np.array([[3, 3, 0]]), 1),
        RasterizedRigidComponent(10, np.array([[3, 3, 0]]), 1),
    ]
    output = compose_hard_a1(
        anchor,
        baseline,
        replacements,
        dynamic_class_ids=DYNAMIC_CLASS_IDS,
        grid=grid2,
    )
    assert output[1, 1, 0] == 17
    assert output[2, 2, 0] == 11
    assert output[3, 3, 0] == 10


def test_t0_xy_to_world_matches_frozen_height_contract_with_pitch_roll():
    roll = np.deg2rad(7.0)
    pitch = np.deg2rad(10.0)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)

    # q = q_y(pitch) * q_x(roll), wxyz.
    quaternion = np.array([cp * cr, cp * sr, sp * cr, -sp * sr])
    pose = pose_matrix([4.0, -3.0, 1.5], quaternion)

    source_world = np.array([12.0, 5.0, 4.0], dtype=np.float64)
    xy_t0 = np.array([2.25, -1.75], dtype=np.float64)

    source_t0 = world_points_to_t0(source_world[None], pose)[0]
    legacy_point = np.asarray(
        [xy_t0[0], xy_t0[1], source_t0[2], 1.0],
        dtype=np.float64,
    )
    expected = (pose @ legacy_point)[:3]
    actual = t0_xy_to_world_preserve_source_z(
        xy_t0,
        source_world,
        pose,
    )
    np.testing.assert_array_equal(actual, expected)
