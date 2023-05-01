import pytest
import numpy as np
import numdifftools as nd
from pesexp.geometry.primitives import (
    Distance,
    InverseDistance,
    Angle,
    LinearAngle,
    Dihedral,
    Improper,
    OctahedralA1g,
    OctahedralEg1,
    OctahedralEg2,
    OctahedralT1u1,
    OctahedralT1u2,
    OctahedralT1u3,
)


def test_distances(atol=1e-10):
    e = np.array([1.0, 2.0, 3.0])
    e /= np.linalg.norm(e)

    d = Distance(0, 1)
    d_inv = InverseDistance(0, 1)
    xyzs = np.zeros((2, 3))
    for r in np.linspace(0.1, 100, 21):
        xyzs[0, :] = 0.5 * r * e
        xyzs[1, :] = -0.5 * r * e
        assert np.abs(d.value(xyzs) - r) < atol
        assert np.abs(d_inv.value(xyzs) - 1 / r) < atol


def test_angle(atol=1e-10):
    a = Angle(1, 0, 2)

    xyzs = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.7, 0.0, 0.0]])
    for theta in [0.0, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]:
        xyzs[2, :] = 0.7 * np.cos(theta), 0.0, 0.7 * np.sin(theta)
        assert np.abs(a.value(xyzs) - theta) < atol


def test_angle_on_linear_geometry():
    ref = np.array([1, -1, 1])
    a = Angle(1, 0, 2)

    r1 = 1.2
    r2 = 0.7
    xyzs = np.array([[0.0, 0.0, 0.0], [r1, 0.0, 0.0], [-r2, 0.0, 0.0]])
    dq = a.derivative(xyzs).reshape(-1, 3)
    # Assert that derivative is perpendicular to atomic line
    np.testing.assert_allclose(np.dot(dq, [1, 0, 0]), np.zeros(3))

    xyzs = np.array([[0.0, 0.0, 0.0], r1 * ref, -r2 * ref])
    dq = a.derivative(xyzs).reshape(-1, 3)
    # Assert that derivative is perpendicular to atomic line
    np.testing.assert_allclose(np.dot(dq, ref), np.zeros(3))


def test_linear_angle(atol=1e-10):
    au = LinearAngle(2, 0, 1, 0)
    aw = LinearAngle(2, 0, 1, 1)

    # The y coordinates are not exactly equal to force a consistent
    # reference direction (z-axis).
    x1 = 0.7
    x2 = 1.2
    xyzs = np.array([[0.0, 0.0, 0.0], [x1, 0.0, 0.0], [-x2, 1e-10, 0.0]])
    for y in [0.0, 0.5, 1.0, 2.5, 18.0]:
        xyzs[0, 1] = y
        ref = np.cos(np.arctan2(x1, y)) + np.cos(np.arctan2(x2, y))
        assert np.abs(au.value(xyzs) - ref) < atol
        # Due to the orientation the second component should be zero
        assert np.abs(aw.value(xyzs)) < atol
    # Reference vector should be the z axis:
    np.testing.assert_allclose(au.eref, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(aw.eref, [0.0, 0.0, 1.0])


def test_dihedral(atol=1e-10):
    d = Dihedral(0, 1, 2, 3)

    for w in np.pi + np.linspace(-np.pi, np.pi, 101):
        xyzs = np.array(
            [
                [-0.7, 0.7 * np.cos(w), -0.7 * np.sin(w)],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.7, 0.7, 0.0],
            ]
        )

        if w > np.pi:
            w_ref = w - 2 * np.pi
        else:
            w_ref = w
        assert np.abs(d.value(xyzs) - w_ref) < atol


def test_primitive_representations():
    r = Distance(11, 38)
    assert repr(r) == "Distance(11, 38)"
    r_inv = InverseDistance(11, 38)
    assert repr(r_inv) == "InverseDistance(11, 38)"
    a = Angle(8, 101, 38)
    assert repr(a) == "Angle(8, 101, 38)"
    lin = LinearAngle(4, 12, 8, axis=1)
    assert repr(lin) == "LinearAngle(4, 12, 8, axis=1)"
    d = Dihedral(2, 5, 7, 1)
    assert repr(d) == "Dihedral(2, 5, 7, 1)"
    i = Improper(4, 6, 3, 5)
    assert repr(i) == "Improper(4, 6, 3, 5)"


def test_primitive_derivatives(atol=1e-10):
    xyzs = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.3, 0.0, 0.0],
            [-0.2, 1.4, 0],
            [-1.6, 0.0, 0.0],
            [-0.3, -0.5, 0.9],
        ]
    )

    for i in range(len(xyzs)):
        for j in range(i + 1, len(xyzs)):
            r = Distance(i, j)
            dr_ref = nd.Gradient(lambda x: r.value(x.reshape((-1, 3))))
            np.testing.assert_allclose(r.derivative(xyzs), dr_ref(xyzs))

            r_inv = InverseDistance(i, j)
            dr_inv_ref = nd.Gradient(lambda x: r_inv.value(x.reshape((-1, 3))))
            np.testing.assert_allclose(
                r_inv.derivative(xyzs), dr_inv_ref(xyzs), atol=atol
            )

    a1 = Angle(1, 0, 2)
    da1_ref = nd.Gradient(lambda x: a1.value(x.reshape((-1, 3))))
    np.testing.assert_allclose(a1.derivative(xyzs), da1_ref(xyzs), atol=atol)

    a2 = Angle(1, 0, 4)
    da2_ref = nd.Gradient(lambda x: a2.value(x.reshape((-1, 3))))
    np.testing.assert_allclose(a2.derivative(xyzs), da2_ref(xyzs), atol=atol)

    t1 = LinearAngle(1, 0, 3, 0)
    dt1_ref = nd.Gradient(lambda x: t1.value(x.reshape((-1, 3))))
    np.testing.assert_allclose(t1.derivative(xyzs), dt1_ref(xyzs), atol=atol)

    t2 = LinearAngle(1, 0, 3, 1)
    # Step size needed here to avoid underflow error
    dt2_ref = nd.Gradient(lambda x: t2.value(x.reshape((-1, 3))), step=1e-4)
    np.testing.assert_allclose(t2.derivative(xyzs), dt2_ref(xyzs), atol=atol)

    t3 = LinearAngle(1, 0, 4, 0)
    dt3_ref = nd.Gradient(lambda x: t3.value(x.reshape((-1, 3))))
    np.testing.assert_allclose(t3.derivative(xyzs), dt3_ref(xyzs), atol=atol)

    t4 = LinearAngle(1, 0, 4, 1)
    # Step size needed here to avoid underflow error
    dt4_ref = nd.Gradient(lambda x: t4.value(x.reshape((-1, 3))), step=1e-4)
    np.testing.assert_allclose(t4.derivative(xyzs), dt4_ref(xyzs), atol=atol)

    w = Dihedral(1, 0, 2, 4)
    dw_ref = nd.Gradient(lambda x: w.value(x.reshape((-1, 3))))
    np.testing.assert_allclose(w.derivative(xyzs), dw_ref(xyzs), atol=atol)


@pytest.mark.parametrize(
    "prim",
    [
        OctahedralA1g,
        OctahedralEg1,
        OctahedralEg2,
        OctahedralT1u1,
        OctahedralT1u2,
        OctahedralT1u3,
    ],
)
def test_octahedral_derivatives(prim, atol=1e-10):
    r = 1.2
    xyzs = np.array(
        [
            [0.0, 0.0, 0.0],
            [r, 0.0, 0.0],
            [0.0, r, 0.0],
            [-r, 0.0, 0.0],
            [0.0, -r, 0.0],
            [0.0, 0.0, r],
            [0.0, 0.0, -r],
        ]
    )

    q = prim(0, 1, 2, 3, 4, 5, 6)
    dq_ref = nd.Gradient(lambda x: q.value(x.reshape((-1, 3))), step=1e-4)
    np.testing.assert_allclose(q.derivative(xyzs), dq_ref(xyzs), atol=atol)
