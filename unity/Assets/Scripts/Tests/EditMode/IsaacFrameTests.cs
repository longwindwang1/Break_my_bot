using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>
    /// Unity ↔ Isaac conversions (docs/unity_handoff.md §6). The reference is plain quaternion math in the Isaac
    /// frame, written out independently of IsaacFrame (same as quat_apply_inverse in export/test_parity.py).
    /// </summary>
    public class IsaacFrameTests
    {
        const float Tol = 1e-5f;

        static void AssertClose(Vector3 expected, Vector3 actual, string what)
        {
            Assert.That((expected - actual).magnitude, Is.LessThan(Tol), $"{what}: expected {expected:F6}, got {actual:F6}");
        }

        /// <summary>R(q)^T v for an Isaac quaternion (w, x, y, z), right-handed, in double precision.</summary>
        static Vector3 IsaacQuatApplyInverse(Vector4 q, Vector3 v)
        {
            double w = q.x, x = q.y, y = q.z, z = q.w;  // Vector4 holds (w, x, y, z)
            double vx = v.x, vy = v.y, vz = v.z;
            // t = 2 * cross(xyz, v); result = v - w * t + cross(xyz, t)
            double tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
            return new Vector3(
                (float)(vx - w * tx + (y * tz - z * ty)),
                (float)(vy - w * ty + (z * tx - x * tz)),
                (float)(vz - w * tz + (x * ty - y * tx)));
        }

        static Vector4 RandomIsaacQuat(System.Random rng)
        {
            var q = new Vector4((float)(rng.NextDouble() * 2 - 1), (float)(rng.NextDouble() * 2 - 1), (float)(rng.NextDouble() * 2 - 1), (float)(rng.NextDouble() * 2 - 1));
            return q.normalized;
        }

        static Vector3 RandomVector(System.Random rng) =>
            new Vector3((float)(rng.NextDouble() * 4 - 2), (float)(rng.NextDouble() * 4 - 2), (float)(rng.NextDouble() * 4 - 2));

        [Test]
        public void Axes_map_as_the_urdf_importer_does()
        {
            AssertClose(new Vector3(1, 0, 0), IsaacFrame.PolarToIsaac(Vector3.forward), "Unity forward = Isaac +x");
            AssertClose(new Vector3(0, 1, 0), IsaacFrame.PolarToIsaac(Vector3.left), "Unity left = Isaac +y");
            AssertClose(new Vector3(0, 0, 1), IsaacFrame.PolarToIsaac(Vector3.up), "Unity up = Isaac +z");
        }

        [Test]
        public void Round_trips()
        {
            var rng = new System.Random(0);
            for (var k = 0; k < 100; k++)
            {
                var v = RandomVector(rng);
                AssertClose(v, IsaacFrame.PolarToIsaac(IsaacFrame.PolarFromIsaac(v)), "polar");
                AssertClose(v, IsaacFrame.AxialToIsaac(IsaacFrame.AxialFromIsaac(v)), "axial");
                var q = RandomIsaacQuat(rng);
                var back = IsaacFrame.QuatToIsaac(IsaacFrame.QuatFromIsaac(q.x, q.y, q.z, q.w));
                Assert.That((back - q).magnitude, Is.LessThan(Tol), "quaternion");
            }
        }

        [Test]
        public void Upright_gravity_is_minus_z()
        {
            AssertClose(new Vector3(0, 0, -1), IsaacFrame.PolarInBase(Quaternion.identity, IsaacFrame.GravityDirection), "projected gravity");
        }

        [Test]
        public void Yawing_left_gives_positive_wz()
        {
            // Turning left (counter-clockwise seen from above) moves Unity forward toward Unity left.
            var turnLeft = Quaternion.AngleAxis(-10f, Vector3.up);
            Assert.That((turnLeft * Vector3.forward).x, Is.LessThan(0f), "sanity: forward swings to the left");
            turnLeft.ToAngleAxis(out var angleDeg, out var axis);
            var omegaUnity = axis * (angleDeg * Mathf.Deg2Rad);  // rad over a unit time step
            Assert.That(IsaacFrame.AxialInBase(Quaternion.identity, omegaUnity).z, Is.GreaterThan(0f));
        }

        [Test]
        public void Base_frame_vectors_match_isaac_math()
        {
            // Isaac computes R_base^T v in its own frame; Unity gets the same numbers from its converted state.
            var rng = new System.Random(1);
            for (var k = 0; k < 200; k++)
            {
                var q = RandomIsaacQuat(rng);
                var v = RandomVector(rng);
                var expected = IsaacQuatApplyInverse(q, v);
                var baseRot = IsaacFrame.QuatFromIsaac(q.x, q.y, q.z, q.w);
                AssertClose(expected, IsaacFrame.PolarInBase(baseRot, IsaacFrame.PolarFromIsaac(v)), "polar (lin vel, gravity)");
                AssertClose(expected, IsaacFrame.AxialInBase(baseRot, IsaacFrame.AxialFromIsaac(v)), "axial (ang vel)");
            }
        }
    }
}
