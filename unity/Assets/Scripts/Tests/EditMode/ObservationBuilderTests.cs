using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>
    /// Verification level 3 (docs/unity_handoff.md §8): from the recorded physics state, converted into Unity
    /// coordinates first so the conversion code runs, the C# builder reproduces the observation Isaac Lab fed the
    /// policy at every step (history groups included). Tolerance 1e-5; Python reproduces it to ~2e-7.
    /// </summary>
    public class ObservationBuilderTests
    {
        const float Tolerance = 1e-5f;

        [TestCase("flat")]
        [TestCase("rough")]
        [TestCase("v2")]
        public void Rebuilds_isaac_observations(string variant)
        {
            var itf = RobotInterface.Load();
            var r = ReferenceRollout.Load(variant);
            var builder = new ObservationBuilder(itf, variant);
            Assert.AreEqual(r.obs.RowSize, builder.Dim, "obs size");

            var obs = new float[builder.Dim];
            var lastAction = new float[itf.NumJoints];
            var scan = builder.NeedsHeightScan ? new float[r.scan_hits_z.RowSize] : null;
            float maxErr = 0f;
            int worstStep = -1, worstIndex = -1;

            builder.Reset();
            for (var t = 0; t < r.Steps; t++)
            {
                var state = new RobotState
                {
                    BaseRotation = IsaacFrame.QuatFromIsaac(r.root_link_quat_w[t, 0], r.root_link_quat_w[t, 1], r.root_link_quat_w[t, 2], r.root_link_quat_w[t, 3]),
                    BaseComLinearVelocity = IsaacFrame.PolarFromIsaac(Vec3(r.root_com_lin_vel_w, t)),
                    BaseAngularVelocity = IsaacFrame.AxialFromIsaac(Vec3(r.root_com_ang_vel_w, t)),
                    JointPos = r.joint_pos.Row(t),
                    JointVel = r.joint_vel.Row(t),
                };
                if (scan != null)
                    for (var k = 0; k < scan.Length; k++)
                        scan[k] = r.scan_sensor_z.data[t] - r.scan_hits_z[t, k] - itf.HeightScan.Offset;
                if (t > 0)
                    for (var j = 0; j < lastAction.Length; j++)
                        lastAction[j] = r.actions[t - 1, j];

                builder.Build(state, Vec3(r.command, t), lastAction, scan, obs);

                for (var i = 0; i < obs.Length; i++)
                {
                    var err = Mathf.Abs(obs[i] - r.obs[t, i]);
                    if (err > maxErr) (maxErr, worstStep, worstIndex) = (err, t, i);
                }
            }

            Debug.Log($"[BMB] Obs builder {variant}: max |Unity - Isaac| = {maxErr:E2} over {r.Steps} steps x {builder.Dim} (worst: step {worstStep}, index {worstIndex})");
            Assert.That(maxErr, Is.LessThanOrEqualTo(Tolerance));
        }

        [Test]
        public void Reset_fills_history_with_the_current_frame()
        {
            var itf = RobotInterface.Load();
            var builder = new ObservationBuilder(itf, "v2");
            var history = itf.Policy("v2").Groups.Find(g => g.HistoryLength > 0);
            Assert.IsNotNull(history, "v2 has a history group");

            var obs = new float[builder.Dim];
            var state = new RobotState
            {
                BaseRotation = Quaternion.identity,
                BaseComLinearVelocity = Vector3.forward,  // Isaac base_lin_vel x = 1
                BaseAngularVelocity = Vector3.zero,
                JointPos = itf.DefaultJointPos,
                JointVel = new float[itf.NumJoints],
            };
            var scan = new float[itf.Observation.Term("height_scan").Dim];
            builder.Reset();
            builder.Build(state, Vector3.zero, new float[itf.NumJoints], scan, obs);

            // The history block starts after the "policy" group; its first term is base_lin_vel (3 per frame).
            var start = itf.ObsDim("rough");
            Assert.AreEqual("base_lin_vel", history.Terms[0]);
            for (var slot = 0; slot < history.HistoryLength; slot++)
                Assert.AreEqual(1f, obs[start + 3 * slot], 1e-6f, $"history slot {slot} not filled with the current frame");
        }

        static Vector3 Vec3(NdArray a, int t) => new Vector3(a[t, 0], a[t, 1], a[t, 2]);
    }
}
