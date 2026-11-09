using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace BMB.Tests
{
    /// <summary>
    /// World-model verification, docs/unity_handoff.md §11.3:
    /// level 1 the ONNX outputs, level 2 the C# wm_in builder (from the v2 reference state through the observation
    /// builder), level 3 surprisal + CUSUM, against experiments/w3_reference/go2_wm*.json. Plus the yaml layout, the
    /// displacement / attitude conventions the ghost relies on, and the Unity-calibration override.
    /// </summary>
    public class WorldModelTests
    {
        RobotInterface itf;
        WorldModelRunner runner;

        [OneTimeSetUp]
        public void Setup()
        {
            itf = RobotInterface.Load();
            Editor.SyncModels.Sync();
            runner = WorldModelRunner.Load(itf);
        }

        [OneTimeTearDown]
        public void TearDown() => runner?.Dispose();

        [Test]
        public void Layout_follows_the_yaml()
        {
            var wm = itf.WorldModel;
            Assert.IsNotNull(wm, "world_model section");
            CollectionAssert.AreEqual(new[] { "v1", "v2" }, wm.Policies);
            Assert.AreEqual(1f, wm.Surprisal.Weights.Sum(), 1e-4f, "weights sum");
            Assert.That(wm.Surprisal.PerPolicy.Keys, Is.EquivalentTo(new[] { "v1", "v2" }));

            foreach (var (variant, policy, index) in new[] { ("rough", "v1", 0), ("v2", "v2", 1) })
            {
                var l = new WorldModelLayout(itf, variant);
                Assert.AreEqual(policy, l.Policy);
                Assert.AreEqual(index, l.PolicyIndex);
                Assert.AreEqual(471, l.InputDim);
                Assert.AreEqual(45, l.FrameDim);
                Assert.AreEqual(33, l.StateDim);
                Assert.AreEqual(6, l.HistoryLength);
                Assert.AreEqual((270, 282, 469), (l.ActionOffset, l.ScanOffset, l.OneHotOffset));
                // newest frame of each state term block: base_lin_vel 15..17, ..., joint_vel 186..197
                Assert.AreEqual(new[] { 15, 16, 17 }, l.CurrentStateInInput.Take(3).ToArray());
                Assert.AreEqual(Enumerable.Range(186, 12).ToArray(), l.CurrentStateInInput.Skip(21).ToArray());
                // obs: lin 0..2, ang 3..5, gravity 6..8, (command 9..11), joint_pos 12..23, joint_vel 24..35, actions 36..47, scan 48..
                Assert.AreEqual(Enumerable.Range(0, 9).Concat(Enumerable.Range(12, 36)).ToArray(), l.FrameIndexInObs);
                Assert.AreEqual(48, l.ScanOffsetInObs);
            }
            Assert.IsNull(WorldModelSpec.PolicyName("flat"));
            Assert.Throws<ArgumentException>(() => new WorldModelLayout(itf, "flat"));
        }

        [TestCase("wm")]
        [TestCase("wm_mudrop")]
        public void Onnx_reproduces_reference_outputs(string name)
        {
            // Level 1: state_mean / disp_mean 1e-4 absolute, variances and disagreement 1e-3 relative.
            var r = WmReference.Load(name);
            var p = new WorldModelPrediction(itf.WorldModel);
            var err = new double[6];
            for (var t = 0; t < r.steps; t++)
            {
                runner.Run(r.wm_in.Row(t), p);
                err[0] = Math.Max(err[0], MaxAbs(p.StateMean, r.state_mean, t));
                err[1] = Math.Max(err[1], MaxRel(p.StateVar, r.state_var, t));
                err[2] = Math.Max(err[2], MaxAbs(p.DispMean, r.disp_mean, t));
                err[3] = Math.Max(err[3], MaxRel(p.DispVar, r.disp_var, t));
                err[4] = Math.Max(err[4], MaxAbs(p.FallProb, r.fall_prob, t));
                err[5] = Math.Max(err[5], MaxRel(new[] { p.Disagreement }, r.disagreement, t));
            }
            Debug.Log($"[BMB] World model ONNX {name}: max abs state_mean {err[0]:E2}, disp_mean {err[2]:E2}, fall_prob {err[4]:E2}; " +
                      $"max rel state_var {err[1]:E2}, disp_var {err[3]:E2}, disagreement {err[5]:E2}");
            Assert.That(err[0], Is.LessThanOrEqualTo(1e-4), "state_mean");
            Assert.That(err[2], Is.LessThanOrEqualTo(1e-4), "disp_mean");
            Assert.That(err[4], Is.LessThanOrEqualTo(1e-4), "fall_prob");
            Assert.That(err[1], Is.LessThanOrEqualTo(1e-3), "state_var");
            Assert.That(err[3], Is.LessThanOrEqualTo(1e-3), "disp_var");
            Assert.That(err[5], Is.LessThanOrEqualTo(1e-3), "disagreement");
        }

        [TestCase("v2", "wm")]
        [TestCase("v2_mudrop", "wm_mudrop")]
        public void Monitor_builds_reference_inputs(string rollout, string trace)
        {
            // Level 2: observation builder (from the raw reference state, in Unity axes) -> monitor -> wm_in, 1e-5.
            // The monitor then runs its own predictions and detector; with the yaml (Isaac) thresholds its alarms must
            // be the reference's.
            var r = ReferenceRollout.Load(rollout);
            var w = WmReference.Load(trace);
            var builder = new ObservationBuilder(itf, "v2");
            var monitor = new WorldModelMonitor(itf, "v2", runner, SurprisalParams.Isaac(itf, "v2"));
            var obs = new float[builder.Dim];
            var lastAction = new float[itf.NumJoints];
            var scan = new float[r.scan_hits_z.RowSize];
            float maxErr = 0f;
            int worstStep = -1, worstIndex = -1;
            var alarms = new System.Collections.Generic.List<int>();
            double maxG = 0;
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
                for (var k = 0; k < scan.Length; k++)
                    scan[k] = r.scan_sensor_z.data[t] - r.scan_hits_z[t, k] - itf.HeightScan.Offset;
                if (t > 0)
                    for (var j = 0; j < lastAction.Length; j++)
                        lastAction[j] = r.actions[t - 1, j];
                builder.Build(state, Vec3(r.command, t), lastAction, scan, obs);
                monitor.Step(obs, r.actions.Row(t), t);

                for (var i = 0; i < monitor.Input.Length; i++)
                {
                    var e = Mathf.Abs(monitor.Input[i] - w.wm_in[t, i]);
                    if (e > maxErr) (maxErr, worstStep, worstIndex) = (e, t, i);
                }
                if (monitor.Detector.Alarm)
                    alarms.Add(t);
                maxG = Math.Max(maxG, monitor.Detector.GPeak);
            }
            var expected = Enumerable.Range(0, w.steps).Where(t => w.alarm.data[t] > 0.5f).ToArray();
            Debug.Log($"[BMB] World model input {trace}: max |Unity - Isaac| wm_in {maxErr:E2} (step {worstStep}, index {worstIndex}); " +
                      $"closed chain alarms [{string.Join(", ", alarms)}], reference [{string.Join(", ", expected)}], max g {maxG:F2}");
            Assert.That(maxErr, Is.LessThanOrEqualTo(1e-5f));
            CollectionAssert.AreEqual(expected, alarms, "alarms from the C# chain");
        }

        [TestCase("wm")]
        [TestCase("wm_mudrop")]
        public void Surprisal_reproduces_reference(string name)
        {
            // Level 3: from the stored states and outputs, z, S, x, g to 1e-5 (relative above 1), spike and alarm exactly.
            var r = WmReference.Load(name);
            var layout = new WorldModelLayout(itf, "v2");
            var det = new SurprisalDetector(SurprisalParams.Isaac(itf, r.policy));
            var s = new float[layout.StateDim];
            double maxZ = 0, maxS = 0, maxX = 0, maxG = 0;
            var alarms = new System.Collections.Generic.List<int>();
            for (var t = 0; t < r.steps; t++)
            {
                for (var i = 0; i < s.Length; i++)
                    s[i] = r.wm_in[t, layout.CurrentStateInInput[i]];
                var step = (int)r.episode_step.data[t];
                det.Step(s, t > 0 ? r.state_mean.Row(t - 1) : null, t > 0 ? r.state_var.Row(t - 1) : null, step);
                for (var d = 0; d < s.Length; d++)
                    maxZ = Math.Max(maxZ, Rel(det.Z[d], r.z[t, d]));
                maxS = Math.Max(maxS, Rel(det.S, r.S.data[t]));
                maxX = Math.Max(maxX, Rel(det.X, r.x.data[t]));
                maxG = Math.Max(maxG, Rel(det.G, r.g.data[t]));
                Assert.AreEqual(r.scored.data[t] > 0.5f, det.Scored, $"scored at step {t}");
                Assert.AreEqual(r.spike.data[t] > 0.5f, det.Spike, $"spike at step {t}");
                Assert.AreEqual(r.alarm.data[t] > 0.5f, det.Alarm, $"alarm at step {t}");
                if (det.Alarm)
                    alarms.Add(t);
            }
            Debug.Log($"[BMB] Surprisal {name}: max error z {maxZ:E2}, S {maxS:E2}, x {maxX:E2}, g {maxG:E2}; alarms [{string.Join(", ", alarms)}]");
            Assert.That(Math.Max(Math.Max(maxZ, maxS), Math.Max(maxX, maxG)), Is.LessThanOrEqualTo(1e-5));
            if (r.HasFrictionDrop)
            {
                // experiments/w5_surprisal/README.md §6: first alarm 0.24 s after the drop, 12 alarms in total
                Assert.AreEqual(262, alarms.FirstOrDefault(), "first alarm after the friction drop");
                Assert.AreEqual(12, alarms.Count);
            }
            else
                Assert.IsEmpty(alarms, "the clean reference rollout must not raise an alarm");
        }

        [Test]
        public void Displacement_and_attitude_conventions_match_reference()
        {
            // The ghost integrates disp_mean in the yaw frame and builds its attitude from projected_gravity. One-step
            // predictions are accurate (W5: base xy error ~0 at 1 step), so integrating the predicted displacement from
            // the recorded pose at t must land on the recorded pose at t+1, far closer than "no motion" does.
            var r = ReferenceRollout.Load("v2");
            var w = WmReference.Load("wm");
            double posErr = 0, moved = 0, yawErr = 0, attErr = 0;
            var n = r.Steps - 1;
            for (var t = 0; t < n; t++)
            {
                var (p0, q0) = Pose(r, t);
                var (p1, q1) = Pose(r, t + 1);
                var pos = p0;
                var yaw = WmPose.Yaw(q0);
                WmPose.Integrate(ref pos, ref yaw, w.disp_mean.Row(t));
                posErr += (pos - p1).magnitude;
                moved += (p1 - p0).magnitude;
                yawErr += Mathf.Abs(Mathf.DeltaAngle(yaw * Mathf.Rad2Deg, WmPose.Yaw(q1) * Mathf.Rad2Deg));
                var g = new Vector3(r.obs[t, 6], r.obs[t, 7], r.obs[t, 8]);
                attErr += Quaternion.Angle(WmPose.Rotation(WmPose.Yaw(q0), g), q0);
            }
            posErr /= n; moved /= n; yawErr /= n; attErr /= n;
            Debug.Log($"[BMB] Ghost conventions: mean position error {posErr * 1000:F3} mm per step (base moves {moved * 1000:F2} mm), " +
                      $"yaw error {yawErr:F4} deg, attitude from gravity + yaw {attErr:F4} deg");
            Assert.That(posErr, Is.LessThan(0.2 * moved), "integrated displacement");
            Assert.That(yawErr, Is.LessThan(0.1), "integrated yaw (deg)");
            Assert.That(attErr, Is.LessThan(0.5), "attitude from projected gravity (deg)");
        }

        [Test]
        public void Pose_helpers_follow_isaac_axes()
        {
            // Isaac yaw +90 deg faces Isaac +y = Unity -x; moving forward 1 m goes to Unity (-1, 0, 0).
            var pos = Vector3.zero;
            var yaw = Mathf.PI / 2;
            WmPose.Integrate(ref pos, ref yaw, new[] { 1f, 0f, 0f, 0.1f });
            Assert.That((pos - new Vector3(-1f, 0f, 0f)).magnitude, Is.LessThan(1e-5f));
            Assert.AreEqual(Mathf.PI / 2 + 0.1f, yaw, 1e-5f);
            var rot = WmPose.Rotation(yaw, new Vector3(0f, 0f, -1f));
            Assert.AreEqual(yaw, CommandSampler.Heading(rot), 1e-5f);
            Assert.That(Vector3.Angle(rot * Vector3.up, Vector3.up), Is.LessThan(1e-3f));
            // nose down: gravity has a +x component in the base frame
            var pitched = WmPose.Rotation(0f, new Vector3(Mathf.Sin(0.2f), 0f, -Mathf.Cos(0.2f)));
            Assert.Less((pitched * Vector3.forward).y, -0.19f, "positive g_x = nose down");
        }

        [Test]
        public void Unity_calibration_overrides_isaac_thresholds()
        {
            var isaac = SurprisalParams.Isaac(itf, "v2");
            var path = Path.Combine(Path.GetTempPath(), $"bmb_wm_cal_{Guid.NewGuid():N}.json");
            var cal = new UnityWmCalibration
            {
                source = "test",
                z_bias = Enumerable.Repeat(0.5f, 33).ToArray(),
                z_scale = Enumerable.Repeat(2f, 33).ToArray(),
                policies = new[] { new PolicyCusum { policy = "v2", m0 = 1f, s0 = 2f, k = 1f, h = 99f, spike_z = 20f } },
            };
            try
            {
                File.WriteAllText(path, JsonUtility.ToJson(cal));
                var p = SurprisalParams.Load(itf, "v2", preferUnity: true, unityPath: path);
                Assert.AreEqual("unity", p.Source);
                Assert.AreEqual((1f, 2f, 99f, 20f), (p.M0, p.S0, p.H, p.SpikeZ));
                Assert.AreEqual(0.5f, p.ZBias[3]);
                Assert.AreEqual(isaac.VarFloor, p.VarFloor, "everything else stays the yaml's");
                Assert.AreEqual("isaac", SurprisalParams.Load(itf, "v2", preferUnity: false, unityPath: path).Source);
                Assert.Throws<InvalidDataException>(() => SurprisalParams.Load(itf, "v1", true, path), "v1 missing from the file");
                Assert.AreEqual(0f, isaac.ZBias[3], "the yaml values are not modified");
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Test]
        public void Inference_time()
        {
            var w = WmReference.Load("wm");
            var p = new WorldModelPrediction(itf.WorldModel);
            var x = w.wm_in.Row(100);
            var cpu = Time(runner, x, p);
            var gpu = double.NaN;
            try
            {
                using var g = WorldModelRunner.Load(itf, BackendType.GPUCompute);
                gpu = Time(g, x, p);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[BMB] GPUCompute world model unavailable: {e.Message}");
            }
            Debug.Log($"[BMB] World model inference: CPU {cpu:F2} ms, GPUCompute {gpu:F2} ms per call (mean of 200, after warm-up)");
            Assert.That(cpu, Is.LessThan(50.0), "CPU inference far too slow for 50 Hz");
        }

        static double Time(WorldModelRunner r, float[] x, WorldModelPrediction p)
        {
            for (var i = 0; i < 10; i++)
                r.Run(x, p);
            var sw = Stopwatch.StartNew();
            for (var i = 0; i < 200; i++)
                r.Run(x, p);
            return sw.Elapsed.TotalMilliseconds / 200;
        }

        static (Vector3, Quaternion) Pose(ReferenceRollout r, int t) =>
            (IsaacFrame.PolarFromIsaac(Vec3(r.root_link_pos_w, t)),
             IsaacFrame.QuatFromIsaac(r.root_link_quat_w[t, 0], r.root_link_quat_w[t, 1], r.root_link_quat_w[t, 2], r.root_link_quat_w[t, 3]));

        static double MaxAbs(float[] got, NdArray want, int t)
        {
            double m = 0;
            for (var i = 0; i < got.Length; i++)
                m = Math.Max(m, Math.Abs(got[i] - want[t, i]));
            return m;
        }

        static double MaxRel(float[] got, NdArray want, int t)
        {
            double m = 0;
            for (var i = 0; i < got.Length; i++)
                m = Math.Max(m, Math.Abs(got[i] - want[t, i]) / Math.Max(Math.Abs(want[t, i]), 1e-12));
            return m;
        }

        /// <summary>Absolute error below 1, relative above (the reference stores float64 values read as float32).</summary>
        static double Rel(double got, float want) => Math.Abs(got - want) / Math.Max(1.0, Math.Abs(want));

        static Vector3 Vec3(NdArray a, int t) => new Vector3(a[t, 0], a[t, 1], a[t, 2]);
    }
}
