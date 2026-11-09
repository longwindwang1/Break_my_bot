using System;
using System.Collections.Generic;
using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>
    /// The Unity recalibration of the world-model surprisal (docs/unity_handoff.md §11.4): the calibrator's statistics
    /// and CUSUM against the runtime detector, a calibration on synthetic z, and a tiny headless run of both splits.
    /// </summary>
    public class WorldModelEvalTests
    {
        RobotInterface itf;

        [OneTimeSetUp]
        public void Setup() => itf = RobotInterface.Load();

        [Test]
        public void Quantile_interpolates_like_torch()
        {
            Assert.AreEqual(2.5, WmCalibrator.Quantile(new List<float> { 4, 1, 3, 2 }, 0.5), 1e-9);
            Assert.AreEqual(4.0, WmCalibrator.Quantile(new List<float> { 4, 1, 3, 2 }, 1.0), 1e-9);
            Assert.AreEqual(1.3, WmCalibrator.Quantile(new List<float> { 4, 1, 3, 2 }, 0.1), 1e-6);
            var hs = WmCalibrator.HGrid();
            Assert.AreEqual(160, hs.Length);
            Assert.AreEqual(0.2, hs[0], 1e-9);
            Assert.AreEqual(400.0, hs[159], 1e-6);
        }

        [Test]
        public void Offline_cusum_matches_the_runtime_detector()
        {
            // z = s when mean = 0 and var + floor = 1: feed the same z stream to both.
            var p = SurprisalParams.Isaac(itf, "v2");
            var dim = p.Weights.Length;
            var rng = new System.Random(5);
            var e = new WmEpisode { Length = 600, Z = new float[600 * dim] };
            for (var i = 1; i < e.Length; i++)
                for (var d = 0; d < dim; d++)
                    e.Z[i * dim + d] = (float)(Gauss(rng) * (i > 300 ? 2.2 : 1.0) + (i == 450 && d < 2 ? 30 : 0));  // ice-like rise + a push
            var mean = new float[dim];
            var var = p.VarFloor.Select(f => 1f - f).ToArray();
            var det = new SurprisalDetector(p);
            var runtime = new List<int>();
            var s = new float[dim];
            for (var i = 0; i < e.Length; i++)
            {
                Array.Copy(e.Z, i * dim, s, 0, dim);
                det.Step(s, i > 0 ? mean : null, i > 0 ? var : null, i);
                if (det.Alarm)
                    runtime.Add(i);
            }
            var zero = new float[dim];
            var one = Enumerable.Repeat(1f, dim).ToArray();
            var (S, spike) = WmCalibrator.Scores(e, p, zero, one);
            var offline = WmCalibrator.Alarms(e, p, S, spike, p.M0, p.S0, p.K, p.H, p.SpikeZ);
            var offlineAt = Enumerable.Range(0, e.Length).Where(i => offline[i]).ToList();
            Debug.Log($"[BMB] CUSUM runtime vs offline: [{string.Join(", ", runtime)}]");
            Assert.That(runtime, Is.Not.Empty, "the stream must raise alarms");
            CollectionAssert.Contains(runtime, 450, "the push spike");
            CollectionAssert.AreEqual(runtime, offlineAt);
        }

        [Test]
        public void Calibration_recovers_bias_and_scale_and_meets_the_false_alarm_budget()
        {
            var p = new[] { SurprisalParams.Isaac(itf, "v1"), SurprisalParams.Isaac(itf, "v2") };
            var dim = p[0].Weights.Length;
            var rng = new System.Random(11);
            var episodes = new List<WmEpisode>();
            // The spike threshold is the 1 − 1/60000 quantile: half A needs ≥ 2 × target_arl_steps scored steps
            // (handoff §11.4), here 65 envs × 975.
            for (var pi = 0; pi < 2; pi++)
                for (var env = 0; env < 130; env++)
                {
                    var e = new WmEpisode { Policy = pi, Env = env, Length = 1000, Z = new float[1000 * dim] };
                    for (var i = 1; i < e.Length; i++)
                        for (var d = 0; d < dim; d++)
                            e.Z[i * dim + d] = (float)(0.3 + 1.5 * Gauss(rng));
                    episodes.Add(e);
                }
            var (bias, scale, t) = WmCalibrator.Calibrate(episodes, p, 50);
            Debug.Log($"[BMB] synthetic calibration: bias {bias.Average():F3}, scale {scale.Average():F3}, " +
                      string.Join("; ", t.Select(kv => $"{kv.Key} m0 {kv.Value.M0:F3} s0 {kv.Value.S0:F3} h {kv.Value.H:F1} spike {kv.Value.SpikeZ:F2} far A {kv.Value.FarCalib:E1} B {kv.Value.FarVerify:E1}")));
            Assert.AreEqual(0.3, bias.Average(), 0.02);
            Assert.AreEqual(1.5, scale.Average(), 0.03);
            foreach (var kv in t)
            {
                Assert.That(kv.Value.FarCalib, Is.LessThanOrEqualTo(1.0 / p[0].TargetArlSteps), kv.Key);
                Assert.That(kv.Value.ScoredA, Is.EqualTo(65 * (1000 - p[0].WarmupSteps)), kv.Key);
                Assert.That(kv.Value.SpikeZ, Is.InRange(3.5f, 6f), "max |z| of 2 dims at ~1 - 1/60000 for N(0, 1)");
            }
        }

        [Test]
        public void Headless_runs_record_both_splits()
        {
            var clean = new List<WmEpisode>();
            var o = new WmEvalOptions { EnvsPerPolicy = 2, Seconds = 2f, EpisodeSeconds = 1f, TilesPerSide = 1, Border = 4f, Difficulty = new Vector2(0f, 0.2f) };
            var it = WorldModelEval.Run(o, clean);
            while (it.MoveNext()) { }
            Assert.AreEqual(8, clean.Count, "2 envs x 2 policies x 2 episodes");
            var dim = itf.WorldModel.OutputDim("state_mean");
            foreach (var e in clean)
            {
                Assert.AreEqual(50, e.Length);
                Assert.AreEqual(50 * dim, e.Z.Length);
                Assert.IsTrue(e.Z.Take(dim).All(v => v == 0f), "no prediction at the episode start");
                Assert.IsTrue(e.Z.Skip(dim).Any(v => v != 0f), "z recorded");
            }

            var events = new List<WmEpisode>();
            var ev = new WmEvalOptions { Events = true, EnvsPerPolicy = WmEvents.Count, Seconds = 1.2f, EpisodeSeconds = 1.2f, EventStep = 30, TilesPerSide = 1, Border = 4f, Difficulty = new Vector2(0f, 0.2f) };
            it = WorldModelEval.Run(ev, events);
            while (it.MoveNext()) { }
            Assert.AreEqual(2 * WmEvents.Count, events.Count);
            CollectionAssert.AreEquivalent(Enumerable.Range(0, WmEvents.Count).SelectMany(i => new[] { i, i }), events.Select(e => e.Event));
            Assert.IsTrue(events.Where(e => !e.FellBeforeEvent).All(e => e.EventAt == 30));
            // the 1.0 m/s push is a spike at the step after it (|z| of base_lin_vel x, y far above 16)
            var push = events.First(e => WmEvents.Names[e.Event] == "push_1.0" && e.Policy == 1);
            var after = Math.Max(Math.Abs(push.Z[31 * dim]), Math.Abs(push.Z[31 * dim + 1]));
            Debug.Log($"[BMB] events smoke: push_1.0 |z| base vel x/y at the next step {after:F1}");
            Assert.That(after, Is.GreaterThan(16f));
        }

        [Test]
        public void Unity_calibration_is_quiet_on_clean_walking_and_catches_ice()
        {
            // Verification level 4 (handoff §11.3): closed loop in Unity with Unity's own thresholds. v2 walks and turns
            // on flat ground for 30 s without an alarm; then every foot goes to μ 0.05 and an alarm follows within 2 s.
            Assert.IsTrue(System.IO.File.Exists(RepoPaths.WmCalibrationUnity), "run BMB/World model/Calibrate first");
            using var world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero, footFriction: 1f, command: new Vector3(0.5f, 0f, 0.3f), worldModel: true);
            var d = robot.WorldModel.Detector;
            Assert.AreEqual("unity", d.Params.Source, "the runtime prefers wm_calibration_unity.json");
            world.Run(30f);
            var clean = d.Alarms;
            robot.FootFriction.SetFootMu(0.05f);
            var dropAt = robot.SimTime;
            var firstAlarm = float.NaN;
            robot.SurpriseAlarm += (c, m) =>
            {
                if (float.IsNaN(firstAlarm))
                    firstAlarm = c.SimTime - dropAt;
            };
            world.Run(2f);
            Debug.Log($"[BMB] Level 4 (Unity calibration): {clean} alarms in 30 s of clean walking ({d.ScoredSteps} scored steps); " +
                      $"μ 0.05 detected after {firstAlarm:F2} s, falls {robot.Falls}");
            Assert.AreEqual(0, clean, "no false alarm on clean walking");
            Assert.IsFalse(float.IsNaN(firstAlarm), "ice detected within 2 s");  // NUnit orders NaN below every number
            Assert.That(firstAlarm, Is.LessThanOrEqualTo(2f), "ice detected within 2 s");
        }

        static double Gauss(System.Random rng)
        {
            var u1 = 1.0 - rng.NextDouble();
            var u2 = rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}
