using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace BMB.Editor
{
    /// <summary>Events of Isaac's events split (experiments/w4_wm_data): env i gets event i % 7 at EventStep.</summary>
    public static class WmEvents
    {
        public static readonly string[] Names = { "none", "mu_0.3", "mu_0.1", "mu_0.05", "foot_0.05", "push_0.5", "push_1.0" };
        public const int Count = 7;
    }

    public sealed class WmEvalOptions
    {
        public int EnvsPerPolicy = 64;
        /// <summary>Simulated time per robot (s); clean runs restart episodes, the events run is one episode.</summary>
        public float Seconds = 60f;
        public float EpisodeSeconds = 20f;
        public int Seed = 200;
        public int TilesPerSide = 6;
        public float Border = 20f;
        /// <summary>Isaac tile mix and difficulty range of the arena; <see cref="FlatShare"/> of the tiles are flat instead.</summary>
        public (IsaacTileType type, float proportion)[] Mix = IsaacTiles.V2Mix;
        public Vector2 Difficulty = new Vector2(0f, 1f);
        public float FlatShare;
        /// <summary>Foot μ drawn per episode ("normal physics").</summary>
        public Vector2 FootMu = new Vector2(0.6f, 1.25f);
        public EvalPolicy[] Policies = { EvalPolicy.V1, EvalPolicy.V2 };
        /// <summary>Events split: one 12 s episode, event i % 7 at <see cref="EventStep"/>.</summary>
        public bool Events;
        public int EventStep = 200;

        /// <summary>
        /// Calibration data as docs/unity_handoff.md §11.4 asks: clean walking on flat and rough ground (half the tiles
        /// flat, half Isaac random_rough at difficulty 0–1).
        /// </summary>
        public static WmEvalOptions Clean() => new WmEvalOptions
        {
            Mix = new[] { (IsaacTileType.RandomRough, 1f) }, FlatShare = 0.5f,
        };

        /// <summary>Isaac's clean split terrain (the full training mix at difficulty 0–1), for comparison.</summary>
        public static WmEvalOptions CleanIsaacTerrain() => new WmEvalOptions();

        /// <summary>Isaac's events split: the training mix at difficulty 0–0.5, event i % 7 at step 200.</summary>
        public static WmEvalOptions EventsSplit() => new WmEvalOptions
        {
            Events = true, EnvsPerPolicy = 7 * 28, Seconds = 12f, EpisodeSeconds = 12f, Seed = 300, Difficulty = new Vector2(0f, 0.5f),
        };

        public string TerrainText => FlatShare > 0f
            ? $"{FlatShare:P0} flat tiles, the rest {string.Join(" / ", Mix.Select(m => m.type))} at difficulty {Difficulty.x:0.##}–{Difficulty.y:0.##}"
            : $"Isaac tiles ({(Mix == IsaacTiles.V2Mix ? "training mix" : string.Join(" / ", Mix.Select(m => m.type)))}) at difficulty {Difficulty.x:0.##}–{Difficulty.y:0.##}";
    }

    /// <summary>The raw surprisal record of one episode of one robot (z without engine correction).</summary>
    public sealed class WmEpisode
    {
        public int Policy, Env;
        /// <summary>Event type (<see cref="WmEvents"/>) and the episode step it happened at; -1 without.</summary>
        public int Event = -1, EventAt = -1;
        public int Length;
        /// <summary>Length × state dim; row i = z of episode step i against the prediction from i − 1 (0 at i = 0).</summary>
        public float[] Z;
        public bool Fell;
        /// <summary>Events split: the robot fell before its event (Isaac's reset_before_event).</summary>
        public bool FellBeforeEvent;
    }

    /// <summary>Per-dim z correction used to recalibrate (see <see cref="WmCalibrator.Correction"/>).</summary>
    public enum ZCorrection { Mad, None, IsaacStd }

    /// <summary>Thresholds of one policy (the yaml's surprisal block, Unity's z correction applied separately).</summary>
    public sealed class WmThresholds
    {
        public float M0, S0, K, H, SpikeZ;
        public double FarCalib, FarVerify;
        public int ScoredA, ScoredB;
    }

    /// <summary>
    /// Surprisal calibration and detection statistics on recorded episodes; port of models/world_model/evaluate.py
    /// (calibrate) for the deployed configuration (statistic, weights, clip and k from the yaml).
    /// </summary>
    public static class WmCalibrator
    {
        /// <summary>torch.quantile (linear interpolation) of a sorted array.</summary>
        public static double QuantileSorted(IReadOnlyList<float> sorted, double q)
        {
            if (sorted.Count == 0)
                return double.NaN;
            var pos = q * (sorted.Count - 1);
            var lo = (int)Math.Floor(pos);
            var hi = Math.Min(lo + 1, sorted.Count - 1);
            return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
        }

        public static double Quantile(List<float> values, double q)
        {
            values.Sort();
            return QuantileSorted(values, q);
        }

        public static bool Scored(WmEpisode e, int i, SurprisalParams p) => i >= 1 && i >= p.WarmupSteps;

        /// <summary>Within the longest fall horizon before a fall: an alarm there is a hit, not a false alarm.</summary>
        public static bool Prefall(WmEpisode e, int i, int horizon) => e.Fell && i >= e.Length - horizon;

        /// <summary>Corrected z of row i, dim d.</summary>
        static double Zc(WmEpisode e, int i, int d, int dim, float[] bias, float[] scale) =>
            ((double)e.Z[i * dim + d] - bias[d]) / scale[d];

        /// <summary>S (statistic "norm") and the spike statistic of every row, with a z correction.</summary>
        public static (double[] s, double[] spike) Scores(WmEpisode e, SurprisalParams p, float[] bias, float[] scale)
        {
            if (p.Nll)
                throw new NotSupportedException("recalibration implements the yaml's statistic 'norm' only");
            var dim = p.Weights.Length;
            var s = new double[e.Length];
            var spike = new double[e.Length];
            for (var i = 1; i < e.Length; i++)
            {
                double sum = 0, sp = 0;
                for (var d = 0; d < dim; d++)
                {
                    var z = Zc(e, i, d, dim, bias, scale);
                    sum += p.Weights[d] * 0.5 * (Math.Min(z * z, p.ClipZ2) - 1.0);
                }
                foreach (var d in p.SpikeDims)
                    sp = Math.Max(sp, Math.Abs(Zc(e, i, d, dim, bias, scale)));
                s[i] = sum;
                spike[i] = sp;
            }
            return (s, spike);
        }

        /// <summary>
        /// CUSUM + spike alarms of one episode, exactly as <see cref="SurprisalDetector"/> runs them. Returns the
        /// alarm flags per row.
        /// </summary>
        public static bool[] Alarms(WmEpisode e, SurprisalParams p, double[] s, double[] spike, float m0, float s0, float k, float h, float spikeZ)
        {
            var alarm = new bool[e.Length];
            double g = 0;
            for (var i = 0; i < e.Length; i++)
            {
                if (i == 0)
                    g = 0;
                if (!Scored(e, i, p))
                    continue;
                g = Math.Max(0.0, g + (s[i] - m0) / s0 - k);
                var a = g > h || spike[i] > spikeZ;
                alarm[i] = a;
                if (a)
                    g = 0;
            }
            return alarm;
        }

        /// <summary>Isaac's h grid: 160 values log-spaced from 0.2 to 400.</summary>
        public static double[] HGrid()
        {
            var hs = new double[160];
            double a = Math.Log10(0.2), b = Math.Log10(400.0);
            for (var i = 0; i < hs.Length; i++)
                hs[i] = Math.Pow(10.0, a + (b - a) * i / (hs.Length - 1));
            return hs;
        }

        static bool HalfA(WmEpisode e) => e.Env % 2 == 0;

        /// <summary>
        /// Per-dim z correction from clean half A (env % 2 == 0; scored steps not before a fall; both policies pooled,
        /// as the yaml has one correction):
        /// - <see cref="ZCorrection.Mad"/>: docs/unity_handoff.md §11.4, z_bias = median, z_scale = 1.4826 · MAD;
        /// - <see cref="ZCorrection.None"/>: identity (only m0, s0, h, spike_z are re-estimated);
        /// - <see cref="ZCorrection.IsaacStd"/>: z_bias = median, z_scale = Unity's z std / Isaac's clean z std of the
        ///   dim's state term (experiments/w5_surprisal one_step_clean), so the spread matches the model's own data.
        /// </summary>
        public static (float[] bias, float[] scale) Correction(List<WmEpisode> episodes, ZCorrection mode, SurprisalParams p0,
            int fallHorizon, float[] isaacStdPerDim = null)
        {
            var dim = p0.Weights.Length;
            var bias = new float[dim];
            var scale = Enumerable.Repeat(1f, dim).ToArray();
            if (mode == ZCorrection.None)
                return (bias, scale);
            var column = new List<float>();
            for (var d = 0; d < dim; d++)
            {
                column.Clear();
                foreach (var e in episodes.Where(HalfA))
                    for (var i = 0; i < e.Length; i++)
                        if (Scored(e, i, p0) && !Prefall(e, i, fallHorizon))
                            column.Add(e.Z[i * dim + d]);
                double mean = column.Average(v => (double)v);
                var std = Math.Sqrt(column.Sum(v => (v - mean) * (v - mean)) / Math.Max(1, column.Count - 1));
                var med = Quantile(column, 0.5);
                bias[d] = (float)med;
                if (mode == ZCorrection.Mad)
                {
                    for (var j = 0; j < column.Count; j++)
                        column[j] = (float)Math.Abs(column[j] - med);
                    scale[d] = (float)Math.Max(1.4826 * Quantile(column, 0.5), 1e-6);
                }
                else
                    scale[d] = (float)Math.Max(std / (isaacStdPerDim ?? throw new ArgumentNullException(nameof(isaacStdPerDim)))[d], 1e-6);
            }
            return (bias, scale);
        }

        /// <summary>Calibration with the handoff's MAD correction (see <see cref="Thresholds"/>).</summary>
        public static (float[] bias, float[] scale, Dictionary<string, WmThresholds> perPolicy) Calibrate(
            List<WmEpisode> episodes, IReadOnlyList<SurprisalParams> isaac, int fallHorizon)
        {
            var (bias, scale) = Correction(episodes, ZCorrection.Mad, isaac[0], fallHorizon);
            return (bias, scale, Thresholds(episodes, isaac, bias, scale, fallHorizon));
        }

        /// <summary>
        /// Unity thresholds (docs/unity_handoff.md §11.4) on clean episodes with a z correction. Half A calibrates, half B
        /// verifies. Per policy: m0, s0 of S; spike_z the 1 − far/2 quantile of the spike statistic; h the smallest grid
        /// value whose false alarms (CUSUM or spike, not before a fall) per scored step stay within
        /// far = 1 / target_arl_steps. k is kept.
        /// </summary>
        public static Dictionary<string, WmThresholds> Thresholds(List<WmEpisode> episodes, IReadOnlyList<SurprisalParams> isaac,
            float[] bias, float[] scale, int fallHorizon)
        {
            var far = 1.0 / isaac[0].TargetArlSteps;
            var result = new Dictionary<string, WmThresholds>();
            var hs = HGrid();
            for (var pi = 0; pi < isaac.Count; pi++)
            {
                var p = isaac[pi];
                var mine = episodes.Where(e => e.Policy == pi).ToList();
                var scores = mine.ToDictionary(e => e, e => Scores(e, p, bias, scale));
                var a = mine.Where(HalfA).ToList();
                var b = mine.Where(e => !HalfA(e)).ToList();

                // spike threshold: 1 − far / 2 quantile on half A (scored, not before a fall)
                var spikes = new List<float>();
                var sVals = new List<double>();
                foreach (var e in a)
                    for (var i = 0; i < e.Length; i++)
                        if (Scored(e, i, p))
                        {
                            sVals.Add(scores[e].s[i]);
                            if (!Prefall(e, i, fallHorizon))
                                spikes.Add((float)scores[e].spike[i]);
                        }
                var spikeZ = (float)Quantile(spikes, 1.0 - far / 2.0);
                var m0 = sVals.Average();
                var s0 = Math.Sqrt(sVals.Sum(v => (v - m0) * (v - m0)) / Math.Max(1, sVals.Count - 1));

                // smallest h meeting the budget on half A (CUSUM for every h in one pass per episode)
                var falseAlarms = new long[hs.Length];
                foreach (var e in a)
                {
                    var (s, spike) = scores[e];
                    var g = new double[hs.Length];
                    for (var i = 0; i < e.Length; i++)
                    {
                        if (!Scored(e, i, p))
                            continue;
                        var x = (s[i] - m0) / s0;
                        var sp = spike[i] > spikeZ;
                        var pre = Prefall(e, i, fallHorizon);
                        for (var j = 0; j < hs.Length; j++)
                        {
                            g[j] = Math.Max(0.0, g[j] + x - p.K);
                            if (g[j] > hs[j] || sp)
                            {
                                if (!pre)
                                    falseAlarms[j]++;
                                g[j] = 0;
                            }
                        }
                    }
                }
                var scoredA = sVals.Count;
                var chosen = hs.Length - 1;
                for (var j = 0; j < hs.Length; j++)
                    if (falseAlarms[j] / (double)Math.Max(1, scoredA) <= far)
                    {
                        chosen = j;
                        break;
                    }
                var t = new WmThresholds
                {
                    M0 = (float)m0, S0 = (float)s0, K = p.K, H = (float)hs[chosen], SpikeZ = spikeZ,
                    FarCalib = falseAlarms[chosen] / (double)Math.Max(1, scoredA), ScoredA = scoredA,
                };
                var (nb, sb) = FalseAlarms(b, p, e => scores[e], t, fallHorizon);
                t.FarVerify = nb / (double)Math.Max(1, sb);
                t.ScoredB = sb;
                result[p.Policy] = t;
            }
            return result;
        }

        /// <summary>False alarms (not before a fall) and scored steps over episodes with given thresholds.</summary>
        public static (long alarms, int scored) FalseAlarms(IEnumerable<WmEpisode> episodes, SurprisalParams p,
            Func<WmEpisode, (double[] s, double[] spike)> scores, WmThresholds t, int fallHorizon)
        {
            long n = 0;
            var scored = 0;
            foreach (var e in episodes)
            {
                var (s, spike) = scores(e);
                var alarms = Alarms(e, p, s, spike, t.M0, t.S0, t.K, t.H, t.SpikeZ);
                for (var i = 0; i < e.Length; i++)
                {
                    if (!Scored(e, i, p))
                        continue;
                    scored++;
                    if (alarms[i] && !Prefall(e, i, fallHorizon))
                        n++;
                }
            }
            return (n, scored);
        }

        public static WmThresholds FromParams(SurprisalParams p) =>
            new WmThresholds { M0 = p.M0, S0 = p.S0, K = p.K, H = p.H, SpikeZ = p.SpikeZ };

        /// <summary>Per state term: std of raw z, share within 1 and 2 (scored, not before a fall).</summary>
        public static List<(string term, double std, double within1, double within2)> TermStats(
            List<WmEpisode> episodes, SurprisalParams p, WorldModelLayout layout, int fallHorizon, int policy = -1)
        {
            var dim = p.Weights.Length;
            var stats = new List<(string, double, double, double)>();
            var offset = 0;
            for (var k = 0; k < layout.FrameTerms.Length - 1; k++)  // state terms = frame terms without actions
            {
                double sum2 = 0;
                long n = 0, w1 = 0, w2 = 0;
                foreach (var e in episodes)
                {
                    if (policy >= 0 && e.Policy != policy)
                        continue;
                    for (var i = 0; i < e.Length; i++)
                    {
                        if (!Scored(e, i, p) || Prefall(e, i, fallHorizon))
                            continue;
                        for (var d = offset; d < offset + layout.FrameTermDims[k]; d++)
                        {
                            var z = e.Z[i * dim + d];
                            sum2 += z * z;
                            n++;
                            if (Math.Abs(z) < 1f) w1++;
                            if (Math.Abs(z) < 2f) w2++;
                        }
                    }
                }
                stats.Add((layout.FrameTerms[k], Math.Sqrt(sum2 / Math.Max(1, n)), w1 / (double)Math.Max(1, n), w2 / (double)Math.Max(1, n)));
                offset += layout.FrameTermDims[k];
            }
            return stats;
        }
    }

    /// <summary>
    /// World-model surprisal in Unity physics (docs/unity_handoff.md §11.4; Experiment 3):
    /// - Clean walking like Isaac's clean split (experiments/w4_wm_data): Isaac tiles of the training mix at difficulty
    ///   0–1, foot μ 0.6–1.25 per episode, payload / COM / gain randomization, commands as in training, no pushes,
    ///   20 s episodes restarting on the same spot after a fall or time-out. Its raw z gives Unity's calibration
    ///   (experiments/w3_unity/wm_calibration_unity.json) and the clean false-alarm rates.
    /// - The events split: one 12 s episode per robot, event i % 7 at step 200 (none, all feet μ 0.3 / 0.1 / 0.05,
    ///   one foot μ 0.05, pushes of 0.5 / 1.0 m/s), for detection rates and delays (Unity vs Isaac).
    /// Both policies run side by side in one headless world, env e of each paired (same tile, spawn, dynamics, commands).
    /// </summary>
    public static class WorldModelEval
    {
        public static string OutputDir => Path.Combine(RepoPaths.RepoRoot, "experiments", "unity_wm");
        public static string IsaacResults => Path.Combine(RepoPaths.RepoRoot, "experiments", "w5_surprisal", "results.json");

        sealed class Robot
        {
            public Go2Controller Controller;
            public int Env, Policy;
            public CommandSampler Commands;
            public System.Random Rng;
            public WmEpisode Episode;
            public List<float> Z = new List<float>();
            public bool Done;
        }

        /// <summary>Runs a split; yields progress (0–1) after every policy step. Finished episodes go to <paramref name="episodes"/>.</summary>
        public static IEnumerator<float> Run(WmEvalOptions o, List<WmEpisode> episodes)
        {
            var world = HeadlessWorld.Create(flatGround: false);
            var policies = new List<PolicyRunner>();
            WorldModelRunner wm = null;
            try
            {
                var itf = world.Interface;
                UnityEval.BuildTileArena(world, o.TilesPerSide, o.Border, o.Seed, o.Mix, o.Difficulty, out var tiles, o.FlatShare);
                foreach (var p in o.Policies)
                    policies.Add(PolicyRunner.Load(p.Variant, itf));
                wm = WorldModelRunner.Load(itf);
                var order = UnityEval.TileOrder(tiles.Count, o.Seed);
                var robots = new List<Robot>();
                for (var e = 0; e < o.EnvsPerPolicy; e++)
                {
                    var draw = UnityEval.EnvDraw.Sample(o.Seed, e, itf.NumJoints);
                    var tile = tiles[order[e % order.Length]];
                    var spawn = new Vector3(tile.centre.x, 0f, tile.centre.y) + draw.Offset;
                    for (var p = 0; p < o.Policies.Length; p++)
                    {
                        // raw z: the yaml's correction is the identity (asserted below)
                        var raw = SurprisalParams.Isaac(itf, WorldModelSpec.PolicyName(o.Policies[p].Variant));
                        var c = world.AddRobot(o.Policies[p].Variant, spawn, draw.Yaw, footFriction: 1f, resetOnFall: false,
                                               sharedPolicy: policies[p], sharedWorldModel: wm, surprisal: raw);
                        c.SetFallCriteria(0f, 0f);
                        draw.ApplyDynamics(c);
                        var r = new Robot
                        {
                            Controller = c, Env = e, Policy = p,
                            Commands = new CommandSampler(itf.Commands, new System.Random(draw.CommandSeed)),
                            Rng = new System.Random(draw.PushSeed),
                        };
                        StartEpisode(r, o);
                        if (o.Events)
                            // Like the Isaac script: the event happens during step E (after o_E and a_E), so o_{E+1} shows it.
                            c.PolicyStepped += ctl =>
                            {
                                if (r.Episode.EventAt < 0 && ctl.PhysicsStep / ctl.Interface.Timing.Decimation == o.EventStep)
                                    ApplyEvent(r, o);
                            };
                        robots.Add(r);
                    }
                }
                var p0 = robots[0].Controller.WorldModel.Detector.Params;
                if (p0.ZBias.Any(v => v != 0f) || p0.ZScale.Any(v => v != 1f))
                    throw new InvalidOperationException("the yaml's z correction is not the identity; raw z would be biased");

                var terrainMask = LayerMask.GetMask(SimSettings.TerrainLayer);
                var decimation = itf.Timing.Decimation;
                var dt = itf.Timing.PolicyDt;
                var steps = Mathf.RoundToInt(o.Seconds / dt);
                var episodeSteps = Mathf.RoundToInt(o.EpisodeSeconds / dt);
                var dim = itf.WorldModel.OutputDim("state_mean");
                for (var k = 0; k < steps; k++)
                {
                    foreach (var r in robots)
                    {
                        if (r.Done)
                            continue;
                        var c = r.Controller;
                        c.CommandOverride = r.Commands.Next(dt, CommandSampler.Heading(c.Body.BaseRotation));
                    }
                    for (var i = 0; i < decimation; i++)
                        world.Step();

                    foreach (var r in robots)
                    {
                        if (r.Done)
                            continue;
                        var c = r.Controller;
                        var d = c.WorldModel.Detector;
                        for (var j = 0; j < dim; j++)
                            r.Z.Add(d.Valid ? (float)d.Z[j] : 0f);
                        r.Episode.Length++;
                        var fell = c.Body.BaseTouches(world.Physics, terrainMask) || c.Body.BasePosition.y < -5f;
                        var timeOut = r.Episode.Length >= episodeSteps;
                        if (!fell && !timeOut)
                            continue;
                        r.Episode.Fell = fell;
                        r.Episode.FellBeforeEvent = o.Events && fell && r.Episode.EventAt < 0;
                        FinishEpisode(r, episodes);
                        if (o.Events)
                        {
                            r.Done = true;
                            world.Retire(c);
                        }
                        else
                        {
                            c.ResetRobot();
                            StartEpisode(r, o);
                        }
                    }
                    yield return (k + 1f) / steps;
                }
                foreach (var r in robots.Where(r => !r.Done && r.Episode.Length > 0))
                    FinishEpisode(r, episodes);
            }
            finally
            {
                world.Dispose();
                foreach (var r in policies)
                    r.Dispose();
                wm?.Dispose();
            }
        }

        static void StartEpisode(Robot r, WmEvalOptions o)
        {
            r.Controller.FootFriction.SetFootMu(UnityEval.Uniform(r.Rng, o.FootMu.x, o.FootMu.y));
            r.Commands.Reset();
            r.Z.Clear();
            r.Episode = new WmEpisode { Policy = r.Policy, Env = r.Env, Event = o.Events ? r.Env % WmEvents.Count : -1 };
        }

        static void FinishEpisode(Robot r, List<WmEpisode> episodes)
        {
            r.Episode.Z = r.Z.ToArray();
            episodes.Add(r.Episode);
        }

        static void ApplyEvent(Robot r, WmEvalOptions o)
        {
            var c = r.Controller;
            r.Episode.EventAt = o.EventStep;
            switch (WmEvents.Names[r.Episode.Event])
            {
                case "mu_0.3": c.FootFriction.SetFootMu(0.3f); break;
                case "mu_0.1": c.FootFriction.SetFootMu(0.1f); break;
                case "mu_0.05": c.FootFriction.SetFootMu(0.05f); break;
                case "foot_0.05": c.FootFriction.SetFootMu(r.Env / WmEvents.Count % 4, 0.05f); break;
                case "push_0.5":
                case "push_1.0":
                {
                    var speed = WmEvents.Names[r.Episode.Event] == "push_0.5" ? 0.5f : 1f;
                    var angle = UnityEval.Uniform(r.Rng, 0f, 2f * Mathf.PI);
                    c.Body.Root.linearVelocity += IsaacFrame.PolarFromIsaac(new Vector3(Mathf.Cos(angle), Mathf.Sin(angle), 0f) * speed);
                    break;
                }
            }
        }

        // --- logs (raw z of every episode; logs/ is gitignored) -------------------------------------------------

        public static string LogDir => Path.Combine(RepoPaths.RepoRoot, "logs", "unity_wm");

        public static void SaveEpisodes(string path, List<WmEpisode> episodes)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            using var w = new BinaryWriter(File.Create(path));
            w.Write(episodes.Count);
            foreach (var e in episodes)
            {
                w.Write(e.Policy); w.Write(e.Env); w.Write(e.Event); w.Write(e.EventAt); w.Write(e.Length);
                w.Write(e.Fell); w.Write(e.FellBeforeEvent);
                w.Write(e.Z.Length);
                foreach (var v in e.Z)
                    w.Write(v);
            }
        }

        public static List<WmEpisode> LoadEpisodes(string path)
        {
            using var r = new BinaryReader(File.OpenRead(path));
            var n = r.ReadInt32();
            var list = new List<WmEpisode>(n);
            for (var k = 0; k < n; k++)
            {
                var e = new WmEpisode
                {
                    Policy = r.ReadInt32(), Env = r.ReadInt32(), Event = r.ReadInt32(), EventAt = r.ReadInt32(), Length = r.ReadInt32(),
                    Fell = r.ReadBoolean(), FellBeforeEvent = r.ReadBoolean(),
                };
                e.Z = new float[r.ReadInt32()];
                for (var i = 0; i < e.Z.Length; i++)
                    e.Z[i] = r.ReadSingle();
                list.Add(e);
            }
            return list;
        }

        // --- detection ---------------------------------------------------------------------------------------------

        public sealed class Detection
        {
            public int Envs;
            public double Detect1s, Detect2s, Detect3s, MedianDelay = double.NaN;
        }

        /// <summary>The events Isaac selected its configuration on (experiments/w5_surprisal/README.md §4).</summary>
        public static readonly string[] KeyEvents = { "mu_0.1", "mu_0.05", "foot_0.05", "push_0.5", "push_1.0" };

        /// <summary>
        /// Detection of each event for one policy: the first alarm 1..150 steps after it, as evaluate.calibrate counts it.
        /// Robots that fell before their event are left out.
        /// </summary>
        public static Dictionary<string, Detection> Detect(List<WmEpisode> events, int policy, SurprisalParams p,
                                                           float[] bias, float[] scale, WmThresholds t)
        {
            var result = new Dictionary<string, Detection>();
            for (var ev = 0; ev < WmEvents.Count; ev++)
            {
                var delays = new List<int>();
                foreach (var e in events.Where(e => e.Policy == policy && e.Event == ev && !e.FellBeforeEvent && e.EventAt >= 0))
                {
                    var (s, spike) = WmCalibrator.Scores(e, p, bias, scale);
                    var alarms = WmCalibrator.Alarms(e, p, s, spike, t.M0, t.S0, t.K, t.H, t.SpikeZ);
                    var first = int.MaxValue;
                    for (var i = e.EventAt + 1; i < Math.Min(e.Length, e.EventAt + 151); i++)
                        if (alarms[i])
                        {
                            first = i - e.EventAt;
                            break;
                        }
                    delays.Add(first);
                }
                var envs = delays.Count;
                var hits = delays.Where(d => d <= 150).Select(d => (float)d).ToList();
                result[WmEvents.Names[ev]] = new Detection
                {
                    Envs = envs,
                    Detect1s = delays.Count(d => d <= 50) / (double)Math.Max(1, envs),
                    Detect2s = delays.Count(d => d <= 100) / (double)Math.Max(1, envs),
                    Detect3s = hits.Count / (double)Math.Max(1, envs),
                    MedianDelay = hits.Count > 0 ? WmCalibrator.Quantile(hits, 0.5) : double.NaN,
                };
            }
            return result;
        }

        /// <summary>Isaac's detection per event (results.json surprisal.best.per_policy.&lt;policy&gt;.events).</summary>
        static Dictionary<string, Detection> ReadIsaacDetection(string policy)
        {
            var result = new Dictionary<string, Detection>();
            if (!File.Exists(IsaacResults))
                return result;
            var text = File.ReadAllText(IsaacResults);
            var at = text.IndexOf("\"surprisal\"", StringComparison.Ordinal);
            foreach (var key in new[] { "\"best\"", "\"per_policy\"", $"\"{policy}\"", "\"events\"" })
                at = at < 0 ? -1 : text.IndexOf(key, at, StringComparison.Ordinal);
            if (at < 0)
                return result;
            foreach (var name in WmEvents.Names)
            {
                var e = text.IndexOf($"\"{name}\"", at, StringComparison.Ordinal);
                if (e < 0)
                    continue;
                double Field(string f)
                {
                    var i = text.IndexOf($"\"{f}\"", e, StringComparison.Ordinal);
                    var colon = text.IndexOf(':', i);
                    var end = text.IndexOfAny(new[] { ',', '}' }, colon);
                    var v = text.Substring(colon + 1, end - colon - 1).Trim();
                    return v == "NaN" || v == "null" ? double.NaN : double.Parse(v, CultureInfo.InvariantCulture);
                }
                result[name] = new Detection
                {
                    Envs = (int)Field("envs"), Detect1s = Field("detect_1s"), Detect2s = Field("detect_2s"),
                    Detect3s = Field("detect_3s"), MedianDelay = Field("median_delay_steps"),
                };
            }
            return result;
        }

        // --- analysis: calibration variants, detection, outputs ------------------------------------------------------

        sealed class Variant
        {
            public string Name;
            public float[] Bias, Scale;
            public Dictionary<string, WmThresholds> Thresholds;
            public Dictionary<string, Dictionary<string, Detection>> Detection = new Dictionary<string, Dictionary<string, Detection>>();
            public double KeyDetect2s;
            public bool Eligible;
        }

        /// <summary>
        /// Recalibrates on the clean split with each z correction (<see cref="ZCorrection"/>), scores the events split
        /// with each, and keeps the variant with the highest mean detection within 2 s over <see cref="KeyEvents"/>
        /// among those whose clean half-B false-alarm rate is at most 2× the target for both policies (Isaac's own
        /// selection rule). Writes experiments/w3_unity/wm_calibration_unity.json and
        /// experiments/unity_wm/{calibration.md, detection.md}. Returns a one-line summary.
        /// </summary>
        public static string Analyze(List<WmEpisode> clean, List<WmEpisode> events, WmEvalOptions co, WmEvalOptions eo, double wallSeconds)
        {
            var itf = RobotInterface.Load();
            var names = co.Policies.Select(p => WorldModelSpec.PolicyName(p.Variant)).ToArray();
            var isaac = names.Select(n => SurprisalParams.Isaac(itf, n)).ToList();
            var horizon = itf.WorldModel.FallHorizonsSteps.Max();
            var layout = new WorldModelLayout(itf, co.Policies[0].Variant);
            var dim = isaac[0].Weights.Length;
            var far = 1.0 / isaac[0].TargetArlSteps;
            var isaacTerms = ReadIsaacTerms();
            float[] isaacStd = null;
            if (isaacTerms.Count == layout.FrameTerms.Length - 1)
            {
                isaacStd = new float[dim];
                var o = 0;
                for (var k = 0; k < layout.FrameTerms.Length - 1; k++)
                    for (var j = 0; j < layout.FrameTermDims[k]; j++)
                        isaacStd[o++] = (float)isaacTerms[layout.FrameTerms[k]].std;
            }
            var zero = new float[dim];
            var one = Enumerable.Repeat(1f, dim).ToArray();

            Variant Evaluate(string name, float[] bias, float[] scale, Dictionary<string, WmThresholds> t)
            {
                var v = new Variant { Name = name, Bias = bias, Scale = scale, Thresholds = t };
                var key = new List<double>();
                for (var pi = 0; pi < names.Length; pi++)
                {
                    v.Detection[names[pi]] = Detect(events, pi, isaac[pi], bias, scale, t[names[pi]]);
                    key.AddRange(KeyEvents.Select(k => v.Detection[names[pi]][k].Detect2s));
                }
                v.KeyDetect2s = key.Average();
                v.Eligible = t.Values.All(x => x.FarVerify <= 2 * far);
                return v;
            }

            // Isaac's thresholds as they are (no correction): false alarms over all clean data
            var isaacT = new Dictionary<string, WmThresholds>();
            for (var pi = 0; pi < names.Length; pi++)
            {
                var t = WmCalibrator.FromParams(isaac[pi]);
                var mine = clean.Where(e => e.Policy == pi).ToList();
                var (fa, sc) = WmCalibrator.FalseAlarms(mine, isaac[pi], e => WmCalibrator.Scores(e, isaac[pi], zero, one), t, horizon);
                t.FarVerify = t.FarCalib = fa / (double)Math.Max(1, sc);
                t.ScoredA = t.ScoredB = sc;
                isaacT[names[pi]] = t;
            }
            var reference = Evaluate("Isaac thresholds (yaml, no Unity correction)", zero, one, isaacT);

            var variants = new List<Variant>();
            foreach (ZCorrection mode in Enum.GetValues(typeof(ZCorrection)))
            {
                if (mode == ZCorrection.IsaacStd && isaacStd == null)
                    continue;
                var (bias, scale) = WmCalibrator.Correction(clean, mode, isaac[0], horizon, isaacStd);
                variants.Add(Evaluate(Label(mode), bias, scale, WmCalibrator.Thresholds(clean, isaac, bias, scale, horizon)));
            }
            var chosen = variants.Where(v => v.Eligible).OrderByDescending(v => v.KeyDetect2s).FirstOrDefault()
                         ?? variants.OrderBy(v => v.Thresholds.Values.Max(t => t.FarVerify)).First();

            // runtime calibration
            var cal = new UnityWmCalibration
            {
                source = "Unity " + Application.unityVersion + ", BMB/World model/Calibrate + detect events (Editor/WorldModelEval.cs); " +
                         $"clean: {co.EnvsPerPolicy} envs per policy x {co.Seconds:0} s, seed {co.Seed}; {UnityEval.SolverName} solver; " +
                         DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                correction = chosen.Name,
                z_bias = chosen.Bias,
                z_scale = chosen.Scale,
                policies = names.Select(n => new PolicyCusum
                {
                    policy = n, m0 = chosen.Thresholds[n].M0, s0 = chosen.Thresholds[n].S0, k = chosen.Thresholds[n].K,
                    h = chosen.Thresholds[n].H, spike_z = chosen.Thresholds[n].SpikeZ,
                }).ToArray(),
            };
            Directory.CreateDirectory(Path.GetDirectoryName(RepoPaths.WmCalibrationUnity));
            File.WriteAllText(RepoPaths.WmCalibrationUnity, JsonUtility.ToJson(cal, true));
            Directory.CreateDirectory(OutputDir);
            File.WriteAllText(Path.Combine(OutputDir, "calibration.md"), CalibrationReport(clean, co, wallSeconds, names, isaac, horizon, layout, isaacTerms, reference, variants, chosen));
            File.WriteAllText(Path.Combine(OutputDir, "detection.md"), DetectionReport(events, eo, names, reference, variants, chosen));
            return $"chosen correction: {chosen.Name} (key events within 2 s {chosen.KeyDetect2s:P0}; Isaac thresholds {reference.KeyDetect2s:P0}); " +
                   string.Join(", ", variants.Select(v => $"{v.Name} {v.KeyDetect2s:P0}{(v.Eligible ? "" : " (over budget)")}"));
        }

        /// <summary>The tile env <paramref name="env"/> spawned on (same layout and order as <see cref="Run"/>).</summary>
        public static GroundFeature SpawnTile(WmEvalOptions o, int env)
        {
            var tiles = UnityEval.TileLayout(o.TilesPerSide, o.Seed, o.Mix, o.Difficulty, o.FlatShare);
            return tiles[UnityEval.TileOrder(tiles.Count, o.Seed)[env % tiles.Count]];
        }

        static string TerrainKey(WmEvalOptions o, int env)
        {
            var t = SpawnTile(o, env);
            return UnityEval.IsFlat(t) ? "flat" : $"{t.tile} {(t.difficulty < 0.5f ? "easy" : "hard")}";
        }

        static string Label(ZCorrection mode) => mode switch
        {
            ZCorrection.Mad => "MAD (handoff §11.4)",
            ZCorrection.None => "none",
            _ => "Isaac z std",
        };

        static string CalibrationReport(List<WmEpisode> clean, WmEvalOptions o, double wallSeconds, string[] names, List<SurprisalParams> isaac,
            int horizon, WorldModelLayout layout, Dictionary<string, (double std, double w1, double w2)> isaacTerms,
            Variant reference, List<Variant> variants, Variant chosen)
        {
            var sb = new StringBuilder();
            var perMillion = 1e6 / isaac[0].TargetArlSteps;
            sb.AppendLine("# World-model surprisal in Unity physics: calibration on clean walking");
            sb.AppendLine();
            sb.AppendLine($"Generated by `BMB/World model/…` ({DateTime.Now:yyyy-MM-dd HH:mm}; both splits took {wallSeconds / 60:F1} min). Clean split: " +
                          $"{o.EnvsPerPolicy} envs per policy, {o.Seconds:0} s each, {o.EpisodeSeconds:0} s episodes, seed {o.Seed}, " +
                          $"{o.TerrainText}, foot μ {o.FootMu.x}–{o.FootMu.y} per episode, " +
                          $"payload / COM / gain randomization, no pushes, {UnityEval.SolverName}. Half A (even envs) calibrates, half B verifies; " +
                          $"alarms within {horizon} steps before a fall are not false alarms. Target: {perMillion:0.0} false alarms per 10⁶ scored steps " +
                          $"(1 per {isaac[0].TargetArlSteps:0}).");
            sb.AppendLine();
            sb.AppendLine("**Clean data**");
            sb.AppendLine();
            sb.AppendLine("| policy | episodes | falls | scored steps, half A / B |");
            sb.AppendLine("|---|---:|---:|---:|");
            for (var pi = 0; pi < names.Length; pi++)
            {
                var mine = clean.Where(e => e.Policy == pi).ToList();
                var t = chosen.Thresholds[names[pi]];
                sb.AppendLine($"| {names[pi]} | {mine.Count} | {mine.Count(e => e.Fell)} | {t.ScoredA} / {t.ScoredB} |");
            }
            sb.AppendLine();
            sb.AppendLine("**Calibration variants** (false alarms per 10⁶ scored steps on half A / half B; key events = mean detection within 2 s " +
                          "over μ → 0.1, μ → 0.05, one foot μ → 0.05, pushes 0.5 / 1.0 m/s, both policies, from `detection.md`)");
            sb.AppendLine();
            sb.Append("| z correction |");
            foreach (var n in names)
                sb.Append($" {n} m0 | s0 | h | spike_z | false alarms |");
            sb.AppendLine(" key events ≤ 2 s |");
            sb.Append("|---|");
            foreach (var _ in names)
                sb.Append("---:|---:|---:|---:|---:|");
            sb.AppendLine("---:|");
            foreach (var v in new[] { reference }.Concat(variants))
            {
                sb.Append($"| {(v == chosen ? "**" + v.Name + "** (used)" : v.Name)} |");
                foreach (var n in names)
                {
                    var t = v.Thresholds[n];
                    var fa = v == reference ? $"{t.FarVerify * 1e6:0} (all)" : $"{t.FarCalib * 1e6:0} / {t.FarVerify * 1e6:0}";
                    sb.Append($" {t.M0:0.###} | {t.S0:0.###} | {t.H:0.0} | {t.SpikeZ:0.0} | {fa} |");
                }
                sb.AppendLine($" {v.KeyDetect2s * 100:0} % |");
            }
            sb.AppendLine();
            sb.AppendLine("**False alarms by spawn tile** (per 10⁶ scored steps, all clean data; easy = difficulty < 0.5)");
            sb.AppendLine();
            sb.Append("| tile |");
            foreach (var n in names)
                sb.Append($" {n} episodes | falls | Isaac thresholds | {chosen.Name} |");
            sb.AppendLine();
            sb.Append("|---|");
            foreach (var _ in names)
                sb.Append("---:|---:|---:|---:|");
            sb.AppendLine();
            foreach (var group in clean.GroupBy(e => TerrainKey(o, e.Env)).OrderBy(g => g.Key))
            {
                sb.Append($"| {group.Key} |");
                for (var pi = 0; pi < names.Length; pi++)
                {
                    var mine = group.Where(e => e.Policy == pi).ToList();
                    string Rate(Variant v)
                    {
                        var (fa, sc) = WmCalibrator.FalseAlarms(mine, isaac[pi], e => WmCalibrator.Scores(e, isaac[pi], v.Bias, v.Scale), v.Thresholds[names[pi]], horizon);
                        return $"{fa * 1e6 / Math.Max(1, sc):0}";
                    }
                    sb.Append($" {mine.Count} | {mine.Count(e => e.Fell)} | {Rate(reference)} | {Rate(chosen)} |");
                }
                sb.AppendLine();
            }
            sb.AppendLine();
            sb.AppendLine("**Raw z on clean walking, per state term** (Isaac: `experiments/w5_surprisal/results.json` one_step_clean; " +
                          "Unity: scored steps not before a fall, both policies)");
            sb.AppendLine();
            sb.AppendLine("| term | Isaac z std | Unity z std | Isaac \\|z\\| < 1 | Unity | Isaac \\|z\\| < 2 | Unity | MAD scale (mean) |");
            sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|---:|");
            var mad = variants.FirstOrDefault(v => v.Name == Label(ZCorrection.Mad));
            var offset = 0;
            foreach (var (term, std, w1, w2) in WmCalibrator.TermStats(clean, isaac[0], layout, horizon))
            {
                var dims = layout.FrameTermDims[Array.IndexOf(layout.FrameTerms, term)];
                var has = isaacTerms.TryGetValue(term, out var it);
                string F(double x) => has ? x.ToString("0.00", CultureInfo.InvariantCulture) : "–";
                string P(double x) => has ? (x * 100).ToString("0.0", CultureInfo.InvariantCulture) + " %" : "–";
                sb.AppendLine($"| {term} | {F(it.std)} | {std:0.00} | {P(it.w1)} | {w1 * 100:0.0} % | {P(it.w2)} | {w2 * 100:0.0} % | " +
                              $"{(mad != null ? mad.Scale.Skip(offset).Take(dims).Average().ToString("0.00", CultureInfo.InvariantCulture) : "–")} |");
                offset += dims;
            }
            sb.AppendLine();
            sb.AppendLine($"Used correction ({chosen.Name}), per dim in state_terms order:");
            sb.AppendLine();
            sb.AppendLine("```");
            sb.AppendLine("z_bias  " + string.Join(" ", chosen.Bias.Select(v => v.ToString("+0.00;-0.00", CultureInfo.InvariantCulture))));
            sb.AppendLine("z_scale " + string.Join(" ", chosen.Scale.Select(v => v.ToString("0.00", CultureInfo.InvariantCulture))));
            sb.AppendLine("```");
            return sb.ToString();
        }

        static string DetectionReport(List<WmEpisode> events, WmEvalOptions o, string[] names, Variant reference, List<Variant> variants, Variant chosen)
        {
            var sb = new StringBuilder();
            sb.AppendLine("# World-model surprisal in Unity physics: event detection (Unity vs Isaac)");
            sb.AppendLine();
            sb.AppendLine($"Events split ({DateTime.Now:yyyy-MM-dd HH:mm}): {o.EnvsPerPolicy} envs per policy, one {o.EpisodeSeconds:0} s episode each, " +
                          $"event i % 7 at step {o.EventStep}; {o.TerrainText}, foot μ " +
                          $"{o.FootMu.x}–{o.FootMu.y} before the event, seed {o.Seed}, {UnityEval.SolverName}. Detection = first alarm within 1 / 2 s " +
                          "after the event (%); delay in policy steps (50 = 1 s). Isaac: experiments/w5_surprisal/results.json. Unity is scored from the " +
                          $"same rollouts with the yaml's Isaac thresholds (I-thr) and with Unity's calibration ({chosen.Name}).");
            sb.AppendLine();
            for (var pi = 0; pi < names.Length; pi++)
            {
                var n = names[pi];
                var isaac = ReadIsaacDetection(n);
                var a = reference.Detection[n];
                var b = chosen.Detection[n];
                string P(double v) => double.IsNaN(v) ? "–" : $"{v * 100:0}";
                string D(double v) => double.IsNaN(v) ? "–" : $"{v:0}";
                sb.AppendLine($"**{n}**");
                sb.AppendLine();
                sb.AppendLine("| event | envs Isaac / Unity | ≤ 1 s Isaac | Unity I-thr | Unity cal | ≤ 2 s Isaac | Unity I-thr | Unity cal | delay Isaac | Unity I-thr | Unity cal |");
                sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
                foreach (var ev in WmEvents.Names)
                {
                    var has = isaac.TryGetValue(ev, out var i);
                    sb.AppendLine($"| {ev} | {(has ? i.Envs.ToString() : "–")} / {a[ev].Envs} | " +
                                  $"{(has ? P(i.Detect1s) : "–")} | {P(a[ev].Detect1s)} | {P(b[ev].Detect1s)} | " +
                                  $"{(has ? P(i.Detect2s) : "–")} | {P(a[ev].Detect2s)} | {P(b[ev].Detect2s)} | " +
                                  $"{(has ? D(i.MedianDelay) : "–")} | {D(a[ev].MedianDelay)} | {D(b[ev].MedianDelay)} |");
                }
                sb.AppendLine();
                var mine = events.Where(e => e.Policy == pi).ToList();
                sb.AppendLine($"{n}: {mine.Count(e => e.FellBeforeEvent)} robots fell before their event (left out), {mine.Count(e => e.Fell && !e.FellBeforeEvent)} after it.");
                sb.AppendLine();
            }
            sb.AppendLine("**All calibration variants**, detection within 2 s (%), per policy:");
            sb.AppendLine();
            sb.Append("| z correction |");
            foreach (var n in names)
                foreach (var ev in WmEvents.Names)
                    sb.Append($" {n} {ev} |");
            sb.AppendLine();
            sb.Append("|---|");
            foreach (var _ in names)
                foreach (var __ in WmEvents.Names)
                    sb.Append("---:|");
            sb.AppendLine();
            foreach (var v in new[] { reference }.Concat(variants))
            {
                sb.Append($"| {v.Name}{(v.Eligible || v == reference ? "" : " (over budget)")} |");
                foreach (var n in names)
                    foreach (var ev in WmEvents.Names)
                        sb.Append($" {v.Detection[n][ev].Detect2s * 100:0} |");
                sb.AppendLine();
            }
            return sb.ToString();
        }

        /// <summary>Isaac's one_step_clean per-term z std and coverage (a targeted scan of results.json).</summary>
        static Dictionary<string, (double std, double w1, double w2)> ReadIsaacTerms()
        {
            var result = new Dictionary<string, (double, double, double)>();
            if (!File.Exists(IsaacResults))
                return result;
            var text = File.ReadAllText(IsaacResults);
            var block = text.IndexOf("\"one_step_clean\"", StringComparison.Ordinal);
            if (block < 0)
                return result;
            var all = text.IndexOf("\"all\"", block, StringComparison.Ordinal);
            foreach (var term in new[] { "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel" })
            {
                var at = text.IndexOf($"\"{term}\"", all, StringComparison.Ordinal);
                if (at < 0)
                    continue;
                double Field(string name)
                {
                    var f = text.IndexOf($"\"{name}\"", at, StringComparison.Ordinal);
                    var colon = text.IndexOf(':', f);
                    var end = text.IndexOfAny(new[] { ',', '}' }, colon);
                    return double.Parse(text.Substring(colon + 1, end - colon - 1).Trim(), CultureInfo.InvariantCulture);
                }
                result[term] = (Field("z_std"), Field("cov_1sigma"), Field("cov_2sigma"));
            }
            return result;
        }

        // --- background job + menus ----------------------------------------------------------------------------

        static IEnumerator<float> job;
        static int progressId = -1;
        static Stopwatch jobClock;

        static string CleanLog => Path.Combine(LogDir, "clean.bin");
        static string EventsLog => Path.Combine(LogDir, "events.bin");

        /// <param name="runEvents">false: reuse the saved events split (its settings do not depend on the clean split).</param>
        static IEnumerator<float> CalibrateAndDetectJob(bool runEvents = true)
        {
            var co = WmEvalOptions.Clean();
            var eo = WmEvalOptions.EventsSplit();
            var w = runEvents ? co.EnvsPerPolicy * co.Seconds / (co.EnvsPerPolicy * co.Seconds + eo.EnvsPerPolicy * eo.Seconds) : 1f;
            var clean = new List<WmEpisode>();
            using (var it = Run(co, clean))
                while (it.MoveNext())
                {
                    Report(w * it.Current, "clean walking");
                    yield return w * it.Current;
                }
            SaveEpisodes(CleanLog, clean);
            var events = new List<WmEpisode>();
            if (runEvents || !File.Exists(EventsLog))
            {
                using (var it = Run(eo, events))
                    while (it.MoveNext())
                    {
                        Report(w + (1 - w) * it.Current, "events");
                        yield return w + (1 - w) * it.Current;
                    }
                SaveEpisodes(EventsLog, events);
            }
            else
                events = LoadEpisodes(EventsLog);
            var summary = Analyze(clean, events, co, eo, jobClock.Elapsed.TotalSeconds);
            Debug.Log($"[BMB] world-model calibration written ({jobClock.Elapsed.TotalMinutes:F1} min): {RepoPaths.WmCalibrationUnity}, " +
                      $"experiments/unity_wm/calibration.md, detection.md; {summary}");
        }

        static void Report(float progress, string what)
        {
            if (progressId >= 0)
                Progress.Report(progressId, progress, what);
        }

        public static bool Running => job != null;

        public static void Start(IEnumerator<float> work, string title)
        {
            if (job != null || UnityEval.Running)
            {
                Debug.LogWarning("[BMB] an evaluation is already running");
                return;
            }
            job = work;
            jobClock = Stopwatch.StartNew();
            progressId = Progress.Start(title);
            Progress.RegisterCancelCallback(progressId, () => { Stop("cancelled"); return true; });
            EditorApplication.update += Tick;
            AssemblyReloadEvents.beforeAssemblyReload += OnReload;
            Debug.Log($"[BMB] {title} started");
        }

        static void Tick()
        {
            if (job == null)
                return;
            var budget = Stopwatch.StartNew();
            try
            {
                while (budget.ElapsedMilliseconds < 300)
                    if (!job.MoveNext())
                    {
                        Stop(null);
                        return;
                    }
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                Stop("failed");
            }
        }

        static void OnReload() => Stop("interrupted by a script reload");

        static void Stop(string reason)
        {
            EditorApplication.update -= Tick;
            AssemblyReloadEvents.beforeAssemblyReload -= OnReload;
            job?.Dispose();
            job = null;
            if (progressId >= 0 && Progress.Exists(progressId))
                Progress.Finish(progressId, reason == null ? Progress.Status.Succeeded : reason == "cancelled" ? Progress.Status.Canceled : Progress.Status.Failed);
            progressId = -1;
            if (reason != null)
                Debug.LogWarning($"[BMB] world-model job {reason} after {jobClock?.Elapsed.TotalMinutes:F1} min");
        }

        [MenuItem("BMB/World model/Calibrate + detect events (v1 + v2, ~15 min)")]
        public static void CalibrateMenu() => Start(CalibrateAndDetectJob(), "BMB world-model calibration + detection");

        [MenuItem("BMB/World model/Recalibrate (clean split only, reuse saved events)")]
        public static void RecalibrateMenu() => Start(CalibrateAndDetectJob(runEvents: false), "BMB world-model recalibration");

        [MenuItem("BMB/World model/Re-analyse saved runs")]
        public static void ReanalyseMenu()
        {
            if (!File.Exists(CleanLog) || !File.Exists(EventsLog))
            {
                Debug.LogWarning($"[BMB] no saved runs in {LogDir}; run Calibrate + detect events first");
                return;
            }
            var clock = Stopwatch.StartNew();
            var summary = Analyze(LoadEpisodes(CleanLog), LoadEpisodes(EventsLog), WmEvalOptions.Clean(), WmEvalOptions.EventsSplit(), double.NaN);
            Debug.Log($"[BMB] world-model re-analysis ({clock.Elapsed.TotalSeconds:F0} s): {summary}");
        }

        [MenuItem("BMB/World model/Cancel")]
        static void CancelMenu() => Stop("cancelled");

        [MenuItem("BMB/World model/Cancel", true)]
        static bool CancelValidate() => Running;
    }
}
