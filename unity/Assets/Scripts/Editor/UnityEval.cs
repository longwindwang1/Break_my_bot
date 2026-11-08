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
    /// <summary>One evaluation scenario, as in isaac/eval_policy.py SCENARIOS.</summary>
    public sealed class EvalScenario
    {
        public string Name;
        public string Description;
        /// <summary>Foot μ for the whole episode (terrain μ 1.0).</summary>
        public float FootMu = 1f;
        /// <summary>> 0: every foot drops to this μ a quarter into the episode (mu_drop).</summary>
        public float DropToMu;
        /// <summary>Velocity pushes of up to ±1.0 m/s per axis every 3–5 s.</summary>
        public bool Push;
        /// <summary>Pit tiles only, difficulty 0.5–1.0 (otherwise the v2 mix at 0–0.5).</summary>
        public bool Pits;
        /// <summary>The height scan reports flat ground for the whole episode.</summary>
        public bool Deceive;
        /// <summary>Commands capped to policies.v2.cautious_gait.</summary>
        public bool Cautious;

        public static readonly EvalScenario[] Isaac =
        {
            new EvalScenario { Name = "mu_1.0", Description = "constant foot friction 1.0", FootMu = 1.0f },
            new EvalScenario { Name = "mu_0.5", Description = "constant foot friction 0.5 (wet)", FootMu = 0.5f },
            new EvalScenario { Name = "mu_0.2", Description = "constant foot friction 0.2", FootMu = 0.2f },
            new EvalScenario { Name = "mu_0.1", Description = "constant foot friction 0.1 (ice)", FootMu = 0.1f },
            new EvalScenario { Name = "mu_0.05", Description = "constant foot friction 0.05 (polished ice)", FootMu = 0.05f },
            new EvalScenario { Name = "mu_drop", Description = "friction 1.0 -> 0.1 on all feet at t = 5 s", FootMu = 1.0f, DropToMu = 0.1f },
            new EvalScenario { Name = "push", Description = "friction 1.0, velocity pushes of up to 1.0 m/s every 3-5 s", Push = true },
            new EvalScenario { Name = "pits_clean", Description = "pit terrain only, height scan truthful", Pits = true },
            new EvalScenario { Name = "pits_deceived", Description = "pit terrain only, height scan reports flat ground", Pits = true, Deceive = true },
            new EvalScenario { Name = "mu_0.1_cautious", Description = "friction 0.1, command capped to the cautious gait", FootMu = 0.1f, Cautious = true },
        };

        public static EvalScenario Find(string name) =>
            Isaac.FirstOrDefault(s => s.Name == name) ?? throw new ArgumentException($"unknown scenario {name}");
    }

    /// <summary>A policy under test: the Unity variant and the row name of the matching Isaac result.</summary>
    public sealed class EvalPolicy
    {
        public string Name;
        public string Variant;
        public string IsaacName;

        public static readonly EvalPolicy V1 = new EvalPolicy { Name = "v1", Variant = "rough", IsaacName = "v1" };
        public static readonly EvalPolicy V2 = new EvalPolicy { Name = "v2", Variant = "v2", IsaacName = "v2_student" };
    }

    public sealed class EvalOptions
    {
        public int EnvsPerPolicy = 64;
        public float EpisodeSeconds = 20f;
        public int Seed;
        /// <summary>Isaac tiles per side (8 m each); Isaac's eval uses 10.</summary>
        public int TilesPerSide = 6;
        /// <summary>Flat border around the tiles (Isaac: 20 m).</summary>
        public float Border = 20f;
        /// <summary>Go2V2EnvCfg_PLAY keeps payload / COM (startup) and actuator-gain (reset) randomization.</summary>
        public bool RandomizeDynamics = true;
        public EvalPolicy[] Policies = { EvalPolicy.V1, EvalPolicy.V2 };
    }

    /// <summary>One row of results.jsonl (same metric names as experiments/v2_eval/results.jsonl).</summary>
    public sealed class EvalRow
    {
        public string Scenario, Policy, Variant, Model;
        public int Envs, Steps, Seed;
        public float SuccessRate, TrackingError, SlipSpeed;
        /// <summary>Mean time of the falls (s); NaN without falls.</summary>
        public float MeanFallTime;
        public float WallSeconds;
        /// <summary>PhysX solver of the run (project setting): "PGS" or "TGS" (Isaac Lab uses TGS).</summary>
        public string Solver = "PGS";

        public string ToJson()
        {
            string F(float v) => float.IsNaN(v) ? "null" : v.ToString("0.#####", CultureInfo.InvariantCulture);
            return "{" +
                   $"\"scenario\": \"{Scenario}\", \"policy\": \"{Policy}\", \"variant\": \"{Variant}\", \"model\": \"{Model}\", \"sim\": \"unity\", " +
                   $"\"envs\": {Envs}, \"success_rate\": {F(SuccessRate)}, \"tracking_error\": {F(TrackingError)}, \"slip_speed\": {F(SlipSpeed)}, " +
                   $"\"mean_fall_time\": {F(MeanFallTime)}, \"solver\": \"{Solver}\", \"seed\": {Seed}, \"steps\": {Steps}, \"wall_seconds\": {F(WallSeconds)}" +
                   "}";
        }
    }

    /// <summary>
    /// isaac/eval_policy.py in Unity physics (Experiment 3, transfer): the same scenarios, metrics and randomization,
    /// all policies in one headless world (env e of every policy gets the same tile, spawn, payload, gains and
    /// command stream, so the comparison is paired). Robots never collide with each other.
    /// - Terrain: TilesPerSide² Isaac tiles (8 m) of the v2 mix at difficulty 0–0.5 (pits only at 0.5–1.0), flat border.
    /// - Reset: tile centre ± 0.5 m, random yaw, default joints, zero velocity.
    /// - Commands: UniformVelocityCommand, heading mode, resampled every 10 s (CommandSampler).
    /// - Episode: 20 s; success = no base contact (the only termination counted, like Isaac's eval).
    /// - Tracking error ‖v_xy(base frame) − cmd_xy‖ and slip (mean horizontal speed of feet touching the ground)
    ///   averaged over alive policy steps. Foot contact is a 3 mm overlap test instead of Isaac's force > 1 N.
    /// Menu: BMB/Eval/… runs in the background (EditorApplication.update, cancellable from the progress bar) and
    /// writes experiments/unity_eval/{results.jsonl, summary.md}.
    /// </summary>
    public static class UnityEval
    {
        public static string OutputDir => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "experiments", "unity_eval"));
        public static string IsaacResults => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "experiments", "v2_eval", "results.jsonl"));

        const float PushInterval0 = 3f, PushInterval1 = 5f, PushMax = 1f;

        sealed class Env
        {
            public Go2Controller Robot;
            public int Index, Policy;
            public GroundFeature Tile;
            public float Payload;
            public CommandSampler Commands;
            public System.Random PushRng;
            public float PushTimer;
            public bool Alive = true;
            public double TrackSum, SlipSum;
            public int AliveSteps;
            public float FallTime = float.NaN;
        }

        /// <summary>
        /// Runs one scenario; yields the progress (0–1) after every policy step so a caller can spread the work over
        /// editor frames. The rows (one per policy) are added to <paramref name="rows"/> at the end, and one CSV line
        /// per episode (<see cref="EpisodeHeader"/>) to <paramref name="episodes"/> if given.
        /// </summary>
        public static IEnumerator<float> Run(EvalScenario scenario, EvalOptions o, List<EvalRow> rows, List<string> episodes = null)
        {
            var clock = Stopwatch.StartNew();
            var world = HeadlessWorld.Create(flatGround: false);
            var runners = new List<PolicyRunner>();
            try
            {
                var itf = world.Interface;
                var level = BuildArena(world, scenario, o, out var tiles);
                foreach (var p in o.Policies)
                    runners.Add(PolicyRunner.Load(p.Variant, itf));
                var envs = SpawnRobots(world, scenario, o, tiles, runners);

                var terrainMask = LayerMask.GetMask(SimSettings.TerrainLayer);
                var decimation = itf.Timing.Decimation;
                var dt = itf.Timing.PhysicsDt * decimation;
                var steps = Mathf.RoundToInt(o.EpisodeSeconds / dt);
                var dropStep = steps / 4;  // eval_policy.py: step == steps // 4
                for (var k = 0; k < steps; k++)
                {
                    foreach (var e in envs)
                    {
                        if (!e.Alive)
                            continue;
                        var body = e.Robot.Body;
                        if (scenario.DropToMu > 0f && k == dropStep)
                            e.Robot.FootFriction.SetFootMu(scenario.DropToMu);
                        e.Robot.CommandOverride = e.Commands.Next(dt, CommandSampler.Heading(body.BaseRotation));
                        if (scenario.Push && k > 0)
                        {
                            e.PushTimer -= dt;
                            if (e.PushTimer <= 0f)
                            {
                                var dv = new Vector3(Uniform(e.PushRng, -PushMax, PushMax), Uniform(e.PushRng, -PushMax, PushMax), 0f);
                                body.Root.linearVelocity += IsaacFrame.PolarFromIsaac(dv);  // push_by_setting_velocity
                                e.PushTimer = Uniform(e.PushRng, PushInterval0, PushInterval1);
                            }
                        }
                    }
                    for (var i = 0; i < decimation; i++)
                        world.Step();

                    var time = (k + 1) * dt;
                    foreach (var e in envs)
                    {
                        if (!e.Alive)
                            continue;
                        var body = e.Robot.Body;
                        var vb = IsaacFrame.PolarToIsaac(Quaternion.Inverse(body.BaseRotation) * body.BaseComLinearVelocity);
                        var cmd = e.Robot.Command;
                        e.TrackSum += new Vector2(vb.x - cmd.x, vb.y - cmd.y).magnitude;
                        float speedSum = 0f;
                        var touching = 0;
                        for (var f = 0; f < Go2Body.FootNames.Length; f++)
                        {
                            if (!body.FootTouches(f, world.Physics, terrainMask))
                                continue;
                            var v = body.FootVelocity(f);
                            speedSum += new Vector2(v.x, v.z).magnitude;
                            touching++;
                        }
                        e.SlipSum += touching > 0 ? speedSum / touching : 0f;
                        e.AliveSteps++;
                        if (body.BaseTouches(world.Physics, terrainMask) || body.BasePosition.y < -5f)
                        {
                            e.Alive = false;
                            e.FallTime = time;
                            world.Retire(e.Robot);
                        }
                    }
                    yield return (k + 1f) / steps;
                }

                for (var p = 0; p < o.Policies.Length; p++)
                {
                    var mine = envs.Where(e => e.Policy == p).ToList();
                    var aliveSteps = mine.Sum(e => e.AliveSteps);
                    var falls = mine.Where(e => !e.Alive).ToList();
                    rows.Add(new EvalRow
                    {
                        Scenario = scenario.Name,
                        Policy = o.Policies[p].Name,
                        Variant = o.Policies[p].Variant,
                        Model = runners[p].Name,
                        Envs = mine.Count,
                        SuccessRate = mine.Count(e => e.Alive) / (float)mine.Count,
                        TrackingError = (float)(mine.Sum(e => e.TrackSum) / Math.Max(1, aliveSteps)),
                        SlipSpeed = (float)(mine.Sum(e => e.SlipSum) / Math.Max(1, aliveSteps)),
                        MeanFallTime = falls.Count > 0 ? falls.Average(e => e.FallTime) : float.NaN,
                        Seed = o.Seed,
                        Steps = steps,
                        WallSeconds = (float)clock.Elapsed.TotalSeconds,
                        Solver = SolverName,
                    });
                }
                if (episodes != null)
                    foreach (var e in envs)
                        episodes.Add(string.Join(",", scenario.Name, o.Policies[e.Policy].Name, e.Index.ToString(), e.Tile.tile.ToString(),
                            e.Tile.difficulty.ToString("0.000", CultureInfo.InvariantCulture), e.Payload.ToString("0.00", CultureInfo.InvariantCulture),
                            e.Alive ? "1" : "0", float.IsNaN(e.FallTime) ? "" : e.FallTime.ToString("0.00", CultureInfo.InvariantCulture),
                            (e.TrackSum / Math.Max(1, e.AliveSteps)).ToString("0.000", CultureInfo.InvariantCulture),
                            (e.SlipSum / Math.Max(1, e.AliveSteps)).ToString("0.000", CultureInfo.InvariantCulture)));
            }
            finally
            {
                world.Dispose();
                foreach (var r in runners)
                    r.Dispose();
            }
        }

        /// <summary>Runs scenarios to completion (tests, scripts); blocks the editor.</summary>
        public static List<EvalRow> RunSync(IEnumerable<EvalScenario> scenarios, EvalOptions o)
        {
            var rows = new List<EvalRow>();
            foreach (var s in scenarios)
            {
                var it = Run(s, o, rows);
                while (it.MoveNext()) { }
            }
            return rows;
        }

        static LevelGround BuildArena(HeadlessWorld world, EvalScenario s, EvalOptions o, out List<GroundFeature> tiles)
        {
            var n = o.TilesPerSide;
            var side = n * IsaacTiles.Size + 2f * o.Border;
            var level = world.AddLevelGround(new Vector2(side, side));
            var rng = new System.Random(o.Seed);  // same terrain for every scenario of a kind, like Isaac's seed 0
            var mix = s.Pits ? new[] { (IsaacTileType.Pits, 1f) } : IsaacTiles.V2Mix;
            var difficulty = s.Pits ? new Vector2(0.5f, 1f) : new Vector2(0f, 0.5f);
            tiles = new List<GroundFeature>();
            for (var r = 0; r < n; r++)
            for (var c = 0; c < n; c++)
            {
                var centre = new Vector2((c - (n - 1) / 2f) * IsaacTiles.Size, (r - (n - 1) / 2f) * IsaacTiles.Size);
                var f = new GroundFeature
                {
                    type = GroundFeatureType.IsaacTile,
                    tile = IsaacTiles.Sample(rng, mix),
                    difficulty = Uniform(rng, difficulty.x, difficulty.y),
                    seed = rng.Next(),
                    centre = centre,
                    size = new Vector2(IsaacTiles.Size, IsaacTiles.Size),
                };
                level.Add(f, refresh: false);
                tiles.Add(f);
            }
            level.Refresh();
            Physics.SyncTransforms();
            return level;
        }

        static List<Env> SpawnRobots(HeadlessWorld world, EvalScenario s, EvalOptions o, List<GroundFeature> tiles, List<PolicyRunner> runners)
        {
            var itf = world.Interface;
            var cautious = itf.Policy("v2").CautiousGait;
            var order = Enumerable.Range(0, tiles.Count).ToArray();
            var shuffle = new System.Random(o.Seed + 7);
            for (var i = order.Length - 1; i > 0; i--)
            {
                var j = shuffle.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }
            var envs = new List<Env>();
            var n = itf.NumJoints;
            for (var e = 0; e < o.EnvsPerPolicy; e++)
            {
                // Everything random about env e is drawn once and shared by all policies (paired comparison).
                var rng = new System.Random(unchecked(o.Seed * 7919 + e * 104729 + 13));
                var tile = tiles[order[e % order.Length]];
                var tileCentre = new Vector3(tile.centre.x, 0f, tile.centre.y);  // LevelGround sits at the origin
                var offset = IsaacFrame.PolarFromIsaac(new Vector3(Uniform(rng, -0.5f, 0.5f), Uniform(rng, -0.5f, 0.5f), 0f));
                var yaw = Uniform(rng, -180f, 180f);
                var mass = Uniform(rng, -1f, 3f);
                var com = new Vector3(Uniform(rng, -0.05f, 0.05f), Uniform(rng, -0.05f, 0.05f), Uniform(rng, -0.02f, 0.02f));
                var kp = new float[n];
                var kd = new float[n];
                for (var j = 0; j < n; j++)
                {
                    kp[j] = Uniform(rng, 0.9f, 1.1f);
                    kd[j] = Uniform(rng, 0.9f, 1.1f);
                }
                var commandSeed = rng.Next();
                var pushSeed = rng.Next();

                for (var p = 0; p < o.Policies.Length; p++)
                {
                    var robot = world.AddRobot(o.Policies[p].Variant, tileCentre + offset, yaw, footFriction: s.FootMu,
                                               resetOnFall: false, sharedPolicy: runners[p]);
                    robot.SetFallCriteria(0f, 0f);  // base contact only
                    if (o.RandomizeDynamics)
                    {
                        robot.Body.SetPayload(mass, com);
                        robot.SetGainScales(kp, kd);
                    }
                    if (s.Deceive && robot.Scanner != null)
                        robot.Scanner.Deceive = true;
                    var sampler = new CommandSampler(itf.Commands, new System.Random(commandSeed));
                    if (s.Cautious)
                        sampler.Cap(cautious.MaxLinVel, cautious.MaxAngVel);
                    var pushRng = new System.Random(pushSeed);
                    envs.Add(new Env
                    {
                        Robot = robot,
                        Index = e,
                        Policy = p,
                        Tile = tile,
                        Payload = o.RandomizeDynamics ? mass : 0f,
                        Commands = sampler,
                        PushRng = pushRng,
                        PushTimer = Uniform(pushRng, PushInterval0, PushInterval1),
                    });
                }
            }
            return envs;
        }

        static float Uniform(System.Random rng, float lo, float hi) => lo + (float)rng.NextDouble() * (hi - lo);

        // --- output ---------------------------------------------------------------------------------------------

        public static void AppendResults(string path, IEnumerable<EvalRow> rows)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.AppendAllLines(path, rows.Select(r => r.ToJson()));
        }

        [Serializable]
        sealed class JsonRow
        {
            public string scenario, policy;
            public int envs;
            public float success_rate, tracking_error, slip_speed;
        }

        static Dictionary<(string, string), JsonRow> ReadRows(string path)
        {
            var rows = new Dictionary<(string, string), JsonRow>();
            if (!File.Exists(path))
                return rows;
            foreach (var line in File.ReadAllLines(path))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                var r = JsonUtility.FromJson<JsonRow>(line);
                rows[(r.scenario, r.policy)] = r;  // later rows win
            }
            return rows;
        }

        /// <summary>Markdown tables, Isaac (experiments/v2_eval) next to Unity, for each metric.</summary>
        public static string Summary(string unityPath, string isaacPath, EvalPolicy[] policies)
        {
            var unity = ReadRows(unityPath);
            var isaac = ReadRows(isaacPath);
            var sb = new StringBuilder();
            var envs = unity.Values.Select(r => r.envs).DefaultIfEmpty(0).Max();
            var isaacEnvs = isaac.Values.Select(r => r.envs).DefaultIfEmpty(0).Max();
            sb.AppendLine($"Isaac: experiments/v2_eval/results.jsonl ({isaacEnvs} envs per policy). " +
                          $"Unity: experiments/unity_eval/results.jsonl ({envs} envs per policy). Δ = Unity − Isaac.");
            sb.AppendLine();
            void Table(string title, Func<JsonRow, float> metric, bool percent)
            {
                sb.AppendLine($"**{title}**");
                sb.AppendLine();
                sb.Append("| scenario |");
                foreach (var p in policies)
                    sb.Append($" {p.Name} Isaac | {p.Name} Unity | Δ |");
                sb.AppendLine();
                sb.Append("|---|");
                foreach (var _ in policies)
                    sb.Append("---:|---:|---:|");
                sb.AppendLine();
                foreach (var s in EvalScenario.Isaac)
                {
                    if (!policies.Any(p => unity.ContainsKey((s.Name, p.Name))))
                        continue;
                    sb.Append($"| {s.Name} |");
                    foreach (var p in policies)
                    {
                        float? a = isaac.TryGetValue((s.Name, p.IsaacName), out var ir) ? metric(ir) : null;
                        float? b = unity.TryGetValue((s.Name, p.Name), out var ur) ? metric(ur) : null;
                        string Fmt(float? v) => v == null ? "–" : percent ? $"{v * 100:F1} %" : $"{v:F3}";
                        var d = a != null && b != null ? (percent ? $"{(b - a) * 100:+0.0;-0.0;0.0}" : $"{b - a:+0.000;-0.000;0.000}") : "–";
                        sb.Append($" {Fmt(a)} | {Fmt(b)} | {d} |");
                    }
                    sb.AppendLine();
                }
                sb.AppendLine();
            }
            Table("Success rate", r => r.success_rate, true);
            Table("Tracking error (m/s)", r => r.tracking_error, false);
            Table("Foot slip in contact (m/s)", r => r.slip_speed, false);
            return sb.ToString();
        }

        /// <summary>summary{suffix}.md from results{suffix}.jsonl (suffix "" for the main run, e.g. "_tgs" for a variant).</summary>
        public static void WriteSummary(EvalPolicy[] policies = null, string suffix = "")
        {
            policies ??= new[] { EvalPolicy.V1, EvalPolicy.V2 };
            var md = Summary(Path.Combine(OutputDir, $"results{suffix}.jsonl"), IsaacResults, policies);
            File.WriteAllText(Path.Combine(OutputDir, $"summary{suffix}.md"), "# Unity vs Isaac: v2_eval scenarios\n\n" + md);
        }

        public const string EpisodeHeader = "scenario,policy,env,tile,difficulty,payload_kg,success,fall_time_s,tracking_error,slip_speed";

        // --- PhysX solver (project setting) ----------------------------------------------------------------------

        static SerializedObject DynamicsSettings() =>
            new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/DynamicsManager.asset")[0]);

        /// <summary>"PGS" or "TGS": the solver new physics scenes are created with (Isaac Lab's default is TGS).</summary>
        public static string SolverName => DynamicsSettings().FindProperty("m_SolverType").intValue == 1 ? "TGS" : "PGS";

        /// <summary>Switches the project's PhysX solver (Project Settings > Physics > Solver Type).</summary>
        public static void SetSolver(bool tgs)
        {
            var so = DynamicsSettings();
            so.FindProperty("m_SolverType").intValue = tgs ? 1 : 0;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        // --- background job + menus --------------------------------------------------------------------------------

        static IEnumerator<float> job;
        static int progressId = -1;
        static Stopwatch jobClock;

        static IEnumerator<float> RunAll(IReadOnlyList<EvalScenario> scenarios, EvalOptions o, string suffix)
        {
            var path = Path.Combine(OutputDir, $"results{suffix}.jsonl");
            var episodesPath = Path.Combine(OutputDir, $"episodes{suffix}.csv");
            var total = Stopwatch.StartNew();
            for (var i = 0; i < scenarios.Count; i++)
            {
                var rows = new List<EvalRow>();
                var episodes = new List<string>();
                using var it = Run(scenarios[i], o, rows, episodes);  // disposing the job disposes this: the world closes
                while (it.MoveNext())
                {
                    if (progressId >= 0)
                        Progress.Report(progressId, (i + it.Current) / scenarios.Count, $"{scenarios[i].Name} ({i + 1}/{scenarios.Count})");
                    yield return (i + it.Current) / scenarios.Count;
                }
                AppendResults(path, rows);
                File.AppendAllLines(episodesPath, episodes);
                foreach (var r in rows)
                    Debug.Log($"[BMB] eval {r.Scenario,-16} {r.Policy,-3} success {r.SuccessRate:P1}  track {r.TrackingError:F3}  slip {r.SlipSpeed:F3}  ({r.WallSeconds:F0} s)");
                WriteSummary(o.Policies, suffix);
            }
            Debug.Log($"[BMB] eval done in {total.Elapsed.TotalMinutes:F1} min: {Path.Combine(OutputDir, $"summary{suffix}.md")}");
        }

        /// <summary>
        /// Starts a background evaluation writing results{suffix}.jsonl, episodes{suffix}.csv and summary{suffix}.md
        /// in experiments/unity_eval (rewritten from scratch).
        /// </summary>
        public static void Start(IReadOnlyList<EvalScenario> scenarios, EvalOptions o, string suffix = "")
        {
            if (job != null)
            {
                Debug.LogWarning("[BMB] an evaluation is already running (BMB/Eval/Cancel)");
                return;
            }
            Directory.CreateDirectory(OutputDir);
            File.WriteAllText(Path.Combine(OutputDir, $"results{suffix}.jsonl"), "");
            File.WriteAllText(Path.Combine(OutputDir, $"episodes{suffix}.csv"), EpisodeHeader + "\n");
            job = RunAll(scenarios, o, suffix);
            jobClock = Stopwatch.StartNew();
            progressId = Progress.Start("BMB evaluation", $"{scenarios.Count} scenarios × {o.Policies.Length} policies × {o.EnvsPerPolicy} envs");
            Progress.RegisterCancelCallback(progressId, () => { Stop("cancelled"); return true; });
            EditorApplication.update += Tick;
            AssemblyReloadEvents.beforeAssemblyReload += OnReload;
            Debug.Log($"[BMB] eval started ({SolverName} solver): {string.Join(", ", scenarios.Select(s => s.Name))}; {o.EnvsPerPolicy} envs per policy, {o.EpisodeSeconds} s");
        }

        public static bool Running => job != null;

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
            job?.Dispose();  // runs the finally blocks: closes the preview scene, frees the workers
            job = null;
            if (progressId >= 0 && Progress.Exists(progressId))
                Progress.Finish(progressId, reason == null ? Progress.Status.Succeeded : reason == "cancelled" ? Progress.Status.Canceled : Progress.Status.Failed);
            progressId = -1;
            if (reason != null)
                Debug.LogWarning($"[BMB] eval {reason} after {jobClock?.Elapsed.TotalMinutes:F1} min");
        }

        [MenuItem("BMB/Eval/Run Isaac scenarios (v1 + v2, 64 envs)")]
        static void RunAllMenu() => Start(EvalScenario.Isaac, new EvalOptions());

        [MenuItem("BMB/Eval/Quick check (mu_1.0 + mu_0.1, 8 envs)")]
        static void QuickMenu() => Start(new[] { EvalScenario.Find("mu_1.0"), EvalScenario.Find("mu_0.1") }, new EvalOptions { EnvsPerPolicy = 8 });

        [MenuItem("BMB/Eval/Cancel")]
        static void CancelMenu() => Stop("cancelled");

        [MenuItem("BMB/Eval/Cancel", true)]
        static bool CancelValidate() => Running;

        [MenuItem("BMB/Eval/Rewrite summary.md")]
        static void SummaryMenu() => WriteSummary();
    }
}
