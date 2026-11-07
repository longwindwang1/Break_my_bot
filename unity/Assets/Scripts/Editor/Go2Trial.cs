using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BMB.Editor
{
    /// <summary>One headless run setup.</summary>
    public sealed class TrialSpec
    {
        public string Name;
        public string Variant;
        public bool RoughTerrain;
        /// <summary>Static foot μ; ≤ 0 keeps the prefab's v1 material (0.8 / 0.6).</summary>
        public float FootFriction;
        /// <summary>Isaac reference rollout to compare with (null: Unity only).</summary>
        public string Reference;
        public Vector3 Command = new Vector3(0.5f, 0f, 0f);
        public float Seconds = 10f;
        public int TerrainSeed;
    }

    /// <summary>
    /// A trajectory sampled at every policy step, in a common form for Unity runs and Isaac references.
    /// Base xy in the run's initial heading frame (x forward, y left); quaternions (w, x, y, z), Isaac axes.
    /// </summary>
    public sealed class Trajectory
    {
        public float Dt;
        public int Falls;
        public readonly List<Vector2> XY = new List<Vector2>();
        public readonly List<float> HeightAboveTerrain = new List<float>();
        public readonly List<Vector4> Quat = new List<Vector4>();
        public readonly List<float[]> JointPos = new List<float[]>();
        public readonly List<float[]> Torque = new List<float[]>();
        public readonly List<float[]> Actions = new List<float[]>();
        public int Steps => JointPos.Count;

        public static Trajectory FromReference(ReferenceRollout r)
        {
            var tr = new Trajectory { Dt = r.step_dt };
            float w = r.root_link_quat_w[0, 0], x = r.root_link_quat_w[0, 1], y = r.root_link_quat_w[0, 2], z = r.root_link_quat_w[0, 3];
            var yaw0 = Mathf.Atan2(2f * (w * z + x * y), 1f - 2f * (y * y + z * z));
            float c = Mathf.Cos(-yaw0), s = Mathf.Sin(-yaw0);
            var hasScan = r.scan_hits_z != null && !r.scan_hits_z.IsEmpty;
            var centre = hasScan ? r.scan_hits_z.RowSize / 2 : -1;  // ray at (0, 0) offset = the middle of the grid
            for (var t = 0; t < r.Steps; t++)
            {
                float dx = r.root_link_pos_w[t, 0] - r.root_link_pos_w[0, 0], dy = r.root_link_pos_w[t, 1] - r.root_link_pos_w[0, 1];
                tr.XY.Add(new Vector2(c * dx - s * dy, s * dx + c * dy));
                tr.HeightAboveTerrain.Add(r.root_link_pos_w[t, 2] - (hasScan ? r.scan_hits_z[t, centre] : 0f));
                tr.Quat.Add(new Vector4(r.root_link_quat_w[t, 0], r.root_link_quat_w[t, 1], r.root_link_quat_w[t, 2], r.root_link_quat_w[t, 3]));
                tr.JointPos.Add(r.joint_pos.Row(t));
                tr.Torque.Add(r.applied_torque.Row(t));
                tr.Actions.Add(r.actions.Row(t));
            }
            return tr;
        }
    }

    /// <summary>
    /// Runs the closed loop (verification level 4) without play mode: robot, ground (flat, or flat + a random rough
    /// patch) and controller in a preview scene with its own physics, stepped by hand (control step, then
    /// PhysicsScene.Simulate(physics_dt)). Deterministic, independent of frame rate and editor focus. Runs start like
    /// the Isaac reference rollouts (default pose, init_base_height above the terrain, zero velocity, fixed command),
    /// and the report compares gait statistics. Isaac's rough terrain is a different random sample of the same
    /// distribution, so on rough ground only statistics are comparable, not step-by-step trajectories.
    /// </summary>
    public static class Go2Trial
    {
        public static readonly TrialSpec[] Standard =
        {
            new TrialSpec { Name = "flat policy, flat ground", Variant = "flat", Reference = "flat" },
            new TrialSpec { Name = "rough policy, rough patch", Variant = "rough", RoughTerrain = true, Reference = "rough" },
            new TrialSpec { Name = "v2 policy, rough patch, μ 1.0", Variant = "v2", RoughTerrain = true, FootFriction = 1f, Reference = "v2" },
            new TrialSpec { Name = "v2 policy, flat ground, μ 1.0", Variant = "v2", FootFriction = 1f },
            new TrialSpec { Name = "v2 policy, flat ground, μ 0.1 (ice)", Variant = "v2", FootFriction = 0.1f },
        };

        const string ResultsDir = "experiments/w3_unity";

        [MenuItem("BMB/Headless trials/Run all and write report")]
        public static void RunAllAndReport()
        {
            var results = Standard.Select(s => (spec: s, unity: Run(s), isaac: s.Reference != null ? Trajectory.FromReference(ReferenceRollout.Load(s.Reference)) : null)).ToList();
            var md = Report(results);
            var path = Path.Combine(RepoPaths.RepoRoot, ResultsDir, "trials.md");
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, md);
            foreach (var (spec, unity, isaac) in results)
                WriteCsv(spec, unity, isaac);
            Debug.Log($"[BMB] Headless trials written to {path}\n{md}");
        }

        [MenuItem("BMB/Headless trials/flat")] static void RunFlat() => RunOne(Standard[0]);
        [MenuItem("BMB/Headless trials/rough")] static void RunRough() => RunOne(Standard[1]);
        [MenuItem("BMB/Headless trials/v2 (rough patch)")] static void RunV2() => RunOne(Standard[2]);

        static void RunOne(TrialSpec spec)
        {
            var unity = Run(spec);
            var isaac = spec.Reference != null ? Trajectory.FromReference(ReferenceRollout.Load(spec.Reference)) : null;
            Debug.Log($"[BMB] Trial '{spec.Name}'\n{Report(new List<(TrialSpec, Trajectory, Trajectory)> { (spec, unity, isaac) })}");
        }

        public static Trajectory Run(TrialSpec spec)
        {
            var itf = RobotInterface.Load();
            var scene = EditorSceneManager.NewPreviewScene();
            Go2Controller controller = null;
            try
            {
                var terrainLayer = LayerMask.NameToLayer(SimSettings.TerrainLayer);
                var groundFriction = W3Scenes.GroundFriction(itf);
                var ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
                SceneManager.MoveGameObjectToScene(ground, scene);
                ground.transform.localScale = new Vector3(200f, 1f, 200f);
                ground.transform.position = new Vector3(0f, -0.5f, 0f);
                ground.layer = terrainLayer;
                ground.GetComponent<Collider>().sharedMaterial = groundFriction;
                if (spec.RoughTerrain)
                {
                    var patch = new GameObject("RoughPatch", typeof(HeightfieldTerrain));
                    SceneManager.MoveGameObjectToScene(patch, scene);
                    patch.layer = terrainLayer;
                    patch.transform.position = new Vector3(0f, 0f, W3Scenes.RoughPatchCentreZ);
                    patch.GetComponent<MeshCollider>().sharedMaterial = groundFriction;
                    patch.GetComponent<HeightfieldTerrain>().Configure(W3Scenes.RoughPatchSize, spec.TerrainSeed);
                }

                var robot = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>(Go2Setup.PrefabPath), scene);
                robot.transform.position = new Vector3(0f, itf.InitBaseHeight + 0.1f, 0f);
                var body = robot.GetComponent<Go2Body>();
                body.ApplyRuntimeSettings();

                var go = new GameObject("Go2Controller");
                SceneManager.MoveGameObjectToScene(go, scene);
                controller = go.AddComponent<Go2Controller>();
                controller.Configure(body, spec.Variant, resetAfterFall: true, footFrictionMu: spec.FootFriction);
                controller.ExternallyStepped = true;
                controller.CommandOverride = spec.Command;
                Physics.SyncTransforms();  // colliders moved after creation; the spawn raycast must see them in place
                controller.Initialize();

                var tr = new Trajectory { Dt = itf.Timing.PolicyDt };
                var start = body.BasePosition;
                controller.PolicyStepped += c =>
                {
                    // Sampled at the policy step, i.e. from the state before action t (as in the reference).
                    var d = IsaacFrame.PolarToIsaac(c.Body.BasePosition - start);
                    tr.XY.Add(new Vector2(d.x, d.y));
                    tr.HeightAboveTerrain.Add(c.BaseHeight);
                    tr.Quat.Add(IsaacFrame.QuatToIsaac(c.Body.BaseRotation));
                    tr.JointPos.Add(c.JointPos.ToArray());
                    tr.Torque.Add(c.Torques.ToArray());
                    tr.Actions.Add(c.Actions.ToArray());
                };

                var physics = scene.GetPhysicsScene();
                var steps = Mathf.RoundToInt(spec.Seconds / itf.Timing.PhysicsDt);
                for (var i = 0; i < steps; i++)
                {
                    controller.ControlStep();
                    physics.Simulate(itf.Timing.PhysicsDt);
                }
                tr.Falls = controller.Falls;
                return tr;
            }
            finally
            {
                if (controller != null)
                    controller.Shutdown();  // OnDestroy does not run in edit mode
                EditorSceneManager.ClosePreviewScene(scene);
            }
        }

        // --- metrics --------------------------------------------------------------------------------------

        /// <summary>Gait statistics over the steady part of a run (from 20 % of its length).</summary>
        public sealed class Metrics
        {
            public int Falls;
            public float Speed, Lateral, YawDrift, Height, HeightStd, RollStd, PitchMean, PitchStd, Period, TorqueRms, TorqueMax, ActionRate;
            public float[] JointMean = new float[12], JointStd = new float[12];

            public static Metrics Of(Trajectory tr)
            {
                var n = tr.Steps;
                var from = n / 5;
                var span = (n - 1 - from) * tr.Dt;
                var idx = Enumerable.Range(from, n - from).ToList();
                var euler = tr.Quat.Select(Euler).ToList();
                var m = new Metrics
                {
                    Falls = tr.Falls,
                    Speed = (tr.XY[n - 1].x - tr.XY[from].x) / span,
                    Lateral = tr.XY[n - 1].y - tr.XY[from].y,
                    YawDrift = Mathf.DeltaAngle(euler[from].z, euler[n - 1].z),
                    Height = idx.Average(t => tr.HeightAboveTerrain[t]),
                    HeightStd = Std(idx.Select(t => tr.HeightAboveTerrain[t])),
                    RollStd = Std(idx.Select(t => euler[t].x)),
                    PitchMean = idx.Average(t => euler[t].y),
                    PitchStd = Std(idx.Select(t => euler[t].y)),
                    Period = GaitPeriod(idx.Select(t => tr.JointPos[t][8]).ToList(), tr.Dt),
                    TorqueRms = Mathf.Sqrt(idx.Average(t => tr.Torque[t].Average(v => v * v))),
                    TorqueMax = idx.Max(t => tr.Torque[t].Max(Mathf.Abs)),
                    ActionRate = idx.Skip(1).Average(t => tr.Actions[t].Zip(tr.Actions[t - 1], (a, b) => Mathf.Abs(a - b)).Average()),
                };
                for (var j = 0; j < 12; j++)
                {
                    m.JointMean[j] = idx.Average(t => tr.JointPos[t][j]);
                    m.JointStd[j] = Std(idx.Select(t => tr.JointPos[t][j]));
                }
                return m;
            }

            /// <summary>Roll, pitch, yaw (deg, ZYX) of an Isaac quaternion (w, x, y, z).</summary>
            static Vector3 Euler(Vector4 q)
            {
                float w = q.x, x = q.y, y = q.z, z = q.w;
                var roll = Mathf.Atan2(2f * (w * x + y * z), 1f - 2f * (x * x + y * y));
                var pitch = Mathf.Asin(Mathf.Clamp(2f * (w * y - z * x), -1f, 1f));
                var yaw = Mathf.Atan2(2f * (w * z + x * y), 1f - 2f * (y * y + z * z));
                return new Vector3(roll, pitch, yaw) * Mathf.Rad2Deg;
            }

            static float Std(IEnumerable<float> values)
            {
                var v = values.ToList();
                var mean = v.Average();
                return Mathf.Sqrt(v.Average(x => (x - mean) * (x - mean)));
            }

            /// <summary>Mean period between upward crossings of the mean (FL calf angle).</summary>
            static float GaitPeriod(List<float> x, float dt)
            {
                var mean = x.Average();
                var crossings = new List<int>();
                for (var t = 1; t < x.Count; t++)
                    if (x[t - 1] < mean && x[t] >= mean)
                        crossings.Add(t);
                return crossings.Count < 2 ? float.NaN : (crossings[crossings.Count - 1] - crossings[0]) * dt / (crossings.Count - 1);
            }
        }

        // --- report ---------------------------------------------------------------------------------------

        static string F(float v, string format) => float.IsNaN(v) ? "–" : v.ToString(format, CultureInfo.InvariantCulture);

        public static string Report(List<(TrialSpec spec, Trajectory unity, Trajectory isaac)> runs)
        {
            var itf = RobotInterface.Load();
            var sb = new StringBuilder();
            sb.AppendLine("# W3 headless trials: Unity physics vs Isaac Lab reference");
            sb.AppendLine();
            sb.AppendLine($"Generated by `BMB/Headless trials/Run all and write report` on {DateTime.Now:yyyy-MM-dd HH:mm}. Command (vx, vy, wz) = (0.5, 0, 0), " +
                          "10 s, start: default pose, init_base_height above the terrain, zero velocity. Statistics over the last 80 % of each run. " +
                          "Unity rough patch: random_rough-like heightfield (0.01–0.06 m on a 0.1 m grid), a different random sample than Isaac's. " +
                          "Base axes as in Isaac: x forward, y left; drift = displacement over the steady window.");
            sb.AppendLine();

            var metrics = runs.Select(r => (r.spec, u: Metrics.Of(r.unity), i: r.isaac != null ? Metrics.Of(r.isaac) : null)).ToList();
            sb.Append("| metric |");
            foreach (var (spec, _, _) in metrics) sb.Append($" {spec.Name} |");
            sb.AppendLine();
            sb.Append("|---|");
            foreach (var _ in metrics) sb.Append("---|");
            sb.AppendLine();

            void Row(string name, Func<Metrics, float> get, string format)
            {
                sb.Append($"| {name} |");
                foreach (var (_, u, i) in metrics)
                    sb.Append(i != null ? $" {F(get(u), format)} / {F(get(i), format)} |" : $" {F(get(u), format)} |");
                sb.AppendLine();
            }

            Row("falls (resets)", m => m.Falls, "0");
            Row("forward speed (m/s)", m => m.Speed, "0.000");
            Row("lateral drift (m)", m => m.Lateral, "0.00");
            Row("heading drift (deg)", m => m.YawDrift, "0.0");
            Row("base height above terrain (m)", m => m.Height, "0.000");
            Row("base height std (m)", m => m.HeightStd, "0.000");
            Row("pitch mean (deg)", m => m.PitchMean, "0.0");
            Row("pitch std (deg)", m => m.PitchStd, "0.0");
            Row("roll std (deg)", m => m.RollStd, "0.0");
            Row("gait period, FL calf (s)", m => m.Period, "0.000");
            Row("joint torque RMS (N·m)", m => m.TorqueRms, "0.00");
            Row("joint torque max |τ| (N·m)", m => m.TorqueMax, "0.0");
            Row("action rate, mean |Δa| per step", m => m.ActionRate, "0.000");
            sb.AppendLine();
            sb.AppendLine("Cells: Unity / Isaac where an Isaac reference exists, else Unity only.");
            sb.AppendLine();

            sb.AppendLine("## Joint angles (rad): mean ± std, Unity / Isaac");
            sb.AppendLine();
            sb.Append("| joint |");
            foreach (var (spec, _, _) in metrics) sb.Append($" {spec.Name} |");
            sb.AppendLine();
            sb.Append("|---|");
            foreach (var _ in metrics) sb.Append("---|");
            sb.AppendLine();
            for (var j = 0; j < 12; j++)
            {
                sb.Append($"| {itf.JointNames[j]} |");
                foreach (var (_, u, i) in metrics)
                {
                    var cell = $"{F(u.JointMean[j], "0.00")} ± {F(u.JointStd[j], "0.00")}";
                    if (i != null) cell += $" / {F(i.JointMean[j], "0.00")} ± {F(i.JointStd[j], "0.00")}";
                    sb.Append($" {cell} |");
                }
                sb.AppendLine();
            }

            // Same start on flat ground: how quickly do the two engines part ways?
            foreach (var (spec, unity, isaac) in runs.Where(r => r.isaac != null && !r.spec.RoughTerrain))
            {
                sb.AppendLine();
                sb.Append($"Divergence, {spec.Name} (identical start), max |q_unity − q_isaac| (rad) at policy step:");
                foreach (var t in new[] { 1, 2, 5, 10, 25, 50, 100 }.Where(t => t < Math.Min(unity.Steps, isaac.Steps)))
                    sb.Append($" {t}: {unity.JointPos[t].Zip(isaac.JointPos[t], (a, b) => Mathf.Abs(a - b)).Max():0.000};");
                sb.AppendLine();
            }
            return sb.ToString();
        }

        /// <summary>experiments/w3_unity/&lt;variant&gt;_&lt;terrain&gt;_trial.csv: per policy step, Unity (and Isaac) side by side.</summary>
        static void WriteCsv(TrialSpec spec, Trajectory u, Trajectory i)
        {
            var name = $"{spec.Variant}_{(spec.RoughTerrain ? "rough" : "flat")}_mu{(spec.FootFriction > 0 ? spec.FootFriction.ToString("0.0#", CultureInfo.InvariantCulture) : "v1")}";
            var path = Path.Combine(RepoPaths.RepoRoot, ResultsDir, $"{name}_trial.csv");
            var names = RobotInterface.Load().JointNames;
            var ci = CultureInfo.InvariantCulture;
            var sb = new StringBuilder("t,unity_x,unity_y,unity_height");
            if (i != null) sb.Append(",isaac_x,isaac_y,isaac_height");
            foreach (var n in names) sb.Append(i != null ? $",unity_{n},isaac_{n}" : $",unity_{n}");
            sb.AppendLine();
            var steps = i != null ? Math.Min(u.Steps, i.Steps) : u.Steps;
            for (var t = 0; t < steps; t++)
            {
                sb.Append(string.Format(ci, "{0:F3},{1:F5},{2:F5},{3:F5}", t * u.Dt, u.XY[t].x, u.XY[t].y, u.HeightAboveTerrain[t]));
                if (i != null) sb.Append(string.Format(ci, ",{0:F5},{1:F5},{2:F5}", i.XY[t].x, i.XY[t].y, i.HeightAboveTerrain[t]));
                for (var j = 0; j < names.Count; j++)
                    sb.Append(i != null ? string.Format(ci, ",{0:F5},{1:F5}", u.JointPos[t][j], i.JointPos[t][j]) : string.Format(ci, ",{0:F5}", u.JointPos[t][j]));
                sb.AppendLine();
            }
            File.WriteAllText(path, sb.ToString());
        }
    }
}
