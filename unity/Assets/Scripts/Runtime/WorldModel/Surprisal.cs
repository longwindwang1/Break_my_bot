using System;
using System.IO;
using System.Linq;
using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Surprisal + change-detection parameters for one policy: the yaml's world_model.surprisal (calibrated in Isaac
    /// Lab), with z_bias / z_scale and the per-policy CUSUM values replaced by Unity's own calibration when
    /// experiments/w3_unity/wm_calibration_unity.json exists (docs/unity_handoff.md §11.4).
    /// </summary>
    public sealed class SurprisalParams
    {
        public string Policy;
        /// <summary>"isaac" (robot_interface.yaml) or "unity" (wm_calibration_unity.json).</summary>
        public string Source;
        public bool Nll;
        public float ClipZ2;
        public float[] Weights, VarFloor, ZBias, ZScale;
        public int WarmupSteps;
        public int[] SpikeDims;
        public float TargetArlSteps;
        public float M0, S0, K, H, SpikeZ;

        /// <summary>The yaml values (Isaac Lab calibration).</summary>
        public static SurprisalParams Isaac(RobotInterface itf, string policy)
        {
            var sp = itf.WorldModel?.Surprisal ?? throw new InvalidOperationException("robot_interface.yaml has no world_model.surprisal");
            if (!sp.PerPolicy.TryGetValue(policy, out var c))
                throw new ArgumentException($"no world_model.surprisal.per_policy.{policy}");
            var nll = sp.Statistic switch
            {
                "norm" => false,
                "nll" => true,
                _ => throw new InvalidDataException($"unknown surprisal statistic {sp.Statistic}"),
            };
            var n = itf.WorldModel.OutputDim("state_mean");
            foreach (var (name, a) in new[] { ("weights", sp.Weights), ("var_floor", sp.VarFloor), ("z_bias", sp.ZBias), ("z_scale", sp.ZScale) })
                if (a == null || a.Length != n)
                    throw new InvalidDataException($"world_model.surprisal.{name} needs {n} values");
            return new SurprisalParams
            {
                Policy = policy, Source = "isaac", Nll = nll, ClipZ2 = sp.ClipZ2,
                Weights = sp.Weights, VarFloor = sp.VarFloor, ZBias = sp.ZBias, ZScale = sp.ZScale,
                WarmupSteps = sp.WarmupSteps, SpikeDims = sp.SpikeDims, TargetArlSteps = sp.TargetArlSteps,
                M0 = c.M0, S0 = c.S0, K = c.K, H = c.H, SpikeZ = c.SpikeZ,
            };
        }

        /// <summary>Unity's calibration if present (and preferred), else the yaml's.</summary>
        public static SurprisalParams Load(RobotInterface itf, string policy, bool preferUnity = true, string unityPath = null)
        {
            var p = Isaac(itf, policy);
            if (!preferUnity)
                return p;
            var cal = UnityWmCalibration.TryLoad(unityPath ?? RepoPaths.WmCalibrationUnity);
            return cal == null ? p : p.With(cal);
        }

        /// <summary>A copy with a Unity calibration's z correction and this policy's CUSUM values.</summary>
        public SurprisalParams With(UnityWmCalibration cal)
        {
            var c = cal.policies?.FirstOrDefault(x => x.policy == Policy)
                    ?? throw new InvalidDataException($"Unity world-model calibration has no policy {Policy}");
            if (cal.z_bias?.Length != ZBias.Length || cal.z_scale?.Length != ZScale.Length)
                throw new InvalidDataException($"Unity world-model calibration needs {ZBias.Length} z_bias / z_scale values");
            var p = (SurprisalParams)MemberwiseClone();
            p.Source = "unity";
            p.ZBias = cal.z_bias;
            p.ZScale = cal.z_scale;
            (p.M0, p.S0, p.K, p.H, p.SpikeZ) = (c.m0, c.s0, c.k, c.h, c.spike_z);
            return p;
        }
    }

#pragma warning disable IDE1006, CS0649
    /// <summary>experiments/w3_unity/wm_calibration_unity.json (written by BMB/World model/Calibrate).</summary>
    [Serializable]
    public sealed class UnityWmCalibration
    {
        public string source;
        public float[] z_bias, z_scale;
        public PolicyCusum[] policies;

        public static UnityWmCalibration TryLoad(string path) =>
            File.Exists(path) ? JsonUtility.FromJson<UnityWmCalibration>(File.ReadAllText(path)) : null;
    }

    [Serializable]
    public sealed class PolicyCusum
    {
        public string policy;
        public float m0, s0, k, h, spike_z;
    }
#pragma warning restore IDE1006, CS0649

    /// <summary>
    /// Scores the observed state s_t against the prediction made at t−1 and runs the CUSUM + spike detector
    /// (port of models/world_model/surprisal.py and export_wm_onnx.surprisal_trace, in double precision):
    /// <code>
    /// z_d = ((s_d − mean_d) / sqrt(var_d + floor_d) − z_bias_d) / z_scale_d
    /// S   = Σ_d w_d · ½(min(z_d², clip) − 1)          ("nll": ½(min(z², clip) + log 2π(var + floor)))
    /// x   = (S − m0) / s0;   g = max(0, g + x − k)    (only on scored steps)
    /// spike = max_{d ∈ spike_dims} |z_d| > spike_z;   alarm = scored ∧ (g > h ∨ spike), then g = 0
    /// </code>
    /// g = 0 at every episode start; a step is scored when a prediction exists and episode_step ≥ warmup_steps.
    /// </summary>
    public sealed class SurprisalDetector
    {
        public SurprisalParams Params { get; private set; }

        /// <summary>Per-dim z of the last step (0 without a prediction).</summary>
        public readonly double[] Z;
        public double S { get; private set; }
        /// <summary>Standardized score (S − m0) / s0.</summary>
        public double X { get; private set; }
        public double SpikeStat { get; private set; }
        /// <summary>CUSUM statistic after this step (0 right after an alarm).</summary>
        public double G { get; private set; }
        /// <summary>G before an alarm reset it (for display).</summary>
        public double GPeak { get; private set; }
        /// <summary>A prediction from the previous step existed.</summary>
        public bool Valid { get; private set; }
        public bool Scored { get; private set; }
        public bool Spike { get; private set; }
        public bool CusumAlarm { get; private set; }
        public bool Alarm { get; private set; }
        public int Alarms { get; private set; }
        public int ScoredSteps { get; private set; }

        public SurprisalDetector(SurprisalParams p)
        {
            Params = p;
            Z = new double[p.Weights.Length];
        }

        /// <summary>New thresholds; the CUSUM restarts.</summary>
        public void SetParams(SurprisalParams p)
        {
            Params = p;
            G = 0;
        }

        public void ResetCounters() => (Alarms, ScoredSteps) = (0, 0);

        /// <param name="s">Observed state s_t (state_terms order).</param>
        /// <param name="mean">state_mean predicted at t−1, or null (episode start).</param>
        /// <param name="var">state_var predicted at t−1, or null.</param>
        /// <param name="episodeStep">Policy steps since the episode started (0 = first step).</param>
        public void Step(float[] s, float[] mean, float[] var, int episodeStep)
        {
            var p = Params;
            if (episodeStep == 0)
                G = 0;
            Valid = mean != null && var != null;
            S = 0;
            SpikeStat = 0;
            if (Valid)
            {
                for (var d = 0; d < Z.Length; d++)
                {
                    var v = (double)var[d] + p.VarFloor[d];
                    var z = (((double)s[d] - mean[d]) / Math.Sqrt(v) - p.ZBias[d]) / p.ZScale[d];
                    Z[d] = z;
                    var z2 = Math.Min(z * z, p.ClipZ2);
                    S += p.Weights[d] * 0.5 * (p.Nll ? z2 + Math.Log(2 * Math.PI * v) : z2 - 1.0);
                }
                foreach (var d in p.SpikeDims)
                    SpikeStat = Math.Max(SpikeStat, Math.Abs(Z[d]));
            }
            else
                Array.Clear(Z, 0, Z.Length);
            X = (S - p.M0) / p.S0;
            Scored = Valid && episodeStep >= p.WarmupSteps;
            Spike = Scored && SpikeStat > p.SpikeZ;
            if (Scored)
            {
                G = Math.Max(0.0, G + X - p.K);
                ScoredSteps++;
            }
            GPeak = G;
            CusumAlarm = Scored && G > p.H;
            Alarm = Scored && (CusumAlarm || Spike);
            if (Alarm)
            {
                G = 0;
                Alarms++;
            }
        }
    }
}
