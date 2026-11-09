using System;
using System.Collections.Generic;
using System.Linq;

namespace BMB
{
    /// <summary>
    /// robot_interface.yaml → world_model: the learned dynamics monitor (W4–5). Python reader:
    /// models/world_model/layout.py; Unity-side description: docs/unity_handoff.md §11.
    /// </summary>
    public sealed class WorldModelSpec
    {
        /// <summary>Order of the policy one-hot: v1 = W1 rough baseline, v2 = deployed v2 student.</summary>
        public List<string> Policies { get; set; }
        /// <summary>Observation terms that form the predicted state s (33-d), concatenated.</summary>
        public List<string> StateTerms { get; set; }
        /// <summary>wm_in = these blocks concatenated: history, action, height_scan, policy_onehot.</summary>
        public List<WmInputBlock> Input { get; set; }
        public int InputDim { get; set; }
        public Dictionary<string, WmOutputSpec> Outputs { get; set; }
        public int[] FallHorizonsSteps { get; set; }
        public SurprisalSpec Surprisal { get; set; }

        public WmInputBlock Block(string name) =>
            Input.FirstOrDefault(b => b.Name == name) ?? throw new ArgumentException($"no world_model.input block {name}");

        public int OutputDim(string name) =>
            Outputs.TryGetValue(name, out var o) ? o.Dim : throw new ArgumentException($"no world_model.outputs.{name}");

        /// <summary>
        /// World-model policy name of a Go2 policy variant: the yaml's "v1" is the W1 rough baseline (variant
        /// "rough"); "v2" is v2. Flat has no height scan and no world model: null.
        /// </summary>
        public static string PolicyName(string variant) => variant switch
        {
            "v2" => "v2",
            "rough" => "v1",
            _ => null,
        };

        /// <summary>Inverse of <see cref="PolicyName"/>.</summary>
        public static string Variant(string policy) => policy == "v1" ? "rough" : policy;
    }

    /// <summary>One block of wm_in. The history block lists terms and a length; the others have a dim.</summary>
    public sealed class WmInputBlock
    {
        public string Name { get; set; }
        public int Dim { get; set; }
        public int HistoryLength { get; set; }
        public List<string> Terms { get; set; }
    }

    public sealed class WmOutputSpec
    {
        public int Dim { get; set; }
        public string Def { get; set; }
    }

    /// <summary>world_model.surprisal: per-dim z-score, statistic S, CUSUM and spike detector (W5 calibration).</summary>
    public sealed class SurprisalSpec
    {
        /// <summary>"norm": S = Σ w·½(min(z², clip) − 1); "nll": S = Σ w·½(min(z², clip) + log 2π(var + floor)).</summary>
        public string Statistic { get; set; }
        public float ClipZ2 { get; set; }
        public string WeightsScheme { get; set; }
        public float[] Weights { get; set; }
        public float[] VarFloor { get; set; }
        /// <summary>Engine corrections (identity for Isaac Lab); Unity's own live in wm_calibration_unity.json.</summary>
        public float[] ZBias { get; set; }
        public float[] ZScale { get; set; }
        public int WarmupSteps { get; set; }
        public int[] SpikeDims { get; set; }
        public float TargetArlSteps { get; set; }
        public Dictionary<string, CusumSpec> PerPolicy { get; set; }
    }

    public sealed class CusumSpec
    {
        public float M0 { get; set; }
        public float S0 { get; set; }
        public float K { get; set; }
        public float H { get; set; }
        public float SpikeZ { get; set; }
    }
}
