using System;
using System.IO;
using UnityEngine;

namespace BMB
{
#pragma warning disable IDE1006, CS0649
    /// <summary>
    /// World-model reference trace (export/export_wm_onnx.py --reference; docs/unity_handoff.md §11.3): every step of a
    /// v2 reference rollout with wm_in, the six ONNX outputs, and z, S, x, scored, spike, g, alarm from the yaml's
    /// surprisal parameters. go2_wm.json is the clean rollout of go2_v2.json; go2_wm_mudrop.json the rollout of
    /// go2_v2_mudrop.json, whose foot μ drops to 0.05 at friction_drop.at_step.
    /// </summary>
    [Serializable]
    public sealed class WmReference
    {
        public string source, model, policy;
        public int steps;
        public FrictionDrop friction_drop;
        public NdArray wm_in, state_mean, state_var, disp_mean, disp_var, fall_prob, disagreement;
        public NdArray episode_step, z, S, x, scored, spike, g, alarm;

        /// <summary>JsonUtility turns a JSON null into a default object: at_step 0 means no drop.</summary>
        public bool HasFrictionDrop => friction_drop != null && friction_drop.at_step > 0;

        /// <summary>name: "wm" or "wm_mudrop" (experiments/w3_reference/go2_&lt;name&gt;.json).</summary>
        public static WmReference Load(string name)
        {
            var path = RepoPaths.ReferenceJson(name);
            if (!File.Exists(path))
                throw new FileNotFoundException($"world-model reference {path} not found (written on the Linux machine; git pull)");
            return JsonUtility.FromJson<WmReference>(File.ReadAllText(path));
        }
    }

    [Serializable]
    public sealed class FrictionDrop
    {
        public int at_step;
        public float to;
    }
#pragma warning restore IDE1006, CS0649
}
