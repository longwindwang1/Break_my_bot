using System;
using System.IO;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Runs an exported locomotion policy (export/export_onnx.py: input "obs" [1, obs_dim], output "actions"
    /// [1, num_joints], raw and unclipped) on the CPU backend, one observation per call.
    /// Models are loaded from Resources/BMB/Models (copied there by BMB/Sync models).
    /// </summary>
    public sealed class PolicyRunner : IDisposable
    {
        public const string ResourcesDir = "BMB/Models";

        readonly Worker worker;

        public string Name { get; }
        public int ObsDim { get; }
        public int ActionDim { get; }

        /// <summary>Loads go2_&lt;variant&gt; and checks it against policies.&lt;variant&gt; of the interface.</summary>
        public static PolicyRunner Load(string variant, RobotInterface itf)
        {
            var name = $"go2_{variant}";
            var asset = Resources.Load<ModelAsset>($"{ResourcesDir}/{name}");
            if (asset == null)
                throw new FileNotFoundException($"Resources/{ResourcesDir}/{name}.onnx not found; run BMB/Sync models");
            return new PolicyRunner(asset, itf.ObsDim(variant), itf.NumJoints, name);
        }

        public PolicyRunner(ModelAsset asset, int obsDim, int actionDim, string name = null)
        {
            Name = name ?? asset.name;
            ObsDim = obsDim;
            ActionDim = actionDim;

            var model = ModelLoader.Load(asset);
            if (model.inputs.Count != 1 || model.outputs.Count != 1)
                throw new InvalidDataException($"{Name}: expected 1 input and 1 output, got {model.inputs.Count} and {model.outputs.Count}");
            var input = model.inputs[0];
            if (input.name != "obs" || !input.shape.IsStatic() || !input.shape.ToIntArray().SequenceEqual(new[] { 1, obsDim }))
                throw new InvalidDataException($"{Name}: input is {input.name} {input.shape}, expected obs (1, {obsDim})");
            if (model.outputs[0].name != "actions")
                throw new InvalidDataException($"{Name}: output is {model.outputs[0].name}, expected actions");

            worker = new Worker(model, BackendType.CPU);
        }

        /// <summary>actionsOut = policy(obs). Both arrays are reused by the caller.</summary>
        public void Run(float[] obs, float[] actionsOut)
        {
            if (obs.Length != ObsDim)
                throw new ArgumentException($"{Name}: obs has {obs.Length} values, expected {ObsDim}");
            if (actionsOut.Length != ActionDim)
                throw new ArgumentException($"{Name}: actionsOut has {actionsOut.Length} slots, expected {ActionDim}");

            using var input = new Tensor<float>(new TensorShape(1, ObsDim), obs);
            worker.Schedule(input);
            using var output = (worker.PeekOutput() as Tensor<float>).ReadbackAndClone();
            var shape = output.shape.ToArray();
            if (!shape.SequenceEqual(new[] { 1, ActionDim }))
                throw new InvalidDataException($"{Name}: output shape ({string.Join(", ", shape)}), expected (1, {ActionDim})");
            Array.Copy(output.DownloadToArray(), actionsOut, ActionDim);
        }

        public void Dispose() => worker?.Dispose();
    }
}
