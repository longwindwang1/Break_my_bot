using System;
using System.IO;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace BMB
{
    /// <summary>One world-model prediction (robot_interface.yaml world_model.outputs), made at step t for t+1.</summary>
    public sealed class WorldModelPrediction
    {
        /// <summary>E[s_{t+1}], absolute, raw units (state_terms order).</summary>
        public readonly float[] StateMean;
        /// <summary>Var[s_{t+1}]: mean member variance + variance of the member means.</summary>
        public readonly float[] StateVar;
        /// <summary>Base (dx, dy, dz, dyaw) from t to t+1, root link, in the yaw-only frame at t (Isaac axes; m, rad).</summary>
        public readonly float[] DispMean;
        public readonly float[] DispVar;
        /// <summary>P(base contact within h steps), h = world_model.fall_horizons_steps.</summary>
        public readonly float[] FallProb;
        /// <summary>Epistemic part: mean over state dims of (variance of member means) / delta_std².</summary>
        public float Disagreement;

        public WorldModelPrediction(WorldModelSpec spec)
        {
            StateMean = new float[spec.OutputDim("state_mean")];
            StateVar = new float[spec.OutputDim("state_var")];
            DispMean = new float[spec.OutputDim("disp_mean")];
            DispVar = new float[spec.OutputDim("disp_var")];
            FallProb = new float[spec.OutputDim("fall_prob")];
        }

        public void CopyFrom(WorldModelPrediction o)
        {
            Array.Copy(o.StateMean, StateMean, StateMean.Length);
            Array.Copy(o.StateVar, StateVar, StateVar.Length);
            Array.Copy(o.DispMean, DispMean, DispMean.Length);
            Array.Copy(o.DispVar, DispVar, DispVar.Length);
            Array.Copy(o.FallProb, FallProb, FallProb.Length);
            Disagreement = o.Disagreement;
        }
    }

    /// <summary>
    /// Runs export/models/go2_wm.onnx (export/export_wm_onnx.py): input "wm_in" [1, input_dim] in raw units, six
    /// named outputs. One sample per call; models load from Resources/BMB/Models (BMB/Sync models).
    /// It never produces actions.
    /// </summary>
    public sealed class WorldModelRunner : IDisposable
    {
        public const string ModelName = "go2_wm";
        public static readonly string[] OutputNames = { "state_mean", "state_var", "disp_mean", "disp_var", "fall_prob", "disagreement" };

        readonly Worker worker;

        public int InputDim { get; }
        public BackendType Backend { get; }

        public static WorldModelRunner Load(RobotInterface itf, BackendType backend = BackendType.CPU)
        {
            var asset = Resources.Load<ModelAsset>($"{PolicyRunner.ResourcesDir}/{ModelName}");
            if (asset == null)
                throw new FileNotFoundException($"Resources/{PolicyRunner.ResourcesDir}/{ModelName}.onnx not found; run BMB/Sync models");
            return new WorldModelRunner(asset, itf.WorldModel ?? throw new InvalidOperationException("robot_interface.yaml has no world_model"), backend);
        }

        public WorldModelRunner(ModelAsset asset, WorldModelSpec spec, BackendType backend = BackendType.CPU)
        {
            InputDim = spec.InputDim;
            Backend = backend;
            var model = ModelLoader.Load(asset);
            if (model.inputs.Count != 1)
                throw new InvalidDataException($"{ModelName}: expected 1 input, got {model.inputs.Count}");
            var input = model.inputs[0];
            if (input.name != "wm_in" || !input.shape.IsStatic() || !input.shape.ToIntArray().SequenceEqual(new[] { 1, InputDim }))
                throw new InvalidDataException($"{ModelName}: input is {input.name} {input.shape}, expected wm_in (1, {InputDim})");
            var outputs = model.outputs.Select(o => o.name).ToArray();
            if (!outputs.SequenceEqual(OutputNames))
                throw new InvalidDataException($"{ModelName}: outputs [{string.Join(", ", outputs)}], expected [{string.Join(", ", OutputNames)}]");
            foreach (var name in OutputNames)
                spec.OutputDim(name);  // throws if the yaml lacks it
            worker = new Worker(model, backend);
        }

        /// <summary>Predicts from one wm_in row into <paramref name="p"/> (arrays reused).</summary>
        public void Run(float[] wmIn, WorldModelPrediction p)
        {
            if (wmIn.Length != InputDim)
                throw new ArgumentException($"wm_in has {wmIn.Length} values, expected {InputDim}");
            using var input = new Tensor<float>(new TensorShape(1, InputDim), wmIn);
            worker.Schedule(input);
            Read("state_mean", p.StateMean);
            Read("state_var", p.StateVar);
            Read("disp_mean", p.DispMean);
            Read("disp_var", p.DispVar);
            Read("fall_prob", p.FallProb);
            var d = new float[1];
            Read("disagreement", d);
            p.Disagreement = d[0];
        }

        void Read(string name, float[] dst)
        {
            using var output = (worker.PeekOutput(name) as Tensor<float>).ReadbackAndClone();
            if (output.shape.length != dst.Length)
                throw new InvalidDataException($"{ModelName}: {name} has shape {output.shape}, expected (1, {dst.Length})");
            Array.Copy(output.DownloadToArray(), dst, dst.Length);
        }

        public void Dispose() => worker?.Dispose();
    }
}
