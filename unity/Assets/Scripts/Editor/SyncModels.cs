using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using YamlDotNet.Serialization;

namespace BMB.Editor
{
    /// <summary>
    /// Copies the deployable policies from the repo's export/models/ into Assets/Resources/BMB/Models/ (gitignored,
    /// so git keeps a single copy) and checks every imported model against configs/robot_interface.yaml:
    /// one input "obs" [1, policies.&lt;variant&gt;.obs_dim], one output "actions" [1, len(joint_names)].
    /// File names follow export/export_onnx.py: go2_&lt;variant&gt;.onnx.
    /// </summary>
    public static class SyncModels
    {
        public const string ModelsAssetDir = "Assets/Resources/BMB/Models";

        static string RepoRoot => Path.GetFullPath(Path.Combine(Application.dataPath, "..", ".."));
        static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

        [MenuItem("BMB/Sync models")]
        public static void Sync()
        {
            var src = Path.Combine(RepoRoot, "export", "models");
            var dst = Path.Combine(ProjectRoot, ModelsAssetDir);
            Directory.CreateDirectory(dst);

            var copied = new List<string>();
            foreach (var file in Directory.GetFiles(src, "*.onnx"))
            {
                var target = Path.Combine(dst, Path.GetFileName(file));
                if (File.Exists(target) && File.ReadAllBytes(target).SequenceEqual(File.ReadAllBytes(file)))
                    continue;
                File.Copy(file, target, true);
                copied.Add(Path.GetFileName(file));
            }
            foreach (var stale in Directory.GetFiles(dst, "*.onnx").Where(f => !File.Exists(Path.Combine(src, Path.GetFileName(f)))))
                Debug.LogWarning($"[BMB] {Path.GetFileName(stale)} is in {ModelsAssetDir} but no longer in export/models/");

            AssetDatabase.Refresh();
            Debug.Log($"[BMB] Sync models: {(copied.Count == 0 ? "already up to date" : "updated " + string.Join(", ", copied))} (source {src})");
            Check();
        }

        [MenuItem("BMB/Check models")]
        public static bool Check()
        {
            var itf = new DeserializerBuilder().Build().Deserialize<Dictionary<string, object>>(
                File.ReadAllText(Path.Combine(RepoRoot, "configs", "robot_interface.yaml")));
            var policies = (Dictionary<object, object>)itf["policies"];
            var numJoints = ((List<object>)itf["joint_names"]).Count;

            var paths = AssetDatabase.FindAssets("t:ModelAsset", new[] { ModelsAssetDir }).Select(AssetDatabase.GUIDToAssetPath).ToList();
            if (paths.Count == 0)
            {
                Debug.LogError($"[BMB] No models in {ModelsAssetDir}; run BMB/Sync models");
                return false;
            }

            var allOk = true;
            foreach (var path in paths)
            {
                var variant = Path.GetFileNameWithoutExtension(path).Replace("go2_", "");
                if (!policies.TryGetValue(variant, out var spec))
                {
                    Debug.LogError($"[BMB] {path}: no policies.{variant} in robot_interface.yaml");
                    allOk = false;
                    continue;
                }
                var obsDim = int.Parse(((Dictionary<object, object>)spec)["obs_dim"].ToString());
                var error = CheckModel(AssetDatabase.LoadAssetAtPath<ModelAsset>(path), obsDim, numJoints);
                if (error == null)
                    Debug.Log($"[BMB] {path}: OK (obs [1,{obsDim}] -> actions [1,{numJoints}], runs on CPU)");
                else
                    Debug.LogError($"[BMB] {path}: {error}");
                allOk &= error == null;
            }
            return allOk;
        }

        /// <summary>Returns null if the model matches the interface, else a description of the mismatch.</summary>
        static string CheckModel(ModelAsset asset, int obsDim, int numJoints)
        {
            var model = ModelLoader.Load(asset);
            if (model.inputs.Count != 1 || model.outputs.Count != 1)
                return $"expected 1 input and 1 output, got {model.inputs.Count} and {model.outputs.Count}";
            var input = model.inputs[0];
            if (input.name != "obs" || !input.shape.IsStatic() || !input.shape.ToIntArray().SequenceEqual(new[] { 1, obsDim }))
                return $"input is {input.name} {input.shape}, expected obs (1, {obsDim})";
            if (model.outputs[0].name != "actions")
                return $"output is {model.outputs[0].name}, expected actions";

            // Run once on zeros: proves the graph executes and gives the output shape.
            using var worker = new Worker(model, BackendType.CPU);
            using var obs = new Tensor<float>(new TensorShape(1, obsDim), new float[obsDim]);
            worker.Schedule(obs);
            using var actions = (worker.PeekOutput() as Tensor<float>).ReadbackAndClone();
            var shape = actions.shape.ToArray();
            if (!shape.SequenceEqual(new[] { 1, numJoints }))
                return $"output shape ({string.Join(", ", shape)}), expected (1, {numJoints})";
            if (actions.DownloadToArray().Any(v => float.IsNaN(v) || float.IsInfinity(v)))
                return "output on zero input is not finite";
            return null;
        }
    }
}
