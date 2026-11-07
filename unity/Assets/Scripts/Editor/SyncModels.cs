using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace BMB.Editor
{
    /// <summary>
    /// Copies the deployable policies from the repo's export/models/ into Assets/Resources/BMB/Models/ (gitignored,
    /// so git keeps a single copy) and checks every imported model against configs/robot_interface.yaml
    /// (input "obs" [1, policies.&lt;variant&gt;.obs_dim], output "actions" [1, num_joints], runs on CPU).
    /// File names follow export/export_onnx.py: go2_&lt;variant&gt;.onnx.
    /// </summary>
    public static class SyncModels
    {
        public const string ModelsAssetDir = "Assets/Resources/" + PolicyRunner.ResourcesDir;

        [MenuItem("BMB/Sync models")]
        public static void Sync()
        {
            var src = RepoPaths.ExportedModels;
            var dst = Path.GetFullPath(Path.Combine(Application.dataPath, "..", ModelsAssetDir));
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
            var itf = RobotInterface.Load();
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
                try
                {
                    using var runner = new PolicyRunner(AssetDatabase.LoadAssetAtPath<ModelAsset>(path), itf.ObsDim(variant), itf.NumJoints);
                    var actions = new float[runner.ActionDim];
                    runner.Run(new float[runner.ObsDim], actions);  // proves the graph executes
                    if (actions.Any(v => float.IsNaN(v) || float.IsInfinity(v)))
                        throw new InvalidDataException("output on zero input is not finite");
                    Debug.Log($"[BMB] {path}: OK (obs [1,{runner.ObsDim}] -> actions [1,{runner.ActionDim}], runs on CPU)");
                }
                catch (Exception e)
                {
                    Debug.LogError($"[BMB] {path}: {e.Message}");
                    allOk = false;
                }
            }
            return allOk;
        }
    }
}
