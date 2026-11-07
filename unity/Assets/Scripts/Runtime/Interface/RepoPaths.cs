using System.IO;
using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Files the Unity project reads straight from the repo (editor only: the project lives in &lt;repo&gt;/unity).
    /// They stay where the Isaac side writes them, so git holds a single copy.
    /// </summary>
    public static class RepoPaths
    {
        public static string RepoRoot => Path.GetFullPath(Path.Combine(Application.dataPath, "..", ".."));
        public static string InterfaceYaml => Path.Combine(RepoRoot, "configs", "robot_interface.yaml");
        public static string ExportedModels => Path.Combine(RepoRoot, "export", "models");

        /// <summary>Isaac Lab reference rollout converted by export/npz_to_json.py (variant: flat / rough / v2).</summary>
        public static string ReferenceJson(string variant) =>
            Path.Combine(RepoRoot, "experiments", "w3_reference", $"go2_{variant}.json");
    }
}
