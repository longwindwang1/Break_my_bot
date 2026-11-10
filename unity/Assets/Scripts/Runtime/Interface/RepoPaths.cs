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

        /// <summary>
        /// Surprisal thresholds re-estimated in Unity physics (docs/unity_handoff.md §11.4); preferred over the yaml's
        /// Isaac values when present. Player builds read a copy in StreamingAssets.
        /// </summary>
        public static string WmCalibrationUnity =>
            Application.isEditor
                ? Path.Combine(RepoRoot, "experiments", "w3_unity", "wm_calibration_unity.json")
                : Path.Combine(Application.streamingAssetsPath, "wm_calibration_unity.json");

        /// <summary>One JSON line per AI-pilot challenge (Claude vs the player); builds write to persistentDataPath.</summary>
        public static string PilotResults =>
            Application.isEditor
                ? Path.Combine(RepoRoot, "experiments", "claude_pilot", "results.jsonl")
                : Path.Combine(Application.persistentDataPath, "claude_pilot_results.jsonl");
    }
}
