using System;
using UnityEditor;
using UnityEngine;

namespace BMB.Editor
{
    /// <summary>
    /// Writes the project-wide physics settings: timing from robot_interface.yaml, engine knobs from
    /// <see cref="SimSettings"/>, and the Terrain / Robot layers. Idempotent; rerun after the yaml changes.
    /// Checked by the EditMode test ProjectSettingsTests.
    /// </summary>
    public static class ApplyInterfaceSettings
    {
        [MenuItem("BMB/Apply interface settings")]
        public static void Apply()
        {
            var itf = RobotInterface.Load();

            Time.fixedDeltaTime = itf.Timing.PhysicsDt;
            SetExactFixedTimestep(itf.Timing.PhysicsDt);
            Physics.simulationMode = SimulationMode.FixedUpdate;
            Physics.defaultSolverIterations = SimSettings.SolverIterations;
            Physics.defaultSolverVelocityIterations = SimSettings.SolverVelocityIterations;

            EnsureLayer(SimSettings.TerrainLayer);
            var robot = EnsureLayer(SimSettings.RobotLayer);
            Physics.IgnoreLayerCollision(robot, robot, true);

            foreach (var asset in new[] { "ProjectSettings/TimeManager.asset", "ProjectSettings/DynamicsManager.asset", "ProjectSettings/TagManager.asset" })
                EditorUtility.SetDirty(AssetDatabase.LoadAllAssetsAtPath(asset)[0]);
            AssetDatabase.SaveAssets();

            Debug.Log($"[BMB] Applied interface settings: fixedDeltaTime {Time.fixedDeltaTime}, solver {SimSettings.SolverIterations}/{SimSettings.SolverVelocityIterations}, " +
                      $"layers {SimSettings.TerrainLayer}={LayerMask.NameToLayer(SimSettings.TerrainLayer)} {SimSettings.RobotLayer}={robot} (Robot×Robot collisions off)");
        }

        /// <summary>
        /// Unity 6 stores the fixed timestep as a rational count / rate. Setting Time.fixedDeltaTime from a float
        /// truncates (0.005f gives 705599 / 141120000), so write the rounded count directly.
        /// </summary>
        static void SetExactFixedTimestep(float dt)
        {
            var timeManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TimeManager.asset")[0]);
            var count = timeManager.FindProperty("Fixed Timestep.m_Count");
            var numerator = timeManager.FindProperty("Fixed Timestep.m_Rate.m_Numerator");
            var denominator = timeManager.FindProperty("Fixed Timestep.m_Rate.m_Denominator");
            if (count == null || numerator == null || denominator == null)
                return;  // older serialization: the float setter above is all there is
            count.longValue = (long)System.Math.Round((double)dt * numerator.longValue / denominator.longValue);
            timeManager.ApplyModifiedPropertiesWithoutUndo();
        }

        /// <summary>Returns the layer index, creating the layer in the first free user slot (8+) if needed.</summary>
        static int EnsureLayer(string name)
        {
            var existing = LayerMask.NameToLayer(name);
            if (existing >= 0)
                return existing;

            var tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
            var layers = tagManager.FindProperty("layers");
            for (var i = 8; i < layers.arraySize; i++)
            {
                var slot = layers.GetArrayElementAtIndex(i);
                if (!string.IsNullOrEmpty(slot.stringValue))
                    continue;
                slot.stringValue = name;
                tagManager.ApplyModifiedPropertiesWithoutUndo();
                return i;
            }
            throw new InvalidOperationException($"no free layer slot for {name}");
        }
    }
}
