using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace BMB.Editor
{
    /// <summary>
    /// The disturbance playground: v2 policy (foot μ 1.0, manual control), the disturbance manager with its tools and
    /// the debug HUD. Idempotent: rebuilds the scene's contents.
    /// </summary>
    public static class PlaygroundScene
    {
        public const string ScenePath = "Assets/Scenes/Playground.unity";
        public const string LimitsPath = "Assets/Game/DisturbanceLimits.asset";

        [MenuItem("BMB/Build Playground scene")]
        public static void Build()
        {
            var parts = W3Scenes.Build(ScenePath, "v2", rough: false, footFriction: 1f, save: false);
            W3Scenes.Set(parts.Commands, ("useFixedCommand", false));

            var zones = new GameObject("FrictionZones").AddComponent<FrictionZones>();
            W3Scenes.Set(zones, ("overlayMaterial", OverlayMaterial()));
            W3Scenes.Set(parts.Controller, ("frictionZones", zones));

            var game = new GameObject("Game");
            var manager = game.AddComponent<DisturbanceManager>();
            W3Scenes.Set(manager, ("controller", parts.Controller), ("limits", Limits()), ("viewCamera", parts.Camera), ("frictionZones", zones));
            var hud = game.AddComponent<DebugHud>();
            W3Scenes.Set(hud, ("manager", manager), ("commandSource", parts.Commands));

            EditorSceneManager.SaveScene(parts.Scene, ScenePath);
            Debug.Log($"[BMB] Built {ScenePath}: v2, foot μ 1.0, manual control (W/S A/D Q/E), disturbance tools 1–7, H = hard mode, F1 = help");
        }

        public const string OverlayMaterialPath = "Assets/Game/FrictionOverlay.mat";

        /// <summary>Unlit, transparent, vertex-coloured material for the friction overlay.</summary>
        public static Material OverlayMaterial()
        {
            var material = AssetDatabase.LoadAssetAtPath<Material>(OverlayMaterialPath);
            if (material != null)
                return material;
            System.IO.Directory.CreateDirectory("Assets/Game");
            material = new Material(Shader.Find("Sprites/Default"));
            AssetDatabase.CreateAsset(material, OverlayMaterialPath);
            AssetDatabase.SaveAssets();
            return material;
        }

        /// <summary>The limits asset (created with the v2 training ranges and hard-mode defaults if missing).</summary>
        public static DisturbanceLimits Limits()
        {
            var limits = AssetDatabase.LoadAssetAtPath<DisturbanceLimits>(LimitsPath);
            if (limits != null)
                return limits;
            System.IO.Directory.CreateDirectory("Assets/Game");
            limits = DisturbanceLimits.CreateDefault();
            AssetDatabase.CreateAsset(limits, LimitsPath);
            AssetDatabase.SaveAssets();
            return limits;
        }
    }
}
