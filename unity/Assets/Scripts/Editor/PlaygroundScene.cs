using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using UnityEngine.UIElements;

namespace BMB.Editor
{
    /// <summary>
    /// The disturbance playground: v2 policy (foot μ 1.0, manual control), the disturbance manager with its tools and
    /// the neon game HUD (UI Toolkit). Idempotent: rebuilds the scene's contents.
    /// </summary>
    public static class PlaygroundScene
    {
        public const string ScenePath = "Assets/Scenes/Playground.unity";
        public const string LimitsPath = "Assets/Game/DisturbanceLimits.asset";
        /// <summary>Side of the square editable ground (m); the robot starts in its centre.</summary>
        public const float LevelSize = 60f;

        [MenuItem("BMB/Build Playground scene")]
        public static void Build()
        {
            var parts = W3Scenes.Build(ScenePath, "v2", rough: false, footFriction: 1f, save: false);
            W3Scenes.Set(parts.Commands, ("useFixedCommand", false));

            // Editable heightfield instead of the flat cube (which would fill pits).
            Object.DestroyImmediate(parts.Ground);
            var levelObject = new GameObject("LevelGround") { layer = LayerMask.NameToLayer(SimSettings.TerrainLayer) };
            levelObject.SetActive(false);
            var level = levelObject.AddComponent<LevelGround>();
            level.Configure(new Vector2(LevelSize, LevelSize), W3Scenes.LevelMaterial(), W3Scenes.GroundFriction(RobotInterface.Load()));
            levelObject.SetActive(true);

            var zones = new GameObject("FrictionZones").AddComponent<FrictionZones>();
            W3Scenes.Set(zones, ("overlayMaterial", OverlayMaterial()), ("size", new Vector2(LevelSize, LevelSize)));
            W3Scenes.Set(parts.Controller, ("frictionZones", zones));

            var game = new GameObject("Game");
            var manager = game.AddComponent<DisturbanceManager>();
            W3Scenes.Set(manager, ("controller", parts.Controller), ("limits", Limits()), ("viewCamera", parts.Camera), ("frictionZones", zones), ("level", level));

            // Game HUD (UI Toolkit) and an EventSystem fed by the Input System.
            var ui = new GameObject("GameUI");
            var document = ui.AddComponent<UIDocument>();
            document.panelSettings = PanelSettingsAsset();
            document.visualTreeAsset = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(HudPath);
            var gameUi = ui.AddComponent<GameUI>();
            W3Scenes.Set(gameUi, ("manager", manager), ("commands", parts.Commands));
            if (Object.FindFirstObjectByType<EventSystem>() == null)
                new GameObject("EventSystem", typeof(EventSystem), typeof(InputSystemUIInputModule));

            EditorSceneManager.SaveScene(parts.Scene, ScenePath);
            Debug.Log($"[BMB] Built {ScenePath}: v2, foot μ 1.0, {LevelSize} m editable ground, manual control (W/S A/D Q/E), disturbance tools 1–7, H = hard mode, F1 = help");
        }

        public const string HudPath = "Assets/UI/GameHud.uxml";
        public const string ThemePath = "Assets/UI/BmbRuntimeTheme.tss";
        public const string PanelSettingsPath = "Assets/UI/GamePanelSettings.asset";

        /// <summary>Panel scaled from a 1920 × 1080 reference, default runtime theme (our look is in Neon.uss).</summary>
        public static PanelSettings PanelSettingsAsset()
        {
            var settings = AssetDatabase.LoadAssetAtPath<PanelSettings>(PanelSettingsPath);
            if (settings == null)
            {
                settings = ScriptableObject.CreateInstance<PanelSettings>();
                AssetDatabase.CreateAsset(settings, PanelSettingsPath);
            }
            settings.themeStyleSheet = AssetDatabase.LoadAssetAtPath<ThemeStyleSheet>(ThemePath);
            settings.scaleMode = PanelScaleMode.ScaleWithScreenSize;
            settings.referenceResolution = new Vector2Int(1920, 1080);
            settings.screenMatchMode = PanelScreenMatchMode.MatchWidthOrHeight;
            settings.match = 0.5f;
            EditorUtility.SetDirty(settings);
            AssetDatabase.SaveAssets();
            return settings;
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
