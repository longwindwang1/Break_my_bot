using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace BMB.Editor
{
    /// <summary>
    /// Builds the W3 smoke-test scenes: flat ground (Terrain layer, friction from robot_interface.yaml), optionally a
    /// random rough patch, the Go2 prefab, Go2Controller + CommandSource, and a follow camera. Idempotent: rebuilds
    /// the scene's contents.
    ///   W3_Flat:  flat policy on flat ground (v1 friction)
    ///   W3_Rough: v2 policy on a rough patch, foot μ 1.0 (switch the controller's variant to "rough" and its foot
    ///             friction to 0 for the v1 rough policy)
    /// </summary>
    public static class W3Scenes
    {
        public const string FlatScenePath = "Assets/Scenes/W3_Flat.unity";
        public const string RoughScenePath = "Assets/Scenes/W3_Rough.unity";
        /// <summary>Rough patch (x width, z length) placed so a 10 s walk at 0.5 m/s from the origin stays on it.</summary>
        public static readonly Vector2 RoughPatchSize = new Vector2(12f, 20f);
        public const float RoughPatchCentreZ = 7f;
        const string TerrainDir = "Assets/Terrain";
        const string GroundFrictionPath = TerrainDir + "/Ground_friction.asset";
        const string GroundMaterialPath = TerrainDir + "/Ground.mat";
        const string RoughMaterialPath = TerrainDir + "/Rough.mat";
        const string CheckerPath = TerrainDir + "/Checker.png";
        const float GroundSize = 200f;  // m; 1 m checker squares

        [MenuItem("BMB/Build W3_Flat scene")]
        public static void BuildFlat() => Build(FlatScenePath, "flat", rough: false, footFriction: 0f);

        [MenuItem("BMB/Build W3_Rough scene")]
        public static void BuildRough() => Build(RoughScenePath, "v2", rough: true, footFriction: 1f);

        static void Build(string scenePath, string variant, bool rough, float footFriction)
        {
            var itf = RobotInterface.Load();
            var scene = File.Exists(scenePath)
                ? EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single)
                : EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
            foreach (var root in scene.GetRootGameObjects())
                if (root.name is "Ground" or "Go2" or "Go2Controller" or "RoughPatch")
                    Object.DestroyImmediate(root);

            // Ground: top face at y = 0.
            var ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
            ground.name = "Ground";
            ground.transform.localScale = new Vector3(GroundSize, 1f, GroundSize);
            ground.transform.position = new Vector3(0f, -0.5f, 0f);
            ground.layer = LayerMask.NameToLayer(SimSettings.TerrainLayer);
            ground.GetComponent<Collider>().sharedMaterial = GroundFriction(itf);
            ground.GetComponent<Renderer>().sharedMaterial = GroundMaterial();
            ground.isStatic = true;

            if (rough)
            {
                var patch = new GameObject("RoughPatch", typeof(HeightfieldTerrain));
                patch.layer = ground.layer;
                patch.transform.position = new Vector3(0f, 0f, RoughPatchCentreZ);
                patch.GetComponent<MeshCollider>().sharedMaterial = GroundFriction(itf);
                patch.GetComponent<MeshRenderer>().sharedMaterial = RoughMaterial();
                patch.GetComponent<HeightfieldTerrain>().Configure(RoughPatchSize, 0);
            }

            // Robot at the spawn pose (Go2Controller re-applies it on Start).
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(Go2Setup.PrefabPath);
            var robot = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
            robot.name = "Go2";
            robot.transform.position = new Vector3(0f, itf.InitBaseHeight, 0f);
            var body = robot.GetComponent<Go2Body>();

            var control = new GameObject("Go2Controller");
            var controller = control.AddComponent<Go2Controller>();
            var commands = control.AddComponent<CommandSource>();
            Wire(controller, ("body", body), ("commandSource", commands));
            Wire(commands, ("controller", controller));
            var so = new SerializedObject(controller);
            so.FindProperty("variant").stringValue = variant;
            so.FindProperty("footFriction").floatValue = footFriction;
            so.ApplyModifiedPropertiesWithoutUndo();

            var camera = Camera.main != null ? Camera.main.gameObject : new GameObject("Main Camera", typeof(Camera)) { tag = "MainCamera" };
            var follow = camera.GetComponent<FollowCamera>();
            if (follow == null)
                follow = camera.AddComponent<FollowCamera>();
            Wire(follow, ("target", body.Root.transform));
            camera.transform.position = robot.transform.position + new Vector3(0.9f, 0.6f, -1.4f);
            camera.transform.LookAt(robot.transform.position);

            EditorSceneManager.SaveScene(scene, scenePath);
            Debug.Log($"[BMB] Built {scenePath}: ground {GroundSize} m (friction {itf.Friction.TerrainStaticFriction}/{itf.Friction.TerrainDynamicFriction}, " +
                      $"{itf.Friction.CombineMode}){(rough ? $", rough patch {RoughPatchSize.x}x{RoughPatchSize.y} m" : "")}, " +
                      $"variant {variant}, foot μ {(footFriction > 0 ? footFriction.ToString() : "v1 0.8/0.6")}, fixed command (0.5, 0, 0); Tab = manual control");
        }

        /// <summary>Same checker as the ground, tinted, so the rough patch stands out.</summary>
        static Material RoughMaterial()
        {
            var material = AssetDatabase.LoadAssetAtPath<Material>(RoughMaterialPath);
            if (material == null)
            {
                material = new Material(GroundMaterial());
                AssetDatabase.CreateAsset(material, RoughMaterialPath);
            }
            material.SetTexture("_BaseMap", AssetDatabase.LoadAssetAtPath<Texture2D>(CheckerPath));
            material.SetTextureScale("_BaseMap", Vector2.one);  // the mesh UVs already give 1 m squares
            material.SetColor("_BaseColor", new Color(0.85f, 0.75f, 0.6f));
            EditorUtility.SetDirty(material);
            return material;
        }

        static void Wire(Object component, params (string field, Object value)[] refs)
        {
            var so = new SerializedObject(component);
            foreach (var (field, value) in refs)
                so.FindProperty(field).objectReferenceValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        /// <summary>Terrain contact material (robot_interface.yaml friction.terrain_*, combine mode).</summary>
        public static PhysicsMaterial GroundFriction(RobotInterface itf)
        {
            Directory.CreateDirectory(TerrainDir);
            var material = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(GroundFrictionPath);
            if (material == null)
            {
                material = new PhysicsMaterial("Ground_friction");
                AssetDatabase.CreateAsset(material, GroundFrictionPath);
            }
            material.staticFriction = itf.Friction.TerrainStaticFriction;
            material.dynamicFriction = itf.Friction.TerrainDynamicFriction;
            material.frictionCombine = itf.Friction.UnityCombineMode();
            material.bounciness = 0f;
            material.bounceCombine = PhysicsMaterialCombine.Multiply;
            EditorUtility.SetDirty(material);
            AssetDatabase.SaveAssets();
            return material;
        }

        /// <summary>Grey checkerboard with 1 m squares, so speed and drift are visible.</summary>
        static Material GroundMaterial()
        {
            if (!File.Exists(CheckerPath))
            {
                var tex = new Texture2D(2, 2) { filterMode = FilterMode.Point };
                Color a = new Color(0.62f, 0.64f, 0.66f), b = new Color(0.48f, 0.5f, 0.52f);
                tex.SetPixels(new[] { a, b, b, a });
                File.WriteAllBytes(CheckerPath, tex.EncodeToPNG());
                Object.DestroyImmediate(tex);
                AssetDatabase.ImportAsset(CheckerPath);
                var importer = (TextureImporter)AssetImporter.GetAtPath(CheckerPath);
                importer.filterMode = FilterMode.Point;
                importer.mipmapEnabled = false;
                importer.wrapMode = TextureWrapMode.Repeat;
                importer.SaveAndReimport();
            }
            var material = AssetDatabase.LoadAssetAtPath<Material>(GroundMaterialPath);
            if (material == null)
            {
                material = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                AssetDatabase.CreateAsset(material, GroundMaterialPath);
            }
            material.SetTexture("_BaseMap", AssetDatabase.LoadAssetAtPath<Texture2D>(CheckerPath));
            material.SetTextureScale("_BaseMap", new Vector2(GroundSize / 2f, GroundSize / 2f));
            material.SetFloat("_Smoothness", 0.2f);
            EditorUtility.SetDirty(material);
            return material;
        }
    }
}
