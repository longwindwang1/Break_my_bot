using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BMB.Editor
{
    /// <summary>
    /// A self-contained simulation for headless runs (trials, evaluation, tests): a preview scene with its own physics,
    /// flat ground on the Terrain layer, any number of Go2 robots (they never collide with each other), stepped by hand:
    /// every robot's control step, then PhysicsScene.Simulate(physics_dt). Deterministic and independent of play mode,
    /// frame rate and editor focus. Dispose() closes the scene and releases the inference workers.
    /// </summary>
    public sealed class HeadlessWorld : IDisposable
    {
        readonly List<Go2Controller> robots = new List<Go2Controller>();
        readonly List<LevelGround> levels = new List<LevelGround>();

        public Scene Scene { get; }
        public PhysicsScene Physics { get; }
        public RobotInterface Interface { get; }
        public IReadOnlyList<Go2Controller> Robots => robots;
        public int TerrainLayer { get; }

        HeadlessWorld(bool flatGround)
        {
            Interface = RobotInterface.Load();
            Scene = EditorSceneManager.NewPreviewScene();
            Physics = Scene.GetPhysicsScene();
            TerrainLayer = LayerMask.NameToLayer(SimSettings.TerrainLayer);
            if (flatGround)
            {
                var ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
                ground.name = "Ground";
                Add(ground);
                ground.transform.localScale = new Vector3(200f, 1f, 200f);
                ground.transform.position = new Vector3(0f, -0.5f, 0f);
                ground.layer = TerrainLayer;
                ground.GetComponent<Collider>().sharedMaterial = W3Scenes.GroundFriction(Interface);
            }
        }

        /// <summary>A new world; <paramref name="flatGround"/>: a 200 m flat ground with its top at y = 0.</summary>
        public static HeadlessWorld Create(bool flatGround = true) => new HeadlessWorld(flatGround);

        /// <summary>Moves a freshly created object into this world.</summary>
        public T Add<T>(T go) where T : UnityEngine.Object
        {
            var g = go as GameObject ?? (go as Component)?.gameObject;
            SceneManager.MoveGameObjectToScene(g, Scene);
            return go;
        }

        public HeightfieldTerrain AddRoughPatch(Vector2 size, Vector3 centre, int seed)
        {
            var patch = Add(new GameObject("RoughPatch", typeof(HeightfieldTerrain)));
            patch.layer = TerrainLayer;
            patch.transform.position = centre;
            patch.GetComponent<MeshCollider>().sharedMaterial = W3Scenes.GroundFriction(Interface);
            var terrain = patch.GetComponent<HeightfieldTerrain>();
            terrain.Configure(size, seed);
            return terrain;
        }

        /// <summary>
        /// A Go2 with its controller, spawned at <paramref name="position"/> (x, z; y from the terrain + init height)
        /// facing <paramref name="yawDeg"/> (Unity yaw). Initialized and ready to step.
        /// </summary>
        /// <param name="worldModel">Run the world-model monitor (rough / v2); <paramref name="sharedWorldModel"/> optional.</param>
        /// <param name="surprisal">World-model thresholds; null = Unity calibration if present, else the yaml's.</param>
        public Go2Controller AddRobot(string variant, Vector3 position, float yawDeg = 0f, float footFriction = 0f,
                                      bool resetOnFall = false, PolicyRunner sharedPolicy = null, Vector3? command = null,
                                      bool worldModel = false, WorldModelRunner sharedWorldModel = null, SurprisalParams surprisal = null)
        {
            var robot = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>(Go2Setup.PrefabPath), Scene);
            robot.name = $"Go2_{robots.Count}";
            robot.transform.SetPositionAndRotation(new Vector3(position.x, Interface.InitBaseHeight + 0.1f, position.z), Quaternion.Euler(0f, yawDeg, 0f));
            var body = robot.GetComponent<Go2Body>();
            body.ApplyRuntimeSettings();

            var go = Add(new GameObject($"Go2Controller_{robots.Count}"));
            var controller = go.AddComponent<Go2Controller>();
            controller.Configure(body, variant, resetAfterFall: resetOnFall, footFrictionMu: footFriction, sharedPolicy: sharedPolicy);
            controller.ExternallyStepped = true;
            controller.CommandOverride = command ?? Vector3.zero;
            if (worldModel || sharedWorldModel != null)
                controller.EnableWorldModel(sharedWorldModel, surprisal);
            UnityEngine.Physics.SyncTransforms();  // objects moved after creation; the spawn raycast must see them in place
            controller.Initialize();
            robots.Add(controller);
            return controller;
        }

        /// <summary>A disturbance manager (with its own friction map) for one robot, ready for API calls.</summary>
        public DisturbanceManager AddDisturbances(Go2Controller robot, bool hardMode = false, LevelGround level = null)
        {
            var zones = Add(new GameObject($"FrictionZones_{robot.name}")).AddComponent<FrictionZones>();
            var manager = Add(new GameObject($"Disturbances_{robot.name}")).AddComponent<DisturbanceManager>();
            manager.Configure(robot, DisturbanceLimits.CreateDefault(), null, zones, level);
            manager.HardMode = hardMode;
            return manager;
        }

        /// <summary>
        /// Editable heightfield ground (Terrain layer, terrain friction) centred at <paramref name="centre"/>.
        /// Create the world with flatGround = false, or the flat ground would fill its pits.
        /// </summary>
        public LevelGround AddLevelGround(Vector2 size, Vector3 centre = default)
        {
            var go = Add(new GameObject("LevelGround"));
            go.SetActive(false);  // build once, on enable, with the final settings
            go.layer = TerrainLayer;
            go.transform.position = centre;
            var level = go.AddComponent<LevelGround>();
            level.Configure(size, null, W3Scenes.GroundFriction(Interface));
            go.SetActive(true);
            UnityEngine.Physics.SyncTransforms();
            levels.Add(level);
            return level;
        }

        /// <summary>One physics step: every robot's control step and breakable-ice contact, then the physics.</summary>
        public void Step()
        {
            foreach (var r in robots)
                if (r.gameObject.activeSelf)
                    r.ControlStep();
            foreach (var level in levels)
                foreach (var cover in level.Covers.ToArray())
                    cover.Tick(Interface.Timing.PhysicsDt);
            Physics.Simulate(Interface.Timing.PhysicsDt);
        }

        /// <summary>Takes a robot out of the simulation (its body leaves the physics scene, its control stops).</summary>
        public void Retire(Go2Controller robot)
        {
            robot.Body.gameObject.SetActive(false);
            robot.gameObject.SetActive(false);
        }

        public void Run(float seconds, Action afterStep = null)
        {
            var steps = Mathf.RoundToInt(seconds / Interface.Timing.PhysicsDt);
            for (var i = 0; i < steps; i++)
            {
                Step();
                afterStep?.Invoke();
            }
        }

        public void Dispose()
        {
            foreach (var r in robots)
                if (r != null)
                    r.Shutdown();  // OnDestroy does not run in edit mode
            robots.Clear();
            EditorSceneManager.ClosePreviewScene(Scene);
        }
    }
}
