using System;
using System.Collections.Generic;
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
        public Go2Controller AddRobot(string variant, Vector3 position, float yawDeg = 0f, float footFriction = 0f,
                                      bool resetOnFall = false, PolicyRunner sharedPolicy = null, Vector3? command = null)
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
            UnityEngine.Physics.SyncTransforms();  // objects moved after creation; the spawn raycast must see them in place
            controller.Initialize();
            robots.Add(controller);
            return controller;
        }

        /// <summary>One physics step: every robot's control step, then the physics.</summary>
        public void Step()
        {
            foreach (var r in robots)
                r.ControlStep();
            Physics.Simulate(Interface.Timing.PhysicsDt);
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
