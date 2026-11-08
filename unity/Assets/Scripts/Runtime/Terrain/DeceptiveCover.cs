using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace BMB
{
    /// <summary>
    /// The lid of a covered pit (created and owned by <see cref="LevelGround"/> for each covered feature):
    /// - Hidden pit: a quad drawn with the ground material (world-aligned UVs, so it is invisible against the ground)
    ///   with a thin collider on the ScanOnly layer: the height scan sees flat ground, nothing stands on it. Optional
    ///   magenta outline for the player (LevelGround.RevealHidden).
    /// - Breakable ice: a translucent slab with a real collider on the Terrain layer (the scan sees it too). Robot
    ///   contact accumulates; after breakAfter seconds the slab breaks: collider and slab disappear, shards fall in,
    ///   LevelGround.CoverBroken fires. Ticked by FixedUpdate in play mode or by the headless world.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class DeceptiveCover : MonoBehaviour
    {
        const float IceThickness = 0.03f;
        const float ContactProbe = 0.05f;

        readonly Collider[] hits = new Collider[4];
        readonly List<GameObject> shards = new List<GameObject>();
        LevelGround level;
        BoxCollider box;
        Renderer slab;
        LineRenderer outline;
        int robotMask;

        public GroundFeature Feature { get; private set; }
        public bool Broken { get; private set; }
        /// <summary>Seconds the robot has touched the ice so far.</summary>
        public float ContactTime { get; private set; }

        public static DeceptiveCover Create(LevelGround level, GroundFeature f)
        {
            var hidden = f.type == GroundFeatureType.HiddenPit;
            var layer = LayerMask.NameToLayer(hidden ? SimSettings.ScanOnlyLayer : SimSettings.TerrainLayer);
            if (layer < 0)
            {
                Debug.LogError($"[BMB] Layer {(hidden ? SimSettings.ScanOnlyLayer : SimSettings.TerrainLayer)} missing; run BMB/Apply interface settings");
                layer = 0;
            }
            var go = new GameObject($"{f.type}_{f.id}") { hideFlags = HideFlags.DontSave, layer = layer };
            go.transform.SetParent(level.transform, false);
            go.transform.localPosition = new Vector3(f.centre.x, 0f, f.centre.y);
            go.transform.localRotation = Quaternion.Euler(0f, f.yawDeg, 0f);
            var cover = go.AddComponent<DeceptiveCover>();
            cover.level = level;
            cover.Feature = f;
            cover.robotMask = LayerMask.GetMask(SimSettings.RobotLayer);
            cover.Build(hidden);
            return cover;
        }

        void Build(bool hidden)
        {
            var size = Feature.size;
            box = gameObject.AddComponent<BoxCollider>();
            if (hidden)
            {
                box.size = new Vector3(size.x, 0.01f, size.y);
                box.center = new Vector3(0f, -0.005f, 0f);
                var filter = gameObject.AddComponent<MeshFilter>();
                filter.sharedMesh = GroundQuad(size);
                var r = gameObject.AddComponent<MeshRenderer>();
                r.sharedMaterial = level.GroundMaterial;
                r.shadowCastingMode = ShadowCastingMode.Off;
                slab = r;
                UpdateOutline();
            }
            else
            {
                box.size = new Vector3(size.x, IceThickness, size.y);
                box.center = new Vector3(0f, -IceThickness / 2f, 0f);
                box.sharedMaterial = level.Friction;
                var visual = GameObject.CreatePrimitive(PrimitiveType.Cube);
                visual.name = "IceSlab";
                visual.hideFlags = HideFlags.DontSave;
                DestroyImmediate(visual.GetComponent<Collider>());
                visual.transform.SetParent(transform, false);
                visual.transform.localPosition = box.center;
                visual.transform.localScale = box.size;
                visual.layer = gameObject.layer;
                slab = visual.GetComponent<Renderer>();
                if (level.IceMaterial != null)
                    slab.sharedMaterial = level.IceMaterial;
                slab.shadowCastingMode = ShadowCastingMode.Off;
            }
        }

        /// <summary>Top quad with UVs in world space (× 0.5), matching the LevelGround chunks' checker.</summary>
        Mesh GroundQuad(Vector2 size)
        {
            float hx = size.x / 2f, hz = size.y / 2f;
            var local = new[] { new Vector3(-hx, 0f, -hz), new Vector3(hx, 0f, -hz), new Vector3(-hx, 0f, hz), new Vector3(hx, 0f, hz) };
            var uv = new Vector2[4];
            for (var i = 0; i < 4; i++)
            {
                var w = transform.TransformPoint(local[i]);
                uv[i] = new Vector2(w.x, w.z) * 0.5f;
            }
            var mesh = new Mesh { name = "HiddenPitCover", hideFlags = HideFlags.DontSave, vertices = local, uv = uv, triangles = new[] { 0, 2, 3, 0, 3, 1 } };
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        public void UpdateOutline()
        {
            if (Feature.type != GroundFeatureType.HiddenPit)
                return;
            var show = level != null && level.RevealHidden;
            if (show && outline == null)
            {
                var go = new GameObject("Outline") { hideFlags = HideFlags.DontSave, layer = 0 };
                go.transform.SetParent(transform, false);
                outline = go.AddComponent<LineRenderer>();
                outline.useWorldSpace = false;
                outline.loop = true;
                outline.widthMultiplier = 0.025f;
                outline.shadowCastingMode = ShadowCastingMode.Off;
                outline.sharedMaterial = new Material(Shader.Find("Sprites/Default")) { hideFlags = HideFlags.DontSave };
                outline.startColor = outline.endColor = new Color(1f, 0.17f, 0.84f, 0.9f);
                float hx = Feature.size.x / 2f, hz = Feature.size.y / 2f;
                outline.positionCount = 4;
                outline.SetPositions(new[] { new Vector3(-hx, 0.01f, -hz), new Vector3(-hx, 0.01f, hz), new Vector3(hx, 0.01f, hz), new Vector3(hx, 0.01f, -hz) });
            }
            if (outline != null)
                outline.gameObject.SetActive(show);
        }

        void FixedUpdate()
        {
            if (Application.isPlaying)
                Tick(Time.fixedDeltaTime);
        }

        /// <summary>Breakable ice: accumulate robot contact and break once it exceeds breakAfter.</summary>
        public void Tick(float dt)
        {
            if (Broken || Feature.type != GroundFeatureType.BreakableIce)
                return;
            var t = transform;
            var centre = t.TransformPoint(new Vector3(0f, ContactProbe / 2f, 0f));
            var half = new Vector3(Feature.size.x / 2f, ContactProbe / 2f, Feature.size.y / 2f);
            if (gameObject.scene.GetPhysicsScene().OverlapBox(centre, half, hits, t.rotation, robotMask, QueryTriggerInteraction.Ignore) == 0)
                return;
            ContactTime += dt;
            if (ContactTime >= Feature.breakAfter)
                Break();
        }

        public void Break()
        {
            if (Broken)
                return;
            Broken = true;
            box.enabled = false;
            slab.enabled = false;
            if (Feature.type == GroundFeatureType.BreakableIce)
                SpawnShards();
            level.NotifyBroken(this);
        }

        public void Restore()
        {
            Broken = false;
            ContactTime = 0f;
            box.enabled = true;
            slab.enabled = true;
            foreach (var s in shards)
                if (s != null)
                    DestroyImmediate(s);
            shards.Clear();
        }

        void SpawnShards()
        {
            var rng = new System.Random(Feature.id);
            for (var i = 0; i < 8; i++)
            {
                var s = GameObject.CreatePrimitive(PrimitiveType.Cube);
                s.name = "IceShard";
                s.hideFlags = HideFlags.DontSave;
                if (s.scene != gameObject.scene)
                    SceneManager.MoveGameObjectToScene(s, gameObject.scene);
                s.transform.SetParent(transform, false);
                s.transform.localPosition = new Vector3(
                    ((float)rng.NextDouble() - 0.5f) * Feature.size.x * 0.8f, -IceThickness, ((float)rng.NextDouble() - 0.5f) * Feature.size.y * 0.8f);
                s.transform.localRotation = Quaternion.Euler((float)rng.NextDouble() * 90f, (float)rng.NextDouble() * 90f, 0f);
                s.transform.localScale = Vector3.one * (0.08f + 0.1f * (float)rng.NextDouble());
                if (level.IceMaterial != null)
                    s.GetComponent<Renderer>().sharedMaterial = level.IceMaterial;
                var body = s.AddComponent<Rigidbody>();
                body.mass = 0.2f;
                shards.Add(s);
                if (Application.isPlaying)
                    Destroy(s, 4f);
            }
        }

        void OnDestroy()
        {
            if (outline != null && outline.sharedMaterial != null)
                DestroyImmediate(outline.sharedMaterial);
            var filter = GetComponent<MeshFilter>();
            if (filter != null && filter.sharedMesh != null)
                DestroyImmediate(filter.sharedMesh);
        }
    }
}
