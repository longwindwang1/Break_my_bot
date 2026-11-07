using UnityEngine;
using UnityEngine.Rendering;

namespace BMB
{
    /// <summary>
    /// Random rough patch like Isaac Lab's HfRandomUniformTerrain ("random_rough" in the Go2 rough task):
    /// vertex heights on a horizontal_scale grid drawn uniformly from {noise_min, noise_min + step, …, noise_max},
    /// a flat border at height 0, two triangles per cell. Defaults were read off the reference rollouts' scan hits
    /// (heights 0–0.06 m, vertex heights on a 0.01 m grid, 0.1 m cells). Deterministic per seed; the mesh is
    /// regenerated on enable rather than stored in the scene. Put it on the Terrain layer.
    /// </summary>
    [ExecuteAlways]
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer), typeof(MeshCollider))]
    public sealed class HeightfieldTerrain : MonoBehaviour
    {
        [Tooltip("Patch size (m): x = Unity x (right), y = Unity z (forward)")]
        [SerializeField] Vector2 size = new Vector2(12f, 20f);
        [SerializeField] float horizontalScale = 0.1f;
        [SerializeField] float noiseMin = 0.01f;
        [SerializeField] float noiseMax = 0.06f;
        [SerializeField] float noiseStep = 0.01f;
        [SerializeField] float borderWidth = 0.25f;
        [SerializeField] int seed;

        Mesh mesh;

        public void Configure(Vector2 patchSize, int randomSeed)
        {
            size = patchSize;
            seed = randomSeed;
            Generate();
        }

        void OnEnable() => Generate();

        void OnDisable()
        {
            if (mesh != null)
                DestroyImmediate(mesh);
            mesh = null;
        }

        public void Generate()
        {
            int nx = Mathf.RoundToInt(size.x / horizontalScale) + 1, nz = Mathf.RoundToInt(size.y / horizontalScale) + 1;
            var levels = Mathf.RoundToInt((noiseMax - noiseMin) / noiseStep) + 1;
            var border = Mathf.RoundToInt(borderWidth / horizontalScale);
            var rng = new System.Random(seed);

            var vertices = new Vector3[nx * nz];
            var uv = new Vector2[vertices.Length];
            for (var iz = 0; iz < nz; iz++)
                for (var ix = 0; ix < nx; ix++)
                {
                    var inBorder = ix < border || iz < border || ix >= nx - border || iz >= nz - border;
                    var h = inBorder ? 0f : noiseMin + noiseStep * rng.Next(levels);
                    var p = new Vector3(-size.x / 2f + ix * horizontalScale, h, -size.y / 2f + iz * horizontalScale);
                    vertices[iz * nx + ix] = p;
                    uv[iz * nx + ix] = new Vector2(p.x, p.z) * 0.5f;  // 2 m texture repeat = 1 m checker squares
                }

            var triangles = new int[(nx - 1) * (nz - 1) * 6];
            var t = 0;
            for (var iz = 0; iz < nz - 1; iz++)
                for (var ix = 0; ix < nx - 1; ix++)
                {
                    int a = iz * nx + ix, b = a + 1, c = a + nx, d = c + 1;
                    triangles[t++] = a; triangles[t++] = c; triangles[t++] = b;  // clockwise from above = facing up
                    triangles[t++] = b; triangles[t++] = c; triangles[t++] = d;
                }

            if (mesh == null)
                mesh = new Mesh { name = "HeightfieldTerrain", hideFlags = HideFlags.DontSave };
            mesh.Clear();
            mesh.indexFormat = vertices.Length > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
            mesh.vertices = vertices;
            mesh.uv = uv;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            GetComponent<MeshFilter>().sharedMesh = mesh;
            var collider = GetComponent<MeshCollider>();
            collider.sharedMesh = null;  // force the collider to rebuild
            collider.sharedMesh = mesh;
        }
    }
}
