using System;
using System.Collections.Generic;
using UnityEngine;

namespace BMB
{
    public enum GroundFeatureType
    {
        /// <summary>Raised block (height = top above the base level).</summary>
        Box,
        /// <summary>Hole (height = depth).</summary>
        Pit,
        /// <summary>Hill: ramp up, plateau, ramp down along the feature's length (height = rise, slopeDeg = angle).</summary>
        Ramp,
        /// <summary>Stairs up, plateau, stairs down (height = step height, steps = steps per side, 0.3 m deep).</summary>
        Stairs,
        /// <summary>Random rough patch like random_rough (height = max noise).</summary>
        Rough,
        /// <summary>An 8 m Isaac Lab tile (tile, difficulty, seed); yaw in multiples of 90°.</summary>
        IsaacTile,
        /// <summary>Pit under a ground-looking cover on the ScanOnly layer: the height scan sees flat ground, feet fall in.</summary>
        HiddenPit,
        /// <summary>Pit under a solid ice slab that breaks after the robot has stood on it for breakAfter seconds.</summary>
        BreakableIce,
    }

    /// <summary>One edit of the ground, kept as data so the level can be recomputed, edited and (later) saved.</summary>
    [Serializable]
    public sealed class GroundFeature
    {
        public int id;
        public GroundFeatureType type;
        [Tooltip("Centre (Unity x, z) relative to the LevelGround object")] public Vector2 centre;
        [Tooltip("Footprint: x = width, y = length (along the feature's forward)")] public Vector2 size = Vector2.one;
        public float yawDeg;
        public float height = 0.1f;
        public int steps = 3;
        public float slopeDeg = 15f;
        public IsaacTileType tile;
        public float difficulty = 0.5f;
        public int seed;
        [Tooltip("Breakable ice: seconds of contact with the robot before it breaks")]
        public float breakAfter = 0.4f;

        public const float StairDepth = 0.3f;
        public const float Plateau = 1f;

        /// <summary>Isaac mesh terrains and our blocks / pits / stairs get vertical walls at any height.</summary>
        public bool Hard => type is GroundFeatureType.Box or GroundFeatureType.Pit or GroundFeatureType.Stairs
                            or GroundFeatureType.HiddenPit or GroundFeatureType.BreakableIce
                            || (type == GroundFeatureType.IsaacTile && IsaacTiles.IsHard(tile));

        /// <summary>A pit with a cover object (hidden pit, breakable ice).</summary>
        public bool Covered => type is GroundFeatureType.HiddenPit or GroundFeatureType.BreakableIce;

        public Vector2 ToLocal(Vector2 p)
        {
            var d = p - centre;
            var a = -yawDeg * Mathf.Deg2Rad;
            // Unity yaw rotates z (forward) toward x (right): local x = across, local y = along.
            return new Vector2(d.x * Mathf.Cos(a) + d.y * Mathf.Sin(a), -d.x * Mathf.Sin(a) + d.y * Mathf.Cos(a));
        }

        public bool Contains(Vector2 p, float margin = 1e-4f)
        {
            var l = ToLocal(p);
            return Mathf.Abs(l.x) <= size.x / 2f + margin && Mathf.Abs(l.y) <= size.y / 2f + margin;
        }

        /// <summary>Length a ramp needs for its rise at its slope.</summary>
        public float RampLength => height / Mathf.Tan(Mathf.Max(1f, slopeDeg) * Mathf.Deg2Rad);

        /// <summary>Footprint length implied by the parameters (stairs, ramps); width unchanged.</summary>
        public static float NaturalLength(GroundFeatureType type, int steps, float rampLength) => type switch
        {
            GroundFeatureType.Stairs => 2f * steps * StairDepth + Plateau,
            GroundFeatureType.Ramp => 2f * rampLength + Plateau,
            GroundFeatureType.IsaacTile => IsaacTiles.Size,
            _ => -1f,
        };

        public string Describe() => type switch
        {
            GroundFeatureType.Box => $"box {size.x:F1}×{size.y:F1} m, {height * 100:F0} cm high",
            GroundFeatureType.Pit => $"pit {size.x:F1}×{size.y:F1} m, {height * 100:F0} cm deep",
            GroundFeatureType.Ramp => $"ramp {slopeDeg:F0}°, rise {height * 100:F0} cm",
            GroundFeatureType.Stairs => $"stairs {steps}×{height * 100:F0} cm",
            GroundFeatureType.Rough => $"rough {size.x:F1}×{size.y:F1} m, up to {height * 100:F0} cm",
            GroundFeatureType.IsaacTile => $"Isaac tile {tile}, difficulty {difficulty:F2}",
            GroundFeatureType.HiddenPit => $"hidden pit {size.x:F1}×{size.y:F1} m, {height * 100:F0} cm deep",
            GroundFeatureType.BreakableIce => $"breakable ice {size.x:F1}×{size.y:F1} m over {height * 100:F0} cm, breaks after {breakAfter:F1} s",
            _ => type.ToString(),
        };
    }

    /// <summary>
    /// Editable ground: a heightfield on a 0.1 m vertex grid (Isaac's horizontal_scale), built from a list of
    /// <see cref="GroundFeature"/>s applied in order, meshed in 10 m chunks with MeshColliders on the Terrain layer.
    /// Meshing ports Isaac Lab's convert_height_field_to_mesh: where neighbouring vertices differ by more than
    /// slope_threshold (0.75) × 0.1 m, the lower vertex moves under the higher one, giving a vertical wall. Vertices
    /// written by "hard" features (Isaac's mesh terrains, our blocks, pits and stairs) use a zero threshold, so their
    /// walls are vertical at any height, like Isaac's box meshes. Only chunks whose vertices changed are rebuilt.
    /// The feature list is serialized; meshes are regenerated on enable.
    /// </summary>
    [ExecuteAlways]
    public sealed class LevelGround : MonoBehaviour
    {
        public const float Cell = 0.1f;
        public const float SlopeThreshold = 0.75f;
        const int ChunkCells = 100;
        const float HardThreshold = 1e-4f;

        [Tooltip("Area (m), centred on this object; must be a multiple of 0.2 m")]
        [SerializeField] Vector2 size = new Vector2(60f, 60f);
        [SerializeField] Material material;
        [SerializeField] PhysicsMaterial friction;
        [Tooltip("Breakable ice slabs (translucent)")]
        [SerializeField] Material iceMaterial;
        [Tooltip("Outline hidden pits for the player (the robot's scan is fooled either way)")]
        [SerializeField] bool revealHidden;
        [SerializeField] List<GroundFeature> features = new List<GroundFeature>();
        [SerializeField] int nextId = 1;

        readonly Dictionary<int, DeceptiveCover> covers = new Dictionary<int, DeceptiveCover>();

        /// <summary>Raised when a breakable-ice slab breaks.</summary>
        public event Action<DeceptiveCover> CoverBroken;

        public IReadOnlyCollection<DeceptiveCover> Covers => covers.Values;
        public Material GroundMaterial => material;
        public Material IceMaterial { get => iceMaterial; set => iceMaterial = value; }
        public PhysicsMaterial Friction => friction;

        public bool RevealHidden
        {
            get => revealHidden;
            set
            {
                revealHidden = value;
                foreach (var c in covers.Values)
                    c.UpdateOutline();
            }
        }

        /// <summary>Puts every broken ice slab back.</summary>
        public void RestoreCovers()
        {
            foreach (var c in covers.Values)
                c.Restore();
        }

        internal void NotifyBroken(DeceptiveCover cover) => CoverBroken?.Invoke(cover);

        /// <summary>Creates covers for new covered features and removes those of deleted ones (broken stays broken).</summary>
        void SyncCovers()
        {
            var alive = new HashSet<int>();
            foreach (var f in features)
            {
                if (!f.Covered)
                    continue;
                alive.Add(f.id);
                if (!covers.TryGetValue(f.id, out var c) || c == null)
                    covers[f.id] = DeceptiveCover.Create(this, f);
            }
            foreach (var id in new List<int>(covers.Keys))
                if (!alive.Contains(id))
                {
                    if (covers[id] != null)
                        DestroyImmediate(covers[id].gameObject);
                    covers.Remove(id);
                }
        }

        void DestroyCovers()
        {
            foreach (var c in covers.Values)
                if (c != null)
                    DestroyImmediate(c.gameObject);
            covers.Clear();
        }

        int nx, nz;  // cells; vertices are (nx + 1) × (nz + 1)
        float[] heights;
        bool[] hard;
        GameObject[] chunks;
        Mesh[] chunkMeshes;
        int chunksX, chunksZ;
        readonly Dictionary<int, float[,]> tileCache = new Dictionary<int, float[,]>();

        public IReadOnlyList<GroundFeature> Features => features;
        public Vector2 Size => size;

        public void Configure(Vector2 areaSize, Material renderMaterial, PhysicsMaterial physicsMaterial)
        {
            DestroyChunks();
            size = areaSize;
            material = renderMaterial;
            friction = physicsMaterial;
            heights = null;
            if (isActiveAndEnabled)
                Refresh();  // otherwise OnEnable builds it
        }

        void OnEnable()
        {
            heights = null;
            Refresh();
        }

        void OnDisable()
        {
            DestroyChunks();
            DestroyCovers();
        }

        // --- features ---------------------------------------------------------------------------------------------

        /// <summary>Adds a feature (assigning its id) and rebuilds the affected chunks unless <paramref name="refresh"/> is false.</summary>
        public int Add(GroundFeature f, bool refresh = true)
        {
            f.id = nextId++;
            features.Add(f);
            if (refresh)
                Refresh();
            return f.id;
        }

        public bool Remove(int id)
        {
            var removed = features.RemoveAll(f => f.id == id) > 0;
            tileCache.Remove(id);
            if (removed)
                Refresh();
            return removed;
        }

        public void Clear()
        {
            features.Clear();
            tileCache.Clear();
            Refresh();
        }

        public GroundFeature Feature(int id) => features.Find(f => f.id == id);

        /// <summary>The topmost (last added) feature whose footprint contains the point, or null.</summary>
        public GroundFeature FeatureAt(Vector3 world)
        {
            var p = new Vector2(world.x - transform.position.x, world.z - transform.position.z);
            for (var i = features.Count - 1; i >= 0; i--)
                if (features[i].Contains(p))
                    return features[i];
            return null;
        }

        /// <summary>Snaps a world point to the vertex grid (x, z).</summary>
        public Vector3 Snap(Vector3 world)
        {
            var o = transform.position;
            return new Vector3(o.x + Mathf.Round((world.x - o.x) / Cell) * Cell, world.y, o.z + Mathf.Round((world.z - o.z) / Cell) * Cell);
        }

        /// <summary>Height of the vertex nearest to a world point (before meshing moves vertices sideways).</summary>
        public float VertexHeight(Vector3 world)
        {
            EnsureGrid();
            var (ix, iz) = VertexIndex(world);
            return heights[V(ix, iz)] + transform.position.y;
        }

        // --- heightfield ----------------------------------------------------------------------------------------

        int V(int ix, int iz) => iz * (nx + 1) + ix;

        (int, int) VertexIndex(Vector3 world)
        {
            var o = transform.position;
            var ix = Mathf.Clamp(Mathf.RoundToInt((world.x - o.x + size.x / 2f) / Cell), 0, nx);
            var iz = Mathf.Clamp(Mathf.RoundToInt((world.z - o.z + size.y / 2f) / Cell), 0, nz);
            return (ix, iz);
        }

        /// <summary>Vertex position relative to the object (x, z), before meshing moves.</summary>
        Vector2 VertexLocal(int ix, int iz) => new Vector2(-size.x / 2f + ix * Cell, -size.y / 2f + iz * Cell);

        void EnsureGrid()
        {
            if (heights != null)
                return;
            nx = Mathf.RoundToInt(size.x / Cell);
            nz = Mathf.RoundToInt(size.y / Cell);
            heights = new float[(nx + 1) * (nz + 1)];
            hard = new bool[heights.Length];
            chunksX = Mathf.CeilToInt(nx / (float)ChunkCells);
            chunksZ = Mathf.CeilToInt(nz / (float)ChunkCells);
        }

        /// <summary>Recomputes the heightfield from the feature list and rebuilds the chunks that changed.</summary>
        public void Refresh()
        {
            var first = heights == null;
            EnsureGrid();
            var oldH = (float[])heights.Clone();
            var oldHard = (bool[])hard.Clone();
            Array.Clear(heights, 0, heights.Length);
            Array.Clear(hard, 0, hard.Length);
            foreach (var f in features)
                Stamp(f);

            var dirty = new bool[chunksX * chunksZ];
            for (var iz = 0; iz <= nz; iz++)
                for (var ix = 0; ix <= nx; ix++)
                {
                    var v = V(ix, iz);
                    if (!first && oldH[v] == heights[v] && oldHard[v] == hard[v])
                        continue;
                    // A vertex's move depends on its neighbours: dirty every chunk within one vertex.
                    for (var cz = Mathf.Max(0, (iz - 1) / ChunkCells); cz <= Mathf.Min(chunksZ - 1, (iz + 1) / ChunkCells); cz++)
                        for (var cx = Mathf.Max(0, (ix - 1) / ChunkCells); cx <= Mathf.Min(chunksX - 1, (ix + 1) / ChunkCells); cx++)
                            dirty[cz * chunksX + cx] = true;
                }
            if (chunks == null || chunks.Length != dirty.Length || chunks[0] == null)
            {
                DestroyChunks();
                chunks = new GameObject[dirty.Length];
                chunkMeshes = new Mesh[dirty.Length];
                for (var i = 0; i < dirty.Length; i++)
                    dirty[i] = true;
            }
            for (var i = 0; i < dirty.Length; i++)
                if (dirty[i])
                    BuildChunk(i % chunksX, i / chunksX);
            SyncCovers();
        }

        void Stamp(GroundFeature f)
        {
            // Vertex bounding box of the footprint (any yaw), one vertex of margin.
            var r = 0.5f * new Vector2(f.size.x, f.size.y).magnitude + Cell;
            var lo = VertexIndex(transform.position + new Vector3(f.centre.x - r, 0f, f.centre.y - r));
            var hi = VertexIndex(transform.position + new Vector3(f.centre.x + r, 0f, f.centre.y + r));

            if (f.type == GroundFeatureType.IsaacTile)
            {
                StampTile(f, lo, hi);
                return;
            }
            for (var iz = lo.Item2; iz <= hi.Item2; iz++)
                for (var ix = lo.Item1; ix <= hi.Item1; ix++)
                {
                    var p = VertexLocal(ix, iz);
                    var v = V(ix, iz);
                    if (f.Hard)
                    {
                        // Cell-based: a vertex takes the highest of its cells inside the footprint; vertices on the
                        // footprint's edge keep the existing height if that is higher (so walls fall on cell edges).
                        var inside = 0;
                        var best = float.NegativeInfinity;
                        for (var dz = -0.5f; dz <= 0.5f; dz += 1f)
                            for (var dx = -0.5f; dx <= 0.5f; dx += 1f)
                            {
                                var c = p + new Vector2(dx * Cell, dz * Cell);
                                if (!f.Contains(c, 0f))
                                    continue;
                                inside++;
                                best = Mathf.Max(best, Value(f, f.ToLocal(c), ix, iz));
                            }
                        if (inside == 0)
                            continue;
                        heights[v] = inside == 4 ? best : Mathf.Max(heights[v], best);
                        hard[v] = true;
                    }
                    else if (f.Contains(p))
                    {
                        heights[v] = Value(f, f.ToLocal(p), ix, iz);
                        hard[v] = false;
                    }
                }
        }

        void StampTile(GroundFeature f, (int, int) lo, (int, int) hi)
        {
            if (!tileCache.TryGetValue(f.id, out var tile))
                tileCache[f.id] = tile = IsaacTiles.Generate(f.tile, f.difficulty, new System.Random(f.seed));
            var isHard = IsaacTiles.IsHard(f.tile);
            const int half = (IsaacTiles.Vertices - 1) / 2;
            for (var iz = lo.Item2; iz <= hi.Item2; iz++)
                for (var ix = lo.Item1; ix <= hi.Item1; ix++)
                {
                    var l = f.ToLocal(VertexLocal(ix, iz));
                    var i = Mathf.RoundToInt(l.x / Cell) + half;
                    var j = Mathf.RoundToInt(l.y / Cell) + half;
                    if (i < 0 || j < 0 || i >= IsaacTiles.Vertices || j >= IsaacTiles.Vertices)
                        continue;
                    var v = V(ix, iz);
                    heights[v] = tile[i, j];
                    hard[v] = isHard;
                }
        }

        /// <summary>Height of a feature at a point in its local frame (x across, y along).</summary>
        static float Value(GroundFeature f, Vector2 local, int ix, int iz)
        {
            switch (f.type)
            {
                case GroundFeatureType.Box:
                    return f.height;
                case GroundFeatureType.Pit:
                case GroundFeatureType.HiddenPit:
                case GroundFeatureType.BreakableIce:
                    return -f.height;
                case GroundFeatureType.Stairs:
                {
                    var fromEnd = f.size.y / 2f - Mathf.Abs(local.y);
                    var k = Mathf.FloorToInt(fromEnd / GroundFeature.StairDepth);
                    return Mathf.Min(k + 1, f.steps) * f.height;
                }
                case GroundFeatureType.Ramp:
                {
                    var fromEnd = f.size.y / 2f - Mathf.Abs(local.y);
                    return Mathf.Clamp01(fromEnd / f.RampLength) * f.height;
                }
                case GroundFeatureType.Rough:
                {
                    var levels = Mathf.Max(1, Mathf.RoundToInt((f.height - 0.01f) / 0.01f) + 1);
                    var hash = (ix * 73856093) ^ (iz * 19349663) ^ (f.seed * 83492791);
                    return 0.01f + 0.01f * ((hash & int.MaxValue) % levels);
                }
                default:
                    return 0f;
            }
        }

        // --- meshing (Isaac convert_height_field_to_mesh with per-vertex hard edges) ------------------------------

        float H(int ix, int iz) => heights[V(ix, iz)];

        bool Steeper(int ax, int az, int bx, int bz)
        {
            // Neighbour b rises above a by more than the threshold.
            var threshold = hard[V(ax, az)] || hard[V(bx, bz)] ? HardThreshold : SlopeThreshold * Cell;
            return H(bx, bz) - H(ax, az) > threshold;
        }

        Vector2 Move(int ix, int iz)
        {
            int mx = 0, mz = 0, mc = 0;
            if (ix < nx && Steeper(ix, iz, ix + 1, iz)) mx += 1;
            if (ix > 0 && Steeper(ix, iz, ix - 1, iz)) mx -= 1;
            if (iz < nz && Steeper(ix, iz, ix, iz + 1)) mz += 1;
            if (iz > 0 && Steeper(ix, iz, ix, iz - 1)) mz -= 1;
            if (ix < nx && iz < nz && Steeper(ix, iz, ix + 1, iz + 1)) mc += 1;
            if (ix > 0 && iz > 0 && Steeper(ix, iz, ix - 1, iz - 1)) mc -= 1;
            return new Vector2(mx + (mx == 0 ? mc : 0), mz + (mz == 0 ? mc : 0)) * Cell;
        }

        void BuildChunk(int cx, int cz)
        {
            var i = cz * chunksX + cx;
            int x0 = cx * ChunkCells, z0 = cz * ChunkCells;
            int x1 = Mathf.Min(x0 + ChunkCells, nx), z1 = Mathf.Min(z0 + ChunkCells, nz);
            int w = x1 - x0 + 1, d = z1 - z0 + 1;

            var vertices = new Vector3[w * d];
            var uv = new Vector2[w * d];
            for (var z = 0; z < d; z++)
                for (var x = 0; x < w; x++)
                {
                    int ix = x0 + x, iz = z0 + z;
                    var p = VertexLocal(ix, iz) + Move(ix, iz);
                    vertices[z * w + x] = new Vector3(p.x, H(ix, iz), p.y);
                    uv[z * w + x] = (p + new Vector2(transform.position.x, transform.position.z)) * 0.5f;  // 1 m checker
                }
            var triangles = new int[(w - 1) * (d - 1) * 6];
            var t = 0;
            for (var z = 0; z < d - 1; z++)
                for (var x = 0; x < w - 1; x++)
                {
                    int a = z * w + x, b = a + 1, c = a + w, e = c + 1;  // a = (x, z), b = (x+1, z), c = (x, z+1), e = (x+1, z+1)
                    // Isaac's diagonal (ind0–ind3); clockwise seen from above = facing up in Unity.
                    triangles[t++] = a; triangles[t++] = c; triangles[t++] = e;
                    triangles[t++] = a; triangles[t++] = e; triangles[t++] = b;
                }

            if (chunks[i] == null)
            {
                chunks[i] = new GameObject($"Chunk_{cx}_{cz}", typeof(MeshFilter), typeof(MeshRenderer), typeof(MeshCollider))
                {
                    hideFlags = HideFlags.DontSave,
                    layer = gameObject.layer,
                };
                chunks[i].transform.SetParent(transform, false);
                chunkMeshes[i] = new Mesh { name = chunks[i].name, hideFlags = HideFlags.DontSave };
                chunks[i].GetComponent<MeshFilter>().sharedMesh = chunkMeshes[i];
            }
            var mesh = chunkMeshes[i];
            mesh.Clear();
            mesh.vertices = vertices;
            mesh.uv = uv;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            chunks[i].layer = gameObject.layer;
            chunks[i].GetComponent<MeshRenderer>().sharedMaterial = material;
            var collider = chunks[i].GetComponent<MeshCollider>();
            collider.sharedMaterial = friction;
            collider.sharedMesh = null;  // force re-cooking
            collider.sharedMesh = mesh;
        }

        void DestroyChunks()
        {
            if (chunks != null)
                foreach (var c in chunks)
                    if (c != null)
                        DestroyImmediate(c);
            if (chunkMeshes != null)
                foreach (var m in chunkMeshes)
                    if (m != null)
                        DestroyImmediate(m);
            chunks = null;
            chunkMeshes = null;
        }
    }
}
