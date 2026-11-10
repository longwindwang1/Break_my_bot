using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace BMB
{
    /// <summary>
    /// Ground friction map: a 0.1 m grid of painted μ over the play area plus a global μ for everything unpainted
    /// (1.0 = the terrain material of robot_interface.yaml). The robot reads it per foot (FootFrictionDriver), so a
    /// single foot can stand on ice, as with v2's per-foot friction changes. Painted cells are drawn as a translucent
    /// overlay that follows the terrain (blue = slippery, orange = grippy).
    /// </summary>
    public sealed class FrictionZones : MonoBehaviour
    {
        [Tooltip("Area covered by the grid (m), centred on this object; outside it only the global μ applies")]
        [SerializeField] Vector2 size = new Vector2(40f, 40f);
        [SerializeField] float cellSize = 0.1f;
        [Tooltip("Transparent, vertex-coloured material for the overlay (none = no overlay)")]
        [SerializeField] Material overlayMaterial;

        float[] grid;  // NaN = not painted
        int nx, nz;
        float globalMu = 1f;
        bool overlayDirty;
        Mesh overlayMesh;
        GameObject overlay;

        public float GlobalMu
        {
            get => globalMu;
            set => globalMu = value;
        }

        public int PaintedCells { get; private set; }
        public float CellSize => cellSize;

        public void Configure(Vector2 areaSize, Material overlay = null)
        {
            size = areaSize;
            overlayMaterial = overlay;
            grid = null;
            EnsureGrid();
        }

        void EnsureGrid()
        {
            if (grid != null)
                return;
            nx = Mathf.CeilToInt(size.x / cellSize);
            nz = Mathf.CeilToInt(size.y / cellSize);
            grid = new float[nx * nz];
            for (var i = 0; i < grid.Length; i++)
                grid[i] = float.NaN;
            PaintedCells = 0;
        }

        bool Cell(Vector3 world, out int ix, out int iz)
        {
            EnsureGrid();
            var local = world - transform.position;
            ix = Mathf.FloorToInt((local.x + size.x / 2f) / cellSize);
            iz = Mathf.FloorToInt((local.z + size.y / 2f) / cellSize);
            return ix >= 0 && iz >= 0 && ix < nx && iz < nz;
        }

        Vector3 CellCentre(int ix, int iz) =>
            transform.position + new Vector3(-size.x / 2f + (ix + 0.5f) * cellSize, 0f, -size.y / 2f + (iz + 0.5f) * cellSize);

        /// <summary>Ground μ at a world point: the painted value, else the global one.</summary>
        public float MuAt(Vector3 world)
        {
            if (!Cell(world, out var ix, out var iz))
                return globalMu;
            var v = grid[iz * nx + ix];
            return float.IsNaN(v) ? globalMu : v;
        }

        /// <summary>Painted μ at a world point, NaN where nothing is painted (the global μ applies there).</summary>
        public float PaintedMuAt(Vector3 world)
        {
            if (!Cell(world, out var ix, out var iz))
                return float.NaN;
            return grid[iz * nx + ix];
        }

        /// <summary>
        /// Painted patches with μ below <paramref name="maxMu"/> within <paramref name="radius"/> of a point: the painted
        /// grid sampled every <paramref name="step"/> m, grouped into 8-connected blobs. Read-only; for the AI pilot.
        /// </summary>
        public List<FrictionPatch> SamplePatches(Vector3 centre, float radius, float maxMu = 0.5f, float step = 0.5f)
        {
            EnsureGrid();
            var n = Mathf.Max(1, Mathf.CeilToInt(radius / step));
            var side = 2 * n + 1;
            var mu = new float[side * side];
            for (var j = 0; j < side; j++)
                for (var i = 0; i < side; i++)
                {
                    var dx = (i - n) * step;
                    var dz = (j - n) * step;
                    var v = float.NaN;
                    if (dx * dx + dz * dz <= radius * radius)
                    {
                        var p = PaintedMuAt(centre + new Vector3(dx, 0f, dz));
                        if (!float.IsNaN(p) && p < maxMu)
                            v = p;
                    }
                    mu[j * side + i] = v;
                }

            var patches = new List<FrictionPatch>();
            var seen = new bool[mu.Length];
            var stack = new Stack<int>();
            var cells = new List<int>();
            for (var s = 0; s < mu.Length; s++)
            {
                if (seen[s] || float.IsNaN(mu[s]))
                    continue;
                cells.Clear();
                seen[s] = true;
                stack.Push(s);
                while (stack.Count > 0)
                {
                    var c = stack.Pop();
                    cells.Add(c);
                    int ci = c % side, cj = c / side;
                    for (var dj = -1; dj <= 1; dj++)
                        for (var di = -1; di <= 1; di++)
                        {
                            int ni = ci + di, nj = cj + dj;
                            if (ni < 0 || nj < 0 || ni >= side || nj >= side)
                                continue;
                            var k = nj * side + ni;
                            if (!seen[k] && !float.IsNaN(mu[k]))
                            {
                                seen[k] = true;
                                stack.Push(k);
                            }
                        }
                }

                var sum = Vector3.zero;
                float minMu = float.MaxValue, meanMu = 0f;
                foreach (var c in cells)
                {
                    sum += Offset(c);
                    minMu = Mathf.Min(minMu, mu[c]);
                    meanMu += mu[c];
                }
                var mid = sum / cells.Count;
                var extent = 0f;
                foreach (var c in cells)
                    extent = Mathf.Max(extent, (Offset(c) - mid).magnitude);
                patches.Add(new FrictionPatch
                {
                    Centre = centre + mid,
                    Radius = extent + step / 2f,
                    MinMu = minMu,
                    MeanMu = meanMu / cells.Count,
                    Samples = cells.Count,
                });
            }
            return patches;

            Vector3 Offset(int c) => new Vector3((c % side - n) * step, 0f, (c / side - n) * step);
        }

        /// <summary>Paints μ on every cell whose centre lies within the disc; returns the number of cells.</summary>
        public int Paint(Vector3 centre, float radius, float mu) => Stamp(centre, radius, mu);

        /// <summary>Removes painted cells within the disc; returns the number of cells.</summary>
        public int Erase(Vector3 centre, float radius) => Stamp(centre, radius, float.NaN);

        public void ClearAll()
        {
            grid = null;
            EnsureGrid();
            overlayDirty = true;
        }

        int Stamp(Vector3 centre, float radius, float value)
        {
            EnsureGrid();
            Cell(centre - new Vector3(radius, 0f, radius), out var x0, out var z0);
            Cell(centre + new Vector3(radius, 0f, radius), out var x1, out var z1);
            var count = 0;
            for (var iz = Mathf.Max(0, z0); iz <= Mathf.Min(nz - 1, z1); iz++)
                for (var ix = Mathf.Max(0, x0); ix <= Mathf.Min(nx - 1, x1); ix++)
                {
                    var c = CellCentre(ix, iz);
                    if ((c.x - centre.x) * (c.x - centre.x) + (c.z - centre.z) * (c.z - centre.z) > radius * radius)
                        continue;
                    var i = iz * nx + ix;
                    var wasPainted = !float.IsNaN(grid[i]);
                    grid[i] = value;
                    PaintedCells += (float.IsNaN(value) ? 0 : 1) - (wasPainted ? 1 : 0);
                    count++;
                }
            overlayDirty |= count > 0;
            return count;
        }

        // --- overlay -----------------------------------------------------------------------------------------

        void LateUpdate()
        {
            if (overlayDirty)
                RebuildOverlay();
        }

        /// <summary>Colour of a μ: blue (ice) through pale (normal) to orange (grippy).</summary>
        public static Color MuColour(float mu)
        {
            var ice = new Color(0.25f, 0.6f, 1f, 0.55f);
            var normal = new Color(0.85f, 0.9f, 0.95f, 0.35f);
            var grip = new Color(1f, 0.55f, 0.15f, 0.55f);
            return mu < 1f ? Color.Lerp(ice, normal, Mathf.InverseLerp(0.05f, 1f, mu)) : Color.Lerp(normal, grip, Mathf.InverseLerp(1f, 1.5f, mu));
        }

        public void RebuildOverlay()
        {
            overlayDirty = false;
            if (overlayMaterial == null)
                return;
            if (overlay == null)
            {
                overlay = new GameObject("FrictionOverlay", typeof(MeshFilter), typeof(MeshRenderer)) { hideFlags = HideFlags.DontSave };
                overlay.transform.SetParent(transform, false);
                overlay.GetComponent<MeshRenderer>().sharedMaterial = overlayMaterial;
                overlay.GetComponent<MeshRenderer>().shadowCastingMode = ShadowCastingMode.Off;
                overlayMesh = new Mesh { name = "FrictionOverlay", hideFlags = HideFlags.DontSave, indexFormat = IndexFormat.UInt32 };
                overlay.GetComponent<MeshFilter>().sharedMesh = overlayMesh;
            }

            var physics = gameObject.scene.GetPhysicsScene();
            var terrain = LayerMask.GetMask(SimSettings.TerrainLayer);
            var vertices = new List<Vector3>();
            var colours = new List<Color>();
            var triangles = new List<int>();
            var h = cellSize / 2f;
            for (var iz = 0; iz < nz; iz++)
                for (var ix = 0; ix < nx; ix++)
                {
                    var mu = grid[iz * nx + ix];
                    if (float.IsNaN(mu))
                        continue;
                    var c = CellCentre(ix, iz);
                    var y = physics.Raycast(c + Vector3.up * 50f, Vector3.down, out var hit, 100f, terrain, QueryTriggerInteraction.Ignore) ? hit.point.y : 0f;
                    var b = vertices.Count;
                    foreach (var (dx, dz) in new[] { (-h, -h), (-h, h), (h, h), (h, -h) })
                        vertices.Add(new Vector3(c.x + dx, y + 0.004f, c.z + dz) - transform.position);
                    var colour = MuColour(mu);
                    for (var k = 0; k < 4; k++)
                        colours.Add(colour);
                    triangles.AddRange(new[] { b, b + 1, b + 2, b, b + 2, b + 3 });
                }
            overlayMesh.Clear();
            overlayMesh.SetVertices(vertices);
            overlayMesh.SetColors(colours);
            overlayMesh.SetTriangles(triangles, 0);
            overlayMesh.RecalculateBounds();
        }

        void OnDestroy()
        {
            if (overlayMesh != null)
                DestroyImmediate(overlayMesh);
            if (overlay != null)
                DestroyImmediate(overlay);
        }
    }

    /// <summary>A blob of painted low-friction ground (Unity world centre; the radius covers every sample).</summary>
    public struct FrictionPatch
    {
        public Vector3 Centre;
        public float Radius;
        public float MinMu;
        public float MeanMu;
        public int Samples;
    }
}
