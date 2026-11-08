using System;
using UnityEngine;

namespace BMB
{
    /// <summary>Isaac Lab sub-terrains used by the Go2 rough / v2 training mix.</summary>
    public enum IsaacTileType
    {
        PyramidStairs,
        PyramidStairsInv,
        Boxes,
        RandomRough,
        PyramidSlope,
        PyramidSlopeInv,
        Pits,
    }

    /// <summary>
    /// 8 m × 8 m terrain tiles generated like Isaac Lab, as 81 × 81 vertex heights on a 0.1 m grid (index [i, j]:
    /// i along the tile's x, j along its y). Sources (local Isaac Lab clone, same as the Linux side's 0.54.2):
    /// terrains/config/rough.py ROUGH_TERRAINS_CFG + the Go2 rough overrides (boxes 0.025–0.1 m, rough 0.01–0.06 m
    /// step 0.01) + v2's pits (isaac/bmb_tasks/go2_v2_env_cfg.py).
    /// - Heightfield terrains (random_rough, pyramid slopes, pits = discrete_obstacles) are exact ports, integer units
    ///   of vertical_scale 0.005 m and Python's float truncation included; they are "soft" (Isaac's slope threshold).
    /// - Mesh terrains (pyramid stairs, random grid boxes) are rasterized on 0.1 m cells; a vertex takes the highest
    ///   of its four cells, so with vertical walls ("hard") the steps sit on cell edges. Stair rings (0.3 m) are exact;
    ///   the 0.45 m box grid is approximated to the nearest 0.1 m.
    /// Random draws use System.Random: same distributions as numpy, different samples.
    /// </summary>
    public static class IsaacTiles
    {
        public const float Size = 8f;
        public const float Cell = 0.1f;
        /// <summary>horizontal_scale as Python sees it (the double 0.1): integer conversions must match numpy exactly.</summary>
        const double HorizontalScale = 0.1;
        public const double VerticalScale = 0.005;
        public const int Vertices = 81;  // int(8 / 0.1) + 1

        /// <summary>v2 training mix: ROUGH_TERRAINS_CFG proportions plus pits 0.15.</summary>
        public static readonly (IsaacTileType type, float proportion)[] V2Mix =
        {
            (IsaacTileType.PyramidStairs, 0.2f),
            (IsaacTileType.PyramidStairsInv, 0.2f),
            (IsaacTileType.Boxes, 0.2f),
            (IsaacTileType.RandomRough, 0.2f),
            (IsaacTileType.PyramidSlope, 0.1f),
            (IsaacTileType.PyramidSlopeInv, 0.1f),
            (IsaacTileType.Pits, 0.15f),
        };

        /// <summary>Mesh terrains in Isaac (vertical walls).</summary>
        public static bool IsHard(IsaacTileType t) => t is IsaacTileType.PyramidStairs or IsaacTileType.PyramidStairsInv or IsaacTileType.Boxes;

        public static IsaacTileType Sample(System.Random rng, (IsaacTileType type, float proportion)[] mix)
        {
            var total = 0f;
            foreach (var (_, p) in mix) total += p;
            var r = (float)rng.NextDouble() * total;
            foreach (var (t, p) in mix)
            {
                if (r < p) return t;
                r -= p;
            }
            return mix[mix.Length - 1].type;
        }

        /// <summary>Vertex heights (m) of one tile at a difficulty in [0, 1].</summary>
        public static float[,] Generate(IsaacTileType type, float difficulty, System.Random rng) => type switch
        {
            IsaacTileType.RandomRough => Heightfield(0.25, (w, l) => RandomUniform(w, l, rng)),
            IsaacTileType.PyramidSlope => Heightfield(0.25, (w, l) => PyramidSloped(w, l, difficulty, false)),
            IsaacTileType.PyramidSlopeInv => Heightfield(0.25, (w, l) => PyramidSloped(w, l, difficulty, true)),
            IsaacTileType.Pits => Heightfield(0.0, (w, l) => DiscreteObstacles(w, l, difficulty, rng)),
            IsaacTileType.PyramidStairs => FromCells(PyramidStairs(difficulty, false)),
            IsaacTileType.PyramidStairsInv => FromCells(PyramidStairs(difficulty, true)),
            IsaacTileType.Boxes => FromCells(RandomGrid(difficulty, rng)),
            _ => throw new ArgumentOutOfRangeException(nameof(type)),
        };

        // --- heightfield terrains (height_field/hf_terrains.py, wrapped like height_field_to_mesh) ------------

        /// <summary>height_field_to_mesh: zero border of int(border / 0.1) + 1 vertices, generator inside.</summary>
        static float[,] Heightfield(double borderWidth, Func<int, int, int[,]> generator)
        {
            var border = (int)(borderWidth / HorizontalScale) + 1;
            var inner = Vertices - 2 * border;
            var units = generator(inner, inner);
            var h = new float[Vertices, Vertices];
            for (var i = 0; i < inner; i++)
                for (var j = 0; j < inner; j++)
                    h[border + i, border + j] = (float)(units[i, j] * VerticalScale);
            return h;
        }

        /// <summary>random_uniform_terrain, Go2 values: noise (0.01, 0.06) step 0.01, downsampled = horizontal scale.</summary>
        static int[,] RandomUniform(int w, int l, System.Random rng)
        {
            int min = (int)(0.01 / VerticalScale), max = (int)(0.06 / VerticalScale), step = (int)(0.01 / VerticalScale);
            var levels = (max - min) / step + 1;  // np.arange(min, max + step, step)
            var u = new int[w, l];
            for (var i = 0; i < w; i++)
                for (var j = 0; j < l; j++)
                    u[i, j] = min + step * rng.Next(levels);
            return u;
        }

        /// <summary>pyramid_sloped_terrain: slope_range (0, 0.4), platform 2.0 m.</summary>
        static int[,] PyramidSloped(int w, int l, float difficulty, bool inverted)
        {
            var slope = inverted ? -(0.0 + difficulty * 0.4) : 0.0 + difficulty * 0.4;
            var sizeX = w * HorizontalScale;
            var heightMax = (int)(slope * sizeX / 2 / VerticalScale);
            int cx = w / 2, cy = l / 2;
            var raw = new double[w, l];
            for (var i = 0; i < w; i++)
                for (var j = 0; j < l; j++)
                    raw[i, j] = heightMax * ((cx - Math.Abs(cx - i)) / (double)cx) * ((cy - Math.Abs(cy - j)) / (double)cy);
            var platform = (int)(2.0 / HorizontalScale / 2);
            var zPf = raw[w / 2 - platform, l / 2 - platform];
            double lo = Math.Min(0, zPf), hi = Math.Max(0, zPf);
            var u = new int[w, l];
            for (var i = 0; i < w; i++)
                for (var j = 0; j < l; j++)
                    u[i, j] = (int)Math.Round(Math.Clamp(raw[i, j], lo, hi), MidpointRounding.ToEven);
            return u;
        }

        /// <summary>discrete_obstacles_terrain, v2 pits: widths (0.3, 0.8), heights (0.04, 0.16) "choice", 30 obstacles, platform 1.5 m.</summary>
        static int[,] DiscreteObstacles(int w, int l, float difficulty, System.Random rng)
        {
            var obsHeight = 0.04 + difficulty * (0.16 - 0.04);
            var hUnits = (int)(obsHeight / VerticalScale);
            int wMin = (int)(0.3 / HorizontalScale), wMax = (int)(0.8 / HorizontalScale);  // Python: 2 and 8
            var widths = Arange(wMin, wMax, 4);
            var xs = Arange(0, w, 4);
            var ys = Arange(0, l, 4);
            var choices = new[] { -hUnits, FloorDiv(-hUnits, 2), FloorDiv(hUnits, 2), hUnits };
            var u = new int[w, l];
            for (var n = 0; n < 30; n++)
            {
                var height = choices[rng.Next(choices.Length)];
                var width = widths[rng.Next(widths.Length)];
                var length = widths[rng.Next(widths.Length)];
                var x0 = xs[rng.Next(xs.Length)];
                var y0 = ys[rng.Next(ys.Length)];
                if (x0 + width > w) x0 = w - width;
                if (y0 + length > l) y0 = l - length;
                for (var i = x0; i < x0 + width; i++)
                    for (var j = y0; j < y0 + length; j++)
                        u[i, j] = height;
            }
            var pf = (int)(1.5 / HorizontalScale);
            for (var i = (w - pf) / 2; i < (w + pf) / 2; i++)
                for (var j = (l - pf) / 2; j < (l + pf) / 2; j++)
                    u[i, j] = 0;
            return u;
        }

        // --- mesh terrains, rasterized on cells (trimesh/mesh_terrains.py) ---------------------------------------

        const int Cells = Vertices - 1;  // 80 cells of 0.1 m

        static Vector2 CellCentre(int i, int j) => new Vector2((i + 0.5f) * Cell, (j + 0.5f) * Cell);

        /// <summary>pyramid_stairs_terrain (+ inverted): steps 0.05–0.23 m, step width 0.3, platform 3.0, border 1.0.</summary>
        static float[,] PyramidStairs(float difficulty, bool inverted)
        {
            var stepHeight = 0.05f + difficulty * (0.23f - 0.05f);
            const double border = 1.0, platform = 3.0, stepWidth = 0.3;
            var numSteps = (int)(Math.Floor((Size - 2 * border - platform) / (2 * stepWidth)) + 1);  // 6
            var c = new float[Cells, Cells];
            for (var i = 0; i < Cells; i++)
                for (var j = 0; j < Cells; j++)
                {
                    var p = CellCentre(i, j);
                    var dist = Math.Min(Math.Min(p.x - border, Size - border - p.x), Math.Min(p.y - border, Size - border - p.y));
                    if (dist < 0)
                        continue;  // border at height 0
                    var k = (int)Math.Floor(dist / stepWidth);
                    var h = k < numSteps ? (k + 1) * stepHeight : (numSteps + 1) * stepHeight;
                    c[i, j] = inverted ? -h : h;
                }
            return c;
        }

        /// <summary>random_grid_terrain, Go2 values: grid 0.45 m, heights ±(0.025–0.1) m, platform 2.0 m at +g.</summary>
        static float[,] RandomGrid(float difficulty, System.Random rng)
        {
            var g = 0.025f + difficulty * (0.1f - 0.025f);
            const double grid = 0.45;
            var n = (int)(Size / grid);  // 17
            var border = Size - n * grid;  // 0.35
            var boxes = new float[n, n];
            for (var a = 0; a < n; a++)
                for (var b = 0; b < n; b++)
                    boxes[a, b] = (float)((rng.NextDouble() * 2 - 1) * g);
            var c = new float[Cells, Cells];
            for (var i = 0; i < Cells; i++)
                for (var j = 0; j < Cells; j++)
                {
                    var p = CellCentre(i, j);
                    if (Mathf.Abs(p.x - Size / 2) < 1f && Mathf.Abs(p.y - Size / 2) < 1f)
                    {
                        c[i, j] = g;  // central platform
                        continue;
                    }
                    var a = (int)Math.Floor((p.x - border / 2) / grid);
                    var b = (int)Math.Floor((p.y - border / 2) / grid);
                    if (a >= 0 && b >= 0 && a < n && b < n)
                        c[i, j] = boxes[a, b];
                }
            return c;
        }

        /// <summary>Vertex heights from cell heights: each vertex takes the highest of its (up to four) cells.</summary>
        static float[,] FromCells(float[,] cells)
        {
            var h = new float[Vertices, Vertices];
            for (var i = 0; i < Vertices; i++)
                for (var j = 0; j < Vertices; j++)
                {
                    var best = float.NegativeInfinity;
                    for (var di = -1; di <= 0; di++)
                        for (var dj = -1; dj <= 0; dj++)
                        {
                            int ci = i + di, cj = j + dj;
                            if (ci >= 0 && cj >= 0 && ci < Cells && cj < Cells)
                                best = Mathf.Max(best, cells[ci, cj]);
                        }
                    h[i, j] = best;
                }
            return h;
        }

        static int[] Arange(int start, int stop, int step)
        {
            var count = Math.Max(0, (stop - start + step - 1) / step);
            var a = new int[count];
            for (var k = 0; k < count; k++)
                a[k] = start + k * step;
            return a;
        }

        static int FloorDiv(int a, int b) => (int)Math.Floor(a / (double)b);
    }
}
