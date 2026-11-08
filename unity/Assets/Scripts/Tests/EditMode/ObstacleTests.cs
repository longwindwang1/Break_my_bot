using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>Step 3 of the disturbance system: editable ground, obstacles, Isaac tiles.</summary>
    public class ObstacleTests
    {
        static float GroundY(HeadlessWorld w, float x, float z)
        {
            Assert.IsTrue(w.Physics.Raycast(new Vector3(x, 5f, z), Vector3.down, out var hit, 20f, LayerMask.GetMask(SimSettings.TerrainLayer)),
                $"no ground at ({x}, {z})");
            return hit.point.y;
        }

        [TestCase(GroundFeatureType.Pit, -0.1f)]
        [TestCase(GroundFeatureType.Box, 0.1f)]
        public void Block_and_pit_walls_are_vertical_on_the_footprint_edge(GroundFeatureType type, float top)
        {
            using var w = HeadlessWorld.Create(flatGround: false);
            var level = w.AddLevelGround(new Vector2(10f, 10f));
            level.Add(new GroundFeature { type = type, centre = Vector2.zero, size = new Vector2(1f, 1f), height = 0.1f });
            UnityEngine.Physics.SyncTransforms();
            foreach (var (x, z) in new[] { (0.49f, 0f), (-0.49f, 0f), (0f, 0.49f), (0f, -0.49f), (0.2f, 0.3f) })
                Assert.AreEqual(top, GroundY(w, x, z), 1e-4f, $"inside at ({x}, {z})");
            foreach (var (x, z) in new[] { (0.51f, 0f), (-0.51f, 0f), (0f, 0.51f), (0f, -0.51f), (2f, 2f) })
                Assert.AreEqual(0f, GroundY(w, x, z), 1e-4f, $"outside at ({x}, {z})");
        }

        [Test]
        public void Removing_a_feature_restores_the_ground()
        {
            using var w = HeadlessWorld.Create(flatGround: false);
            var level = w.AddLevelGround(new Vector2(10f, 10f));
            var id = level.Add(new GroundFeature { type = GroundFeatureType.Pit, centre = new Vector2(1f, -1f), size = new Vector2(0.6f, 0.6f), height = 0.15f });
            UnityEngine.Physics.SyncTransforms();
            Assert.AreEqual(-0.15f, GroundY(w, 1f, -1f), 1e-4f);
            Assert.AreEqual(id, level.FeatureAt(new Vector3(1f, 0f, -1f)).id);
            Assert.IsTrue(level.Remove(id));
            UnityEngine.Physics.SyncTransforms();
            Assert.AreEqual(0f, GroundY(w, 1f, -1f), 1e-4f);
            Assert.IsNull(level.FeatureAt(new Vector3(1f, 0f, -1f)));
        }

        [Test]
        public void Stairs_rise_by_one_step_per_tread()
        {
            using var w = HeadlessWorld.Create(flatGround: false);
            var level = w.AddLevelGround(new Vector2(10f, 10f));
            var length = GroundFeature.NaturalLength(GroundFeatureType.Stairs, 3, 0f);  // 2 * 3 * 0.3 + 1 = 2.8 m
            level.Add(new GroundFeature { type = GroundFeatureType.Stairs, centre = Vector2.zero, size = new Vector2(1f, length), height = 0.08f, steps = 3 });
            UnityEngine.Physics.SyncTransforms();
            var start = -length / 2f;
            Assert.AreEqual(0f, GroundY(w, 0f, start - 0.05f), 1e-4f, "before the stairs");
            for (var k = 0; k < 3; k++)
                Assert.AreEqual((k + 1) * 0.08f, GroundY(w, 0f, start + (k + 0.5f) * 0.3f), 1e-4f, $"tread {k}");
            Assert.AreEqual(0.24f, GroundY(w, 0f, 0f), 1e-4f, "plateau");
            Assert.AreEqual(0.08f, GroundY(w, 0f, -start - 0.15f), 1e-4f, "last tread on the way down");
        }

        [Test]
        public void Isaac_tiles_match_the_isaac_parameters()
        {
            var rng = new System.Random(7);
            float[] Flat(float[,] a) => a.Cast<float>().ToArray();
            bool Multiple(float v, float of) => Mathf.Abs(v / of - Mathf.Round(v / of)) < 1e-3f;

            var rough = Flat(IsaacTiles.Generate(IsaacTileType.RandomRough, 0.5f, rng));
            Assert.IsTrue(rough.All(v => v == 0f || (v >= 0.01f - 1e-6f && v <= 0.06f + 1e-6f && Multiple(v, 0.01f))), "random_rough: 0 border, else 0.01–0.06 in 0.01 steps");

            var pits = Flat(IsaacTiles.Generate(IsaacTileType.Pits, 0.5f, rng));  // h = 0.10: choices ±0.10, ±0.05
            Assert.IsTrue(pits.All(v => new[] { 0f, 0.05f, 0.1f }.Any(c => Mathf.Abs(Mathf.Abs(v) - c) < 1e-6f)), "pits heights");
            Assert.AreEqual(0f, IsaacTiles.Generate(IsaacTileType.Pits, 1f, rng)[40, 40], "central platform");

            var stairs = IsaacTiles.Generate(IsaacTileType.PyramidStairs, 0.5f, rng);  // step 0.14 m, 6 steps
            Assert.AreEqual(7 * 0.14f, stairs[40, 40], 1e-5f, "top platform at (n + 1) step");
            Assert.AreEqual(0f, stairs[5, 40], 1e-6f, "1 m border at 0");
            Assert.IsTrue(Flat(stairs).All(v => Multiple(v, 0.14f)));
            Assert.AreEqual(-7 * 0.14f, IsaacTiles.Generate(IsaacTileType.PyramidStairsInv, 0.5f, rng)[40, 40], 1e-5f);

            var boxes = Flat(IsaacTiles.Generate(IsaacTileType.Boxes, 0.5f, rng));  // g = 0.0625
            Assert.IsTrue(boxes.All(v => Mathf.Abs(v) <= 0.0625f + 1e-6f));
            Assert.AreEqual(0.0625f, IsaacTiles.Generate(IsaacTileType.Boxes, 0.5f, rng)[40, 40], 1e-6f, "platform at +g");

            var slope = Flat(IsaacTiles.Generate(IsaacTileType.PyramidSlope, 0.5f, rng));
            Assert.IsTrue(slope.All(v => v >= 0f && v <= 0.4f + 1e-3f), $"pyramid slope 0..0.4 m, max {slope.Max()}");
            Assert.Greater(slope.Max(), 0.35f);
            var inv = Flat(IsaacTiles.Generate(IsaacTileType.PyramidSlopeInv, 0.5f, rng));
            Assert.IsTrue(inv.All(v => v <= 0f && v >= -0.4f - 1e-3f));
        }

        [Test]
        public void Isaac_stairs_tile_has_vertical_risers()
        {
            using var w = HeadlessWorld.Create(flatGround: false);
            var level = w.AddLevelGround(new Vector2(12f, 12f));
            level.Add(new GroundFeature { type = GroundFeatureType.IsaacTile, tile = IsaacTileType.PyramidStairs, difficulty = 0.5f, centre = Vector2.zero, size = new Vector2(8f, 8f) });
            UnityEngine.Physics.SyncTransforms();
            // Tile x from -4: border to -3, first riser at -3 (0.14), second at -2.7 (0.28).
            Assert.AreEqual(0f, GroundY(w, -3.01f, 0f), 1e-4f);
            Assert.AreEqual(0.14f, GroundY(w, -2.99f, 0f), 1e-4f);
            Assert.AreEqual(0.14f, GroundY(w, -2.71f, 0f), 1e-4f);
            Assert.AreEqual(0.28f, GroundY(w, -2.69f, 0f), 1e-4f);
        }

        [Test]
        public void Height_scan_sees_a_block_ahead()
        {
            using var w = HeadlessWorld.Create(flatGround: false);
            var level = w.AddLevelGround(new Vector2(10f, 10f));
            // Block from 0.5 to 1.5 m ahead (Unity +z = the robot's forward), 1 m wide, 0.1 m high.
            level.Add(new GroundFeature { type = GroundFeatureType.Box, centre = new Vector2(0f, 1f), size = new Vector2(1f, 1f), height = 0.1f });
            var robot = w.AddRobot("v2", Vector3.zero, footFriction: 1f);
            for (var i = 0; i < 4; i++)
                w.Step();
            var itf = w.Interface;
            var raw = new float[robot.Scanner.Count];
            robot.Scanner.Scan(robot.Body.BasePosition, robot.Body.BaseRotation, raw);
            var baseY = robot.Body.BasePosition.y;
            int nx = itf.HeightScan.GridShape[0], seen = 0;
            for (var k = 0; k < raw.Length; k++)
            {
                float x = -0.8f + 0.1f * (k % nx), y = -0.5f + 0.1f * (k / nx);
                if (x >= 0.65f && Mathf.Abs(y) <= 0.35f)
                {
                    Assert.AreEqual(baseY - 0.1f - itf.HeightScan.Offset, raw[k], 0.02f, $"ray {k} over the block");
                    seen++;
                }
                if (x <= 0.3f)
                    Assert.AreEqual(baseY - itf.HeightScan.Offset, raw[k], 0.02f, $"ray {k} on the ground");
            }
            Assert.Greater(seen, 10);
        }

        [Test]
        public void V2_steps_onto_a_10cm_block_without_falling()
        {
            using var w = HeadlessWorld.Create(flatGround: false);
            var level = w.AddLevelGround(new Vector2(20f, 20f));
            level.Add(new GroundFeature { type = GroundFeatureType.Box, centre = new Vector2(0f, 2f), size = new Vector2(3f, 1.4f), height = 0.1f });
            var robot = w.AddRobot("v2", Vector3.zero, footFriction: 1f, command: new Vector3(0.5f, 0f, 0f));
            var maxY = 0f;
            w.Run(10f, () => maxY = Mathf.Max(maxY, robot.Body.BasePosition.y));
            Debug.Log($"[BMB] v2 over a 0.10 m block: falls {robot.Falls}, travelled {robot.Displacement.x:F2} m, max base y {maxY:F3} m");
            Assert.AreEqual(0, robot.Falls);
            Assert.Greater(robot.Displacement.x, 3f, "crossed the block (it ends 2.7 m ahead)");
        }
    }
}
