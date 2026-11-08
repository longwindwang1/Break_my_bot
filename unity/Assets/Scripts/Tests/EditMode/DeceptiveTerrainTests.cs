using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>Step 4 of the disturbance system: hidden pits, breakable ice, the deceptive scan mode.</summary>
    public class DeceptiveTerrainTests
    {
        static float Ray(HeadlessWorld w, float x, float z, int mask)
        {
            Assert.IsTrue(w.Physics.Raycast(new Vector3(x, 5f, z), Vector3.down, out var hit, 20f, mask), $"no hit at ({x}, {z})");
            return hit.point.y;
        }

        static int Terrain => LayerMask.GetMask(SimSettings.TerrainLayer);

        /// <summary>Raw scan values of the rays over a box ahead of the robot (Isaac x 0.6–0.8 m, |y| ≤ 0.2 m).</summary>
        static float[] RaysAhead(Go2Controller robot, HeadlessWorld w)
        {
            var raw = new float[robot.Scanner.Count];
            robot.Scanner.Scan(robot.Body.BasePosition, robot.Body.BaseRotation, raw);
            var nx = w.Interface.HeightScan.GridShape[0];
            return Enumerable.Range(0, raw.Length)
                .Where(k => -0.8f + 0.1f * (k % nx) >= 0.55f && Mathf.Abs(-0.5f + 0.1f * (k / nx)) <= 0.25f)
                .Select(k => raw[k]).ToArray();
        }

        [TestCase(true)]
        [TestCase(false)]
        public void Hidden_pit_fools_the_scan_but_not_the_feet(bool hidden)
        {
            using var w = HeadlessWorld.Create(flatGround: false);
            var level = w.AddLevelGround(new Vector2(12f, 12f));
            var robot = w.AddRobot("v2", Vector3.zero, footFriction: 1f);
            var m = w.AddDisturbances(robot, level: level);
            m.PlaceObstacle(new GroundFeature { type = hidden ? GroundFeatureType.HiddenPit : GroundFeatureType.Pit, size = new Vector2(0.8f, 0.8f), height = 0.15f }, new Vector3(0f, 0f, 0.9f));
            UnityEngine.Physics.SyncTransforms();

            Assert.AreEqual(-0.15f, Ray(w, 0f, 0.9f, Terrain), 1e-4f, "the pit is real for physics");
            Assert.AreEqual(hidden ? 0f : -0.15f, Ray(w, 0f, 0.9f, SimSettings.ScanMask), 1e-4f, "what the scan rays hit");

            for (var i = 0; i < 4; i++)
                w.Step();
            var baseY = robot.Body.BasePosition.y;
            var offset = w.Interface.HeightScan.Offset;
            foreach (var v in RaysAhead(robot, w))
                Assert.AreEqual(hidden ? baseY - offset : baseY + 0.15f - offset, v, 0.02f, hidden ? "scan sees flat ground" : "scan sees the pit");

            // Walk into it: a foot must go below ground level only if it is there physically (it is, in both cases).
            robot.CommandOverride = new Vector3(0.5f, 0f, 0f);
            var lowestFoot = float.MaxValue;
            w.Run(5f, () =>
            {
                for (var f = 0; f < 4; f++)
                    lowestFoot = Mathf.Min(lowestFoot, robot.Body.FootPosition(f).y);
            });
            Debug.Log($"[BMB] {(hidden ? "Hidden" : "Visible")} 15 cm pit: lowest foot {lowestFoot:F3} m, falls {robot.Falls}");
            Assert.Less(lowestFoot, -0.05f, "a foot went into the pit");
        }

        [Test]
        public void Breakable_ice_holds_then_breaks_and_can_be_restored()
        {
            using var w = HeadlessWorld.Create(flatGround: false);
            var level = w.AddLevelGround(new Vector2(12f, 12f));
            // Ice under the spawn point, placed before the robot so it spawns standing on it.
            var tmpRobot = default(Go2Controller);
            var probe = w.Add(new GameObject("probe")).AddComponent<DisturbanceManager>();
            probe.Configure(null, DisturbanceLimits.CreateDefault(), null, null, level);
            probe.PlaceObstacle(new GroundFeature { type = GroundFeatureType.BreakableIce, size = new Vector2(1.4f, 1.4f), height = 0.15f, breakAfter = 0.4f }, Vector3.zero);
            UnityEngine.Physics.SyncTransforms();
            Assert.AreEqual(0f, Ray(w, 0f, 0f, Terrain), 1e-4f, "ice is solid");

            tmpRobot = w.AddRobot("v2", Vector3.zero, footFriction: 1f);
            var m = w.AddDisturbances(tmpRobot, level: level);
            var cover = level.Covers.Single();

            w.Run(0.3f);
            Assert.IsFalse(cover.Broken, "holds for a moment");
            Assert.Greater(tmpRobot.BaseHeight, 0.25f, "standing on the ice");
            Assert.Greater(cover.ContactTime, 0f, "feet touching the slab");

            var lowest = float.MaxValue;
            w.Run(1.5f, () => lowest = Mathf.Min(lowest, tmpRobot.Body.BasePosition.y));
            Assert.IsTrue(cover.Broken, "broke after 0.4 s of contact");
            Assert.AreEqual(-0.15f, Ray(w, 0f, 0f, Terrain), 1e-4f, "the hole is open");
            Assert.AreEqual(-0.15f, Ray(w, 0f, 0f, SimSettings.ScanMask), 1e-4f, "and the scan sees it");
            Assert.Less(lowest, 0.25f, "the robot dropped");
            var broke = m.Log.Events.Single(e => e.Type == "deceptive" && e.Detail.StartsWith("ice broke"));
            Debug.Log($"[BMB] Breakable ice: {broke.Detail} at {broke.Time:F2} s; base went down to {lowest:F3} m");

            m.RestoreIce();
            UnityEngine.Physics.SyncTransforms();
            Assert.IsFalse(cover.Broken);
            Assert.AreEqual(0f, Ray(w, 2f, 2f, Terrain), 1e-4f);
        }

        [Test]
        public void Removing_a_hidden_pit_removes_its_cover()
        {
            using var w = HeadlessWorld.Create(flatGround: false);
            var level = w.AddLevelGround(new Vector2(10f, 10f));
            var id = level.Add(new GroundFeature { type = GroundFeatureType.HiddenPit, centre = new Vector2(1f, 1f), size = new Vector2(0.6f, 0.6f), height = 0.1f });
            Assert.AreEqual(1, level.Covers.Count);
            level.Remove(id);
            Assert.AreEqual(0, level.Covers.Count);
            UnityEngine.Physics.SyncTransforms();
            Assert.AreEqual(0f, Ray(w, 1f, 1f, SimSettings.ScanMask), 1e-4f);
        }

        [Test]
        public void Deceive_mode_reports_the_median_everywhere()
        {
            using var w = HeadlessWorld.Create(flatGround: false);
            var level = w.AddLevelGround(new Vector2(10f, 10f));
            level.Add(new GroundFeature { type = GroundFeatureType.Pit, centre = new Vector2(0f, 0.6f), size = new Vector2(1f, 0.4f), height = 0.15f });
            UnityEngine.Physics.SyncTransforms();
            var itf = w.Interface;
            var scanner = new HeightScanner(itf.HeightScan, w.Physics, SimSettings.ScanMask);
            var raw = new float[scanner.Count];
            var basePos = new Vector3(0f, 0.4f, 0f);
            scanner.Scan(basePos, Quaternion.identity, raw);
            Assert.Greater(raw.Distinct().Count(), 1, "honest scan sees the pit");
            scanner.Deceive = true;
            scanner.Scan(basePos, Quaternion.identity, raw);
            Assert.AreEqual(1, raw.Distinct().Count(), "deceived scan is flat");
            Assert.AreEqual(basePos.y - itf.HeightScan.Offset, raw[0], 1e-5f, "at the median level (most rays see ground)");
        }

        [Test]
        public void Deceptive_tool_controls()
        {
            using var w = HeadlessWorld.Create(flatGround: false);
            var level = w.AddLevelGround(new Vector2(10f, 10f));
            var robot = w.AddRobot("v2", Vector3.zero, footFriction: 1f);
            var m = w.AddDisturbances(robot, level: level);
            var tool = (DeceptiveTool)m.Tools.Single(t => t is DeceptiveTool);
            Assert.IsTrue(tool.Available);
            var controls = tool.BuildControls(m);
            var breaks = controls.OfType<SliderControl>().Single(c => c.Label == "Breaks after");
            tool.Kind = 0;
            Assert.IsFalse(breaks.Visible(), "hidden pit has no break delay");
            tool.Kind = 1;
            Assert.IsTrue(breaks.Visible());
            controls.OfType<SliderControl>().Single(c => c.Label == "Depth").SetClamped(50f);
            Assert.AreEqual(0.16f, tool.Depth, 1e-5f, "training range");
            var reveal = controls.OfType<ToggleControl>().Single();
            reveal.Set(true);
            Assert.IsTrue(level.RevealHidden);
        }
    }
}
