using NUnit.Framework;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BMB.Tests
{
    /// <summary>
    /// Height scan (docs/unity_handoff.md §5.6). Ray placement is checked against Isaac's recorded hit points (pure
    /// geometry, so it holds on any terrain); values are checked on synthetic terrain with known heights.
    /// </summary>
    public class HeightScannerTests
    {
        RobotInterface itf;
        int terrainMask;

        [OneTimeSetUp]
        public void Load()
        {
            itf = RobotInterface.Load();
            terrainMask = LayerMask.GetMask(SimSettings.TerrainLayer);
        }

        [TestCase("rough")]
        [TestCase("v2")]
        public void Ray_positions_match_isaac(string variant)
        {
            var r = ReferenceRollout.Load(variant);
            var scanner = new HeightScanner(itf.HeightScan, Physics.defaultPhysicsScene, terrainMask);
            var origins = new Vector3[scanner.Count];
            Assert.AreEqual(r.scan_hits_xy.RowSize / 2, scanner.Count);
            float maxErr = 0f;
            for (var t = 0; t < r.Steps; t += 10)
            {
                var basePos = IsaacFrame.PolarFromIsaac(new Vector3(r.root_link_pos_w[t, 0], r.root_link_pos_w[t, 1], r.root_link_pos_w[t, 2]));
                var baseRot = IsaacFrame.QuatFromIsaac(r.root_link_quat_w[t, 0], r.root_link_quat_w[t, 1], r.root_link_quat_w[t, 2], r.root_link_quat_w[t, 3]);
                scanner.RayOrigins(basePos, baseRot, origins);
                for (var k = 0; k < scanner.Count; k++)
                {
                    var o = IsaacFrame.PolarToIsaac(origins[k]);
                    var row = r.scan_hits_xy.data;
                    var i = (t * scanner.Count + k) * 2;
                    maxErr = Mathf.Max(maxErr, new Vector2(o.x - row[i], o.y - row[i + 1]).magnitude);
                }
            }
            Debug.Log($"[BMB] Height-scan ray positions {variant}: max xy error {maxErr * 1000:F4} mm");
            Assert.That(maxErr, Is.LessThan(1e-4f));
        }

        [Test]
        public void Values_on_known_terrain()
        {
            var scene = EditorSceneManager.NewPreviewScene();
            try
            {
                var terrain = LayerMask.NameToLayer(SimSettings.TerrainLayer);
                Box(scene, "ground", new Vector3(0f, -0.5f, 0f), new Vector3(20f, 1f, 20f), terrain);
                // A 0.2 m step on the robot's left (Isaac +y = Unity −x), from 0.25 m outwards.
                Box(scene, "step", new Vector3(-1.25f, 0.1f, 0f), new Vector3(2f, 0.2f, 4f), terrain);
                // Something on the Robot layer right under the rays: must be ignored.
                Box(scene, "robot part", new Vector3(0f, 0.3f, 0f), new Vector3(0.3f, 0.05f, 0.3f), LayerMask.NameToLayer(SimSettings.RobotLayer));
                Physics.SyncTransforms();

                var scanner = new HeightScanner(itf.HeightScan, scene.GetPhysicsScene(), terrainMask);
                var raw = new float[scanner.Count];
                var basePos = new Vector3(0f, 0.4f, 0f);
                scanner.Scan(basePos, Quaternion.identity, raw);

                var nx = itf.HeightScan.GridShape[0];
                for (var k = 0; k < scanner.Count; k++)
                {
                    var y = -itf.HeightScan.GridSize[1] / 2f + itf.HeightScan.Resolution * (k / nx);  // Isaac left offset
                    var groundZ = y > 0.25f ? 0.2f : 0f;
                    Assert.AreEqual(basePos.y - groundZ - itf.HeightScan.Offset, raw[k], 1e-5f, $"ray {k} (y offset {y:F2})");
                }

                // Same spot, robot turned 90° left (now facing Unity −x): the step is ahead of it (Isaac +x).
                scanner.Scan(basePos, Quaternion.AngleAxis(-90f, Vector3.up), raw);
                for (var k = 0; k < scanner.Count; k++)
                {
                    var x = -itf.HeightScan.GridSize[0] / 2f + itf.HeightScan.Resolution * (k % nx);  // Isaac forward offset
                    var groundZ = x > 0.25f ? 0.2f : 0f;
                    Assert.AreEqual(basePos.y - groundZ - itf.HeightScan.Offset, raw[k], 1e-5f, $"turned, ray {k} (x offset {x:F2})");
                }

                // Off the terrain: every ray misses (-inf), which the observation clip turns into -1.
                scanner.Scan(new Vector3(500f, 0.4f, 0f), Quaternion.identity, raw);
                Assert.IsTrue(System.Array.TrueForAll(raw, float.IsNegativeInfinity), "misses must be -inf");
            }
            finally
            {
                EditorSceneManager.ClosePreviewScene(scene);
            }
        }

        static void Box(Scene scene, string name, Vector3 centre, Vector3 size, int layer)
        {
            var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
            box.name = name;
            SceneManager.MoveGameObjectToScene(box, scene);
            box.transform.position = centre;
            box.transform.localScale = size;
            box.layer = layer;
        }
    }
}
