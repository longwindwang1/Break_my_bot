using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>Step 2 of the disturbance system: push, payload, friction zones (per-foot), global ice.</summary>
    public class BasicDisturbanceTests
    {
        static (HeadlessWorld world, Go2Controller robot, DisturbanceManager m) Setup(bool hard = false)
        {
            var world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero, footFriction: 1f);
            var m = world.AddDisturbances(robot, hard);
            world.Run(0.5f);  // settle on the ground
            return (world, robot, m);
        }

        [Test]
        public void Push_adds_dv_to_the_base_velocity()
        {
            var (world, robot, m) = Setup();
            using (world)
            {
                var root = robot.Body.Root;
                var before = root.linearVelocity;
                var applied = m.Push(new Vector2(0.4f, -0.2f));
                var expected = IsaacFrame.PolarFromIsaac(new Vector3(0.4f, -0.2f, 0f));
                Assert.That((root.linearVelocity - before - expected).magnitude, Is.LessThan(1e-4f));
                Assert.AreEqual(new Vector2(0.4f, -0.2f), applied);

                Assert.AreEqual(0.5f, m.Push(new Vector2(3f, 0f)).x, 1e-6f, "training range clamps per axis");
                Assert.IsFalse(m.Log.Events.Last().OutsideTraining);
                m.HardMode = true;
                Assert.AreEqual(1.5f, m.Push(new Vector2(1.5f, 0f)).x, 1e-6f);
                Assert.IsTrue(m.Log.Events.Last(e => e.Type == "push").OutsideTraining);
            }
        }

        [Test]
        public void Payload_changes_mass_inertia_and_com_and_is_removable()
        {
            var (world, robot, m) = Setup();
            using (world)
            {
                var body = robot.Body;
                var total = body.TotalMass;
                var nominalInertia = body.Root.inertiaTensor;
                m.SetPayload(2f, new Vector3(0.03f, -0.01f, 0f));
                Assert.AreEqual(total + 2f, body.TotalMass, 1e-4f);
                var ratio = (body.NominalBaseMass + 2f) / body.NominalBaseMass;
                Assert.That((body.Root.inertiaTensor - nominalInertia * ratio).magnitude, Is.LessThan(1e-6f), "inertia scaled by the mass ratio (Isaac)");
                var expectedCom = body.NominalBaseCom + IsaacFrame.PolarFromIsaac(new Vector3(0.03f, -0.01f, 0f));
                Assert.That((body.Root.centerOfMass - expectedCom).magnitude, Is.LessThan(1e-6f));

                m.SetPayload(5f, Vector3.zero);
                Assert.AreEqual(3f, body.AddedMass, 1e-6f, "training range: at most +3 kg");
                m.SetPayload(0f, Vector3.zero);
                Assert.AreEqual(total, body.TotalMass, 1e-4f, "removed");
                Assert.That((body.Root.centerOfMass - body.NominalBaseCom).magnitude, Is.LessThan(1e-6f));
            }
        }

        [Test]
        public void Friction_zones_paint_erase_and_global()
        {
            var zones = new GameObject("zones").AddComponent<FrictionZones>();
            try
            {
                zones.Paint(new Vector3(1f, 0f, 1f), 0.5f, 0.1f);
                Assert.AreEqual(0.1f, zones.MuAt(new Vector3(1.1f, 0f, 0.9f)), 1e-6f);
                Assert.AreEqual(1f, zones.MuAt(new Vector3(3f, 0f, 3f)), 1e-6f, "unpainted = global (1.0)");
                zones.GlobalMu = 0.4f;
                Assert.AreEqual(0.4f, zones.MuAt(new Vector3(3f, 0f, 3f)), 1e-6f);
                Assert.AreEqual(0.1f, zones.MuAt(new Vector3(1f, 0f, 1f)), 1e-6f, "painted cells override the global μ");
                Assert.Greater(zones.PaintedCells, 50);
                zones.Erase(new Vector3(1f, 0f, 1f), 0.6f);
                Assert.AreEqual(0, zones.PaintedCells);
                Assert.AreEqual(0.4f, zones.MuAt(new Vector3(1f, 0f, 1f)), 1e-6f);
            }
            finally
            {
                Object.DestroyImmediate(zones.gameObject);
            }
        }

        [Test]
        public void Ice_under_one_foot_changes_only_that_foot()
        {
            var (world, robot, m) = Setup();
            using (world)
            {
                m.PaintFriction(robot.Body.FootPosition(0), 0.08f, 0.1f);
                world.Step();
                var ratio = world.Interface.Friction.FootDynamicRatio;
                Assert.AreEqual(0.1f, robot.Body.FootStaticFriction(0), 1e-6f, "FL on ice: foot μ 1.0 × ground 0.1");
                for (var i = 1; i < 4; i++)
                    Assert.AreEqual(1f, robot.Body.FootStaticFriction(i), 1e-6f, $"{Go2Body.FootNames[i]} on normal ground");
                Assert.AreEqual(0.1f, robot.FootFriction.EffectiveStatic(0), 1e-6f);
                Assert.AreEqual(0.1f * ratio, m.Zones.MuAt(robot.Body.FootPosition(0)) * ratio, 1e-6f);
            }
        }

        [Test]
        public void Same_push_slides_further_on_ice()
        {
            float Slide(float groundMu)
            {
                var (world, robot, m) = Setup();
                using (world)
                {
                    m.SetGlobalFriction(groundMu);
                    world.Run(0.5f);
                    var start = IsaacFrame.PolarToIsaac(robot.Body.BasePosition);
                    m.Push(new Vector2(0f, 0.5f));  // sideways
                    world.Run(1f);
                    return IsaacFrame.PolarToIsaac(robot.Body.BasePosition).y - start.y;
                }
            }

            var normal = Slide(1f);
            var ice = Slide(0.05f);
            Debug.Log($"[BMB] Sideways push 0.5 m/s, displacement after 1 s: μ 1.0 → {normal:F3} m, μ 0.05 → {ice:F3} m");
            Assert.Greater(ice, normal + 0.02f, "on ice the robot slides further");
        }

        [TestCase(false)]
        [TestCase(true)]
        public void V2_keeps_walking_on_ice_and_through_a_friction_drop(bool drop)
        {
            var (world, robot, m) = Setup();
            using (world)
            {
                robot.CommandOverride = new Vector3(0.5f, 0f, 0f);
                if (!drop)
                    m.SetGlobalFriction(0.1f);
                var t = 0f;
                world.Run(10f, () =>
                {
                    t += world.Interface.Timing.PhysicsDt;
                    if (drop && Mathf.Abs(t - 5f) < 1e-4f)
                        m.SetGlobalFriction(0.1f);  // Isaac's mu_drop: 1.0 → 0.1 at 5 s
                });
                Debug.Log($"[BMB] v2 {(drop ? "μ 1.0 → 0.1 at 5 s" : "on μ 0.1")}: falls {robot.Falls}, travelled {robot.Displacement.x:F2} m");
                Assert.AreEqual(0, robot.Falls);
                Assert.Greater(robot.Displacement.x, 2f, "made progress");
            }
        }
    }
}
