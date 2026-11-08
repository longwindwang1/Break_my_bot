using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>Step 1 of the disturbance system: limits, event log, fall detection, manager wiring.</summary>
    public class DisturbanceFrameworkTests
    {
        [Test]
        public void Training_mode_clamps_to_the_training_range()
        {
            var limits = DisturbanceLimits.CreateDefault();
            Assert.AreEqual(0.5f, limits.Clamp(Quantity.PushAxis, 3f, false, out var outside), 1e-6f);
            Assert.IsFalse(outside);
            Assert.AreEqual(0.05f, limits.Clamp(Quantity.Friction, 0f, false, out outside), 1e-6f);
            Assert.IsFalse(outside);
            Assert.AreEqual(-1f, limits.Clamp(Quantity.Payload, -5f, false, out _), 1e-6f);
            Assert.AreEqual(0.4f, limits.Clamp(Quantity.Slope, 1f, false, out _), 1e-6f);
        }

        [Test]
        public void Hard_mode_allows_more_and_flags_it()
        {
            var limits = DisturbanceLimits.CreateDefault();
            Assert.AreEqual(1.5f, limits.Clamp(Quantity.PushAxis, 1.5f, true, out var outside), 1e-6f);
            Assert.IsTrue(outside, "1.5 m/s is beyond the 0.5 m/s training range");
            Assert.AreEqual(0.3f, limits.Clamp(Quantity.PushAxis, 0.3f, true, out outside), 1e-6f);
            Assert.IsFalse(outside, "inside the training range even in hard mode");
            Assert.AreEqual(2f, limits.Clamp(Quantity.PushAxis, 9f, true, out _), 1e-6f, "hard limit");
            Assert.AreEqual(0f, limits.Clamp(Quantity.Friction, -1f, true, out outside), 1e-6f);
            Assert.IsTrue(outside);
        }

        [Test]
        public void Log_writes_csv()
        {
            var log = new DisturbanceLog();
            log.Add(new DisturbanceEvent { Time = 1.5, EpisodeTime = 0.5f, Type = "push", Detail = "dv=(0.4, \"0\")", OutsideTraining = true, BasePosition = new Vector3(1, 2, 3) });
            var lines = log.ToCsv().Trim().Split('\n');
            Assert.AreEqual(2, lines.Length);
            StringAssert.StartsWith("time_s,episode_time_s,type,outside_training", lines[0]);
            StringAssert.StartsWith("1.500,0.500,push,1,1.000,2.000,3.000", lines[1]);
            StringAssert.Contains("'0'", lines[1], "quotes inside the detail are escaped");
        }

        [Test]
        public void Upright_robot_is_not_fallen_and_base_on_the_ground_is()
        {
            using var world = HeadlessWorld.Create();
            var c = world.AddRobot("v2", Vector3.zero, footFriction: 1f);
            var manager = world.Add(new GameObject("Game")).AddComponent<DisturbanceManager>();
            manager.Configure(c, DisturbanceLimits.CreateDefault());

            var terrain = LayerMask.GetMask(SimSettings.TerrainLayer);
            Assert.IsFalse(c.Body.BaseTouches(world.Physics, terrain), "standing at spawn");
            world.Run(1f);
            Assert.AreEqual(0, c.Falls, "standing still for 1 s");

            // Lay the robot on its back (legs up): the next policy steps must report Isaac's base_contact.
            // (TeleportRoot moves the physics body; transforms follow after the next physics step.)
            c.Body.Teleport(new Vector3(0f, 0.07f, 0f), Quaternion.AngleAxis(180f, Vector3.forward), world.Interface.DefaultJointPos);
            for (var i = 0; i < 2 * world.Interface.Timing.Decimation; i++)
                world.Step();
            Assert.IsTrue(c.Body.BaseTouches(world.Physics, terrain), "lying on its back");
            Assert.GreaterOrEqual(c.Falls, 1);
            Assert.AreEqual("base contact", c.LastFallReason);
            var fall = manager.Log.Events.First(e => e.Type == "fall");
            Assert.AreEqual("base contact", fall.Detail);
            Assert.Greater(fall.Time, 1.0, "stamped with the total simulated time");
        }
    }
}
