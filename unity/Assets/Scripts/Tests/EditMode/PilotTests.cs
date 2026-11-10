using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>AI pilot: orders driven through the locomotion policy, waiting semantics, structured observation.</summary>
    public class PilotTests
    {
        HeadlessWorld world;

        [TearDown]
        public void TearDown() => world?.Dispose();

        ClaudePilot AddPilot(Go2Controller robot, DisturbanceManager manager = null)
        {
            var pilot = world.Add(new GameObject("Pilot")).AddComponent<ClaudePilot>();
            pilot.Configure(robot, manager);
            return pilot;
        }

        /// <summary>Steps until the wait settles (or <paramref name="limit"/> seconds pass) and concludes it.</summary>
        PilotActionResult Wait(PilotToolbox tools, PilotActionResult issued, float maxWait, float limit = 60f)
        {
            var w = tools.BeginWait(maxWait);
            var steps = Mathf.RoundToInt(limit / world.Interface.Timing.PhysicsDt);
            for (var i = 0; i < steps && !tools.Settled(w); i++)
                world.Step();
            return tools.Conclude(w, issued);
        }

        [Test]
        public void Walk_to_reaches_a_target_through_the_policy()
        {
            world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero);
            var pilot = AddPilot(robot);
            world.Run(1f);

            var issued = pilot.Toolbox.WalkTo(4f, 3f, 0.6f, "test");
            Assert.IsTrue(issued.accepted, issued.detail);
            Assert.IsTrue(pilot.Engaged);
            var start = robot.TotalTime;
            var result = Wait(pilot.Toolbox, issued, 40f);

            var pose = PilotPose.Of(robot);
            var took = robot.TotalTime - start;
            Debug.Log($"[BMB] walk_to (4, 3): {result.outcome} after {took:F1} s at ({pose.Position.x:F2}, {pose.Position.y:F2}); {result.detail}");
            Assert.AreEqual("arrived", result.outcome, result.detail);
            Assert.Less(pose.DistanceTo(new Vector2(4f, 3f)), PilotDrive.ArriveRadius + 0.05f);
            Assert.Less(took, 5f / 0.6f * 2f, "about the straight-line time, with turning");
            Assert.AreEqual(0, robot.Falls);
            Assert.AreEqual(Vector3.zero, robot.CommandOverride, "stands once there");
            Assert.AreEqual("arrived", result.observation.order.state);
            Assert.AreEqual(4f, result.observation.x, PilotDrive.ArriveRadius + 0.05f);
            Assert.AreEqual(3f, result.observation.y, PilotDrive.ArriveRadius + 0.05f);
        }

        [Test]
        public void Walk_to_behind_the_robot_turns_first()
        {
            world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero);
            var pilot = AddPilot(robot);
            world.Run(0.5f);
            var issued = pilot.Toolbox.WalkTo(-3f, 0f);
            world.Step();
            var cmd = pilot.CurrentCommand();
            Assert.AreEqual(0f, cmd.x, 1e-6f, "target behind: turn on the spot");
            Assert.AreEqual(1f, Mathf.Abs(cmd.z), 1e-6f, "at the full turn rate");
            var result = Wait(pilot.Toolbox, issued, 40f);
            Assert.AreEqual("arrived", result.outcome, result.detail);
            Assert.AreEqual(0, robot.Falls);
        }

        [Test]
        public void Move_holds_its_command_for_its_duration()
        {
            world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero);
            var pilot = AddPilot(robot);
            world.Run(0.5f);
            var x0 = PilotPose.Of(robot).Position.x;
            var issued = pilot.Toolbox.Move(0.5f, 0f, 0f, 2f, "forward");
            Assert.IsTrue(issued.accepted);
            var result = Wait(pilot.Toolbox, issued, 10f);
            Assert.AreEqual("done", result.outcome, result.detail);
            Assert.AreEqual(2f, pilot.Order.Elapsed(robot.TotalTime), 0.03f, "one policy step of slack");
            Assert.AreEqual(Vector3.zero, robot.CommandOverride);
            var moved = PilotPose.Of(robot).Position.x - x0;
            Debug.Log($"[BMB] move 0.5 m/s for 2 s: {moved:F2} m");
            Assert.Greater(moved, 0.6f);

            var clamped = pilot.Toolbox.Move(3f, 0f, -2f, 1f);
            Assert.IsTrue(clamped.accepted);
            StringAssert.Contains("clamped", clamped.detail);
            Assert.AreEqual(new Vector3(1f, 0f, -1f), pilot.Order.Velocity, "the yaml's command ranges");
        }

        [Test]
        public void Bad_arguments_are_rejected_and_release_hands_control_back()
        {
            world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero);
            var pilot = AddPilot(robot);
            var tools = pilot.Toolbox;
            Assert.IsFalse(tools.WalkTo(1f, 1f, 0f).accepted);
            Assert.IsFalse(tools.WalkTo(float.NaN, 1f).accepted);
            Assert.IsFalse(tools.Move(0.5f, 0f, 0f, 20f).accepted);
            Assert.IsFalse(tools.Move(0.5f, 0f, 0f, 0f).accepted);
            Assert.IsFalse(pilot.Engaged, "rejected orders do not take control");

            tools.WalkTo(5f, 0f);
            world.Run(1f);
            Assert.Greater(robot.CommandOverride.Value.x, 0.3f);
            var stop = tools.Stop("enough");
            Assert.IsTrue(stop.accepted);
            Assert.AreEqual(PilotOrderState.Standing, pilot.Order.State);
            world.Run(0.1f);
            Assert.AreEqual(Vector3.zero, robot.CommandOverride);

            // A waiting call returns when the order it waited on is replaced (a stop issued from another call).
            tools.WalkTo(5f, 0f);
            var w = tools.BeginWait(10f);
            world.Run(0.2f);
            tools.Stop();
            Assert.IsTrue(tools.Settled(w));

            pilot.Release();
            Assert.IsFalse(pilot.Engaged);
            Assert.IsNull(robot.CommandOverride, "back to the player's command source");
        }

        [Test]
        public void Observation_reports_goal_obstacles_ice_and_events()
        {
            world = HeadlessWorld.Create(flatGround: false);
            var level = world.AddLevelGround(new Vector2(20f, 20f));
            var robot = world.AddRobot("v2", Vector3.zero, worldModel: true);
            var manager = world.AddDisturbances(robot, level: level);
            var pilot = AddPilot(robot, manager);
            world.Run(0.5f);

            // Isaac (x, y) = Unity (z, −x).
            manager.PlaceObstacle(new GroundFeature { type = GroundFeatureType.Box, size = new Vector2(1f, 1f), height = 0.1f }, new Vector3(-1f, 0f, 3f));
            manager.PlaceObstacle(new GroundFeature { type = GroundFeatureType.HiddenPit, size = new Vector2(1f, 1f), height = 0.1f }, new Vector3(2f, 0f, 2f));
            manager.PaintFriction(new Vector3(1f, 0f, -3f), 1f, 0.1f);
            pilot.Goal = new Vector2(6f, 0f);
            world.Run(0.2f);

            var obs = pilot.Toolbox.Observe();
            var pose = PilotPose.Of(robot);
            Assert.IsTrue(obs.has_goal);
            Assert.AreEqual(pose.DistanceTo(new Vector2(6f, 0f)), obs.goal_distance, 0.01f);
            Assert.AreEqual(pose.BearingTo(new Vector2(6f, 0f)) * Mathf.Rad2Deg, obs.goal_bearing_deg, 0.1f);
            Assert.AreEqual(3, obs.fall_prob.Length, "world model: 0.2, 0.5, 1 s");

            Assert.AreEqual(1, obs.obstacles.Length, "the hidden pit is not reported");
            var box = obs.obstacles[0];
            Assert.AreEqual("box", box.type);
            Assert.AreEqual(3f, box.x, 0.06f);
            Assert.AreEqual(1f, box.y, 0.06f);
            Assert.AreEqual(0.1f, box.height, 1e-3f);
            Assert.AreEqual(Mathf.Sqrt(0.5f * 0.5f + 2.5f * 2.5f), box.distance, 0.15f, "to the footprint, not the centre");

            Assert.AreEqual(1, obs.ice.Length);
            var ice = obs.ice[0];
            Assert.AreEqual(-3f, ice.x, 0.3f);
            Assert.AreEqual(-1f, ice.y, 0.3f);
            Assert.AreEqual(0.1f, ice.min_mu, 1e-3f);
            Assert.AreEqual(1f, ice.radius, 0.3f);
            Assert.AreEqual(1f, obs.ground_mu);

            var types = obs.events.Select(e => e.type).ToArray();
            CollectionAssert.Contains(types, "obstacle");
            CollectionAssert.Contains(types, "friction");
            var trap = obs.events.Single(e => e.type == "deceptive");
            Assert.AreEqual("the player set a hidden trap somewhere", trap.detail, "where the trap is stays secret");
            Assert.AreEqual(0, pilot.Toolbox.Observe().events.Length, "events are reported once");

            var json = JsonUtility.ToJson(obs);
            StringAssert.Contains("\"goal_bearing_deg\"", json);
        }

        [Test]
        public void A_player_disturbance_ends_the_wait()
        {
            world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero);
            var manager = world.AddDisturbances(robot);
            var pilot = AddPilot(robot, manager);
            world.Run(0.5f);
            var issued = pilot.Toolbox.WalkTo(8f, 0f);
            var w = pilot.Toolbox.BeginWait(20f);
            world.Run(1f);
            Assert.IsFalse(pilot.Toolbox.Settled(w));
            manager.Push(new Vector2(0f, 0.4f));
            Assert.IsTrue(pilot.Toolbox.Settled(w));
            var result = pilot.Toolbox.Conclude(w, issued);
            Assert.AreEqual("disturbed", result.outcome);
            StringAssert.StartsWith("push", result.detail);
            Assert.AreEqual("running", result.observation.order.state, "the order keeps going");
        }

        [Test]
        public void Drive_commands_stay_in_range()
        {
            var ranges = RobotInterface.Load().Commands;
            var pose = new PilotPose(Vector2.zero, 0f);
            var ahead = PilotDrive.WalkTo(pose, new Vector2(10f, 0f), 0.8f, ranges);
            Assert.AreEqual(new Vector3(0.8f, 0f, 0f), ahead);
            var near = PilotDrive.WalkTo(pose, new Vector2(0.5f, 0f), 0.8f, ranges);
            Assert.AreEqual(0.4f, near.x, 1e-5f, "slows down within 1 m");
            Assert.AreEqual(Vector3.zero, PilotDrive.WalkTo(pose, new Vector2(0.2f, 0f), 0.8f, ranges), "arrived");
            var left = PilotDrive.WalkTo(pose, new Vector2(0f, 5f), 0.8f, ranges);
            Assert.AreEqual(0f, left.x);
            Assert.AreEqual(1f, left.z, 1e-6f, "turns left at the range limit");
            var cautious = PilotDrive.Cautious(new Vector3(0.8f, 0.6f, -1f), 0.3f, 0.5f);
            Assert.AreEqual(0.3f, new Vector2(cautious.x, cautious.y).magnitude, 1e-5f);
            Assert.AreEqual(-0.5f, cautious.z);
        }
    }
}
