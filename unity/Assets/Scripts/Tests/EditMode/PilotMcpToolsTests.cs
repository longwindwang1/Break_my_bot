using System.Linq;
using System.Reflection;
using BMB.Editor;
using BMB.Editor.Mcp;
using MCPForUnity.Editor.Helpers;
using MCPForUnity.Editor.Tools;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>The MCP tools Claude Code drives the pilot with: registration, errors, parameter mapping.</summary>
    public class PilotMcpToolsTests
    {
        HeadlessWorld world;

        [TearDown]
        public void TearDown()
        {
            PilotMcp.Override = null;
            world?.Dispose();
        }

        static readonly string[] ToolNames =
            { "robot_observe", "robot_walk_to", "robot_move", "robot_stop", "robot_wait", "robot_imagine", "challenge_start", "challenge_status", "pilot_config" };

        [Test]
        public void Tools_are_registered_with_explicit_names_and_handlers()
        {
            var types = typeof(PilotMcp).Assembly.GetTypes()
                .Where(t => t.GetCustomAttribute<McpForUnityToolAttribute>() != null).ToArray();
            CollectionAssert.AreEquivalent(ToolNames, types.Select(t => t.GetCustomAttribute<McpForUnityToolAttribute>().Name));
            foreach (var t in types)
            {
                var attr = t.GetCustomAttribute<McpForUnityToolAttribute>();
                Assert.IsNotNull(t.GetMethod("HandleCommand", BindingFlags.Public | BindingFlags.Static, null, new[] { typeof(JObject) }, null), attr.Name);
                Assert.Greater(attr.Description.Length, 60, $"{attr.Name}: the description is the model's only manual");
                var parameters = t.GetNestedType("Parameters");
                Assert.IsNotNull(parameters, attr.Name);
                foreach (var p in parameters.GetProperties())
                {
                    Assert.IsNotNull(p.GetCustomAttribute<ToolParameterAttribute>(), $"{attr.Name}.{p.Name}");
                    Assert.AreEqual(p.Name.ToLowerInvariant(), p.Name, "snake_case parameter names");
                }
            }
        }

        [Test]
        public void Tools_need_play_mode()
        {
            var r = RobotObserve.HandleCommand(new JObject());
            Assert.IsInstanceOf<ErrorResponse>(r);
            Assert.AreEqual("not_playing", ((ErrorResponse)r).Code);
            var w = RobotWalkTo.HandleCommand(new JObject { ["x"] = 1, ["y"] = 2 }).Result;
            Assert.IsInstanceOf<ErrorResponse>(w);
        }

        [Test]
        public void Tools_map_their_parameters_onto_the_pilot()
        {
            world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero, worldModel: true);
            var manager = world.AddDisturbances(robot);
            var go = world.Add(new GameObject("Pilot"));
            var pilot = go.AddComponent<ClaudePilot>();
            pilot.Configure(robot, manager);
            var challenge = go.AddComponent<PilotChallenge>();
            challenge.Configure(pilot);
            challenge.WriteResults = false;
            world.Run(1f);
            PilotMcp.Override = pilot;

            var obs = (SuccessResponse)RobotObserve.HandleCommand(new JObject());
            var json = JsonConvert.SerializeObject(obs.Data);
            StringAssert.Contains("\"fall_prob\":[", json);
            StringAssert.Contains("\"yaw_deg\":", json);

            var start = (SuccessResponse)ChallengeStart.HandleCommand(new JObject { ["goal_x"] = 4, ["goal_y"] = "1.5", ["time_limit_s"] = 45 });
            StringAssert.Contains("round 1", start.Message);
            Assert.IsTrue(challenge.Running);
            Assert.AreEqual(new Vector2(4f, 1.5f), challenge.Goal);
            Assert.AreEqual(45f, challenge.TimeLimit);
            Assert.IsInstanceOf<ErrorResponse>(ChallengeStart.HandleCommand(new JObject { ["goal_x"] = 4 }));

            var imagine = (SuccessResponse)RobotImagine.HandleCommand(new JObject { ["x"] = 4, ["y"] = 1.5, ["seconds"] = 1 });
            var imagined = ((PilotActionResult)imagine.Data).imagined;
            Assert.AreEqual(1f, imagined.seconds, 1e-4f);
            Assert.AreEqual(4, imagined.path.Length, "a point every 0.25 s");
            Assert.IsInstanceOf<ErrorResponse>(RobotImagine.HandleCommand(new JObject()));

            var walk = RobotWalkTo.HandleCommand(new JObject { ["x"] = 4, ["y"] = 1.5, ["speed"] = 0.5, ["note"] = "towards the goal", ["max_wait_s"] = 0 }).Result;
            var issued = (PilotActionResult)((SuccessResponse)walk).Data;
            Assert.AreEqual("issued", issued.outcome);
            Assert.IsNotNull(issued.observation, "returning at once still reports the state");
            Assert.AreEqual(PilotOrderKind.WalkTo, pilot.Order.Kind);
            Assert.AreEqual(0.5f, pilot.Order.Speed);
            Assert.AreEqual("towards the goal", pilot.Order.Note);
            Assert.IsInstanceOf<ErrorResponse>(RobotWalkTo.HandleCommand(new JObject { ["x"] = 4 }).Result);
            var bad = RobotWalkTo.HandleCommand(new JObject { ["x"] = 4, ["y"] = 1, ["speed"] = -1 }).Result;
            Assert.AreEqual("rejected", ((ErrorResponse)bad).Code);

            RobotMove.HandleCommand(new JObject { ["vx"] = 0.3, ["vy"] = 0, ["wz"] = 0.2, ["seconds"] = 2, ["max_wait_s"] = 0 }).Wait();
            Assert.AreEqual(PilotOrderKind.Move, pilot.Order.Kind);
            Assert.AreEqual(new Vector3(0.3f, 0f, 0.2f), pilot.Order.Velocity);

            Assert.IsInstanceOf<SuccessResponse>(RobotStop.HandleCommand(new JObject { ["note"] = "hold on" }));
            Assert.AreEqual(PilotOrderKind.Stand, pilot.Order.Kind);

            var status = (SuccessResponse)ChallengeStatus.HandleCommand(new JObject());
            Assert.IsTrue(((PilotChallengeStatus)status.Data).running);

            PilotConfig.HandleCommand(new JObject { ["veto"] = false, ["guard"] = "false", ["veto_fall_prob"] = 0.7 });
            Assert.IsFalse(pilot.Veto);
            Assert.IsFalse(pilot.Guard);
            Assert.AreEqual(0.7f, pilot.VetoFallProb, 1e-6f);
            PilotConfig.HandleCommand(new JObject { ["release"] = true });
            Assert.IsFalse(pilot.Engaged);
        }
    }
}
