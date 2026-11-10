using System.Collections.Generic;
using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>AI pilot + world model: imagined orders vs reality, the fall-risk veto, the surprise guard.</summary>
    public class PilotImaginationTests
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

        static float Flat(Vector3 a, Vector3 b) => new Vector2(a.x - b.x, a.z - b.z).magnitude;

        [Test]
        public void Imagined_orders_match_the_real_robot()
        {
            // Imagine a walk-to (or move) for 2 s at a policy step, then execute it from the same step and compare the
            // imagined base path with the real one at 0.5 / 1 / 2 s (25 / 50 / 100 policy steps).
            var cases = new (string name, System.Func<PilotToolbox, PilotActionResult> imagine, System.Func<PilotToolbox, PilotActionResult> run)[]
            {
                ("walk_to (3, 2) ahead-left", t => t.ImagineWalkTo(3f, 2f, 0.6f, 2f), t => t.WalkTo(3f, 2f, 0.6f)),
                ("walk_to (-2, 2) behind: turn first", t => t.ImagineWalkTo(-2f, 2f, 0.6f, 2f), t => t.WalkTo(-2f, 2f, 0.6f)),
                ("walk_to (6, 0) at 1 m/s", t => t.ImagineWalkTo(6f, 0f, 1f, 2f), t => t.WalkTo(6f, 0f, 1f)),
                ("move vy 0.4, wz 0.5", t => t.ImagineMove(0f, 0.4f, 0.5f, 2f), t => t.Move(0f, 0.4f, 0.5f, 2f)),
            };
            var checkpoints = new[] { 25, 50, 100 };
            var table = new List<string>();
            foreach (var (name, imagine, run) in cases)
            {
                world?.Dispose();
                world = HeadlessWorld.Create();
                var robot = world.AddRobot("v2", Vector3.zero, worldModel: true);
                var pilot = AddPilot(robot);
                pilot.Veto = false;
                Vector3[] imagined = null;
                var risk = 0f;
                var real = new List<Vector3>();
                robot.PolicyStepped += c =>
                {
                    if (imagined == null && c.PhysicsStep >= 50 * c.Interface.Timing.Decimation)
                    {
                        var r = imagine(pilot.Toolbox);
                        Assert.IsTrue(r.accepted, r.detail);
                        risk = ClaudePilot.Risk(r.imagined.max_fall_prob);
                        imagined = (Vector3[])pilot.Imagination.LastPath.Clone();
                        Assert.IsTrue(run(pilot.Toolbox).accepted);
                        real.Add(c.Body.BasePosition);
                    }
                    else if (imagined != null && real.Count <= 100)
                        real.Add(c.Body.BasePosition);
                };
                world.Run(3.2f);
                Assert.AreEqual(101, imagined.Length);
                Assert.AreEqual(101, real.Count);
                Assert.AreEqual(0, robot.Falls);
                var errors = checkpoints.Select(k => Flat(imagined[k], real[k])).ToArray();
                var moved = Flat(real[100], real[0]);
                table.Add($"| {name} | {errors[0] * 100:F1} | {errors[1] * 100:F1} | {errors[2] * 100:F1} | {moved * 100:F0} | {risk:F3} |");
                Assert.Less(errors[0], 0.1f, $"{name}: 0.5 s");
                Assert.Less(errors[1], 0.2f, $"{name}: 1 s");
                Assert.Less(errors[2], 0.5f, $"{name}: 2 s");
                Assert.Less(risk, 0.5f, $"{name}: open ground is not vetoed");
            }
            Debug.Log("[BMB] Imagined vs real base position (cm), v2 flat ground:\n| order | 0.5 s | 1 s | 2 s | moved in 2 s | max P(fall 1 s) |\n|---|---:|---:|---:|---:|---:|\n" + string.Join("\n", table));
        }

        /// <summary>A 40 cm deep pit (hard mode) across the robot's path, its near edge <paramref name="edge"/> m ahead (Isaac x).</summary>
        (Go2Controller robot, ClaudePilot pilot) PitAhead(float edge)
        {
            world = HeadlessWorld.Create(flatGround: false);
            var level = world.AddLevelGround(new Vector2(20f, 20f));
            var robot = world.AddRobot("v2", Vector3.zero, worldModel: true);
            var manager = world.AddDisturbances(robot, hardMode: true, level: level);
            var pilot = AddPilot(robot, manager);
            // Isaac x ahead = Unity +z; the pit is 2 m long, 3 m wide.
            manager.PlaceObstacle(new GroundFeature { type = GroundFeatureType.Pit, size = new Vector2(3f, 2f), height = 0.4f }, new Vector3(0f, 0f, edge + 1f));
            world.Run(1f);
            return (robot, pilot);
        }

        [Test]
        public void Veto_rejects_an_order_into_a_pit()
        {
            var (robot, pilot) = PitAhead(0.45f);
            var intoPit = pilot.Toolbox.ImagineWalkTo(4f, 0f, 0.8f, ClaudePilot.VetoSeconds).imagined;
            var away = pilot.Toolbox.ImagineWalkTo(-4f, 0f, 0.8f, ClaudePilot.VetoSeconds).imagined;
            Debug.Log($"[BMB] Imagined 1 s, max P(fall) [0.2, 0.5, 1 s]: into a 40 cm pit 0.45 m ahead [{string.Join(", ", intoPit.max_fall_prob)}], " +
                      $"away from it [{string.Join(", ", away.max_fall_prob)}]");
            Assert.Greater(ClaudePilot.Risk(intoPit.max_fall_prob), pilot.VetoFallProb, "the fall head sees the pit through the height scan");
            Assert.Less(ClaudePilot.Risk(away.max_fall_prob), pilot.VetoFallProb, "open ground is not vetoed");

            var vetoed = pilot.Toolbox.WalkTo(4f, 0f, 0.8f);
            Assert.IsFalse(vetoed.accepted);
            Assert.AreEqual("vetoed", vetoed.outcome);
            Assert.IsNotNull(vetoed.imagined);
            Assert.IsFalse(pilot.Engaged, "a vetoed order does not move the robot");
            var ok = pilot.Toolbox.WalkTo(-4f, 0f, 0.8f);
            Assert.IsTrue(ok.accepted, ok.detail);
            Assert.AreEqual("issued", ok.outcome);
        }

        [Test]
        public void Lookahead_halts_a_running_order_before_a_pit()
        {
            // The pit is beyond the first second when the order starts, so only the running lookahead can catch it.
            var (robot, pilot) = PitAhead(3f);
            var issued = pilot.Toolbox.WalkTo(6f, 0f, 0.8f);
            Assert.IsTrue(issued.accepted, issued.detail);
            var w = pilot.Toolbox.BeginWait(20f);
            var n = 0;
            var steps = 0;
            while (!pilot.Toolbox.Settled(w) && steps++ < 2400)
            {
                world.Step();
                if (++n % 3 == 0)
                    pilot.TickLookahead(4);  // ~3 physics steps per frame, a few ms of dreaming each
            }
            var result = pilot.Toolbox.Conclude(w, issued);
            var x = PilotPose.Of(robot).Position.x;
            world.Run(1f);  // let it come to rest
            var xRest = PilotPose.Of(robot).Position.x;
            Debug.Log($"[BMB] Lookahead: {result.outcome} at x {x:F2} m (pit edge 3.0 m), at rest x {xRest:F2}, " +
                      $"{pilot.LookaheadDreams} dreams, risk {pilot.LookaheadRisk:F2}, falls {robot.Falls}");
            Assert.AreEqual("halted", result.outcome, result.detail);
            Assert.AreEqual(PilotOrderState.Halted, pilot.Order.State);
            Assert.AreEqual(0, robot.Falls, "stopped before the pit");
            Assert.Less(xRest, 3f, "short of the edge");
            Assert.Greater(xRest, 1.5f, "but not much earlier");
        }

        [Test]
        public void Surprise_alarm_slows_the_pilot_and_ends_the_wait()
        {
            world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero, footFriction: 1f, worldModel: true);
            var pilot = AddPilot(robot);
            world.Run(1f);
            var issued = pilot.Toolbox.WalkTo(20f, 0f, 0.8f);
            Assert.IsTrue(issued.accepted, issued.detail);
            world.Run(3f);
            Assert.IsFalse(pilot.GuardActive, "no alarm on clean walking");
            Assert.AreEqual(0.8f, robot.CommandOverride.Value.x, 0.05f);

            var w = pilot.Toolbox.BeginWait(10f);
            robot.FootFriction.SetFootMu(0.05f);
            var steps = 0;
            while (!pilot.Toolbox.Settled(w) && steps++ < 2000)
                world.Step();
            var result = pilot.Toolbox.Conclude(w, issued);
            Debug.Log($"[BMB] Ice under the pilot: {result.outcome} after {steps * world.Interface.Timing.PhysicsDt:F2} s — {result.detail}");
            Assert.AreEqual("alarm", result.outcome, result.detail);
            Assert.Less(steps * world.Interface.Timing.PhysicsDt, 2f, "within 2 s");
            Assert.IsTrue(pilot.GuardActive);
            Assert.IsTrue(result.observation.guard_active);
            world.Run(0.05f);
            var cmd = robot.CommandOverride.Value;
            Assert.LessOrEqual(new Vector2(cmd.x, cmd.y).magnitude, 0.3f + 1e-5f, "cautious gait");
            Assert.AreEqual("running", pilot.Order.State.ToString().ToLowerInvariant(), "the order goes on, slower");

            // Off the ice the cap lifts once no alarm came for the hold time.
            robot.FootFriction.SetFootMu(1f);
            world.Run(ClaudePilot.GuardHoldSeconds + 3f);
            Assert.IsFalse(pilot.GuardActive);
            Assert.AreEqual(0.8f, robot.CommandOverride.Value.x, 0.05f);
        }
    }
}
