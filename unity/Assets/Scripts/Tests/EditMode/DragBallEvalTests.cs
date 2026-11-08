using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>Step 5 of the disturbance system: drag, ball, and the Isaac evaluation in Unity physics.</summary>
    public class DragBallEvalTests
    {
        static (HeadlessWorld world, Go2Controller robot, DisturbanceManager m) Setup(bool hard = false)
        {
            var world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero, footFriction: 1f);
            var m = world.AddDisturbances(robot, hard);
            world.Run(0.5f);  // settle on the ground
            return (world, robot, m);
        }

        static float Horizontal(Vector3 v) => new Vector2(v.x, v.z).magnitude;

        // --- drag ------------------------------------------------------------------------------------------------

        [Test]
        public void Drag_force_is_clamped_to_the_active_limit()
        {
            var (world, robot, m) = Setup();
            using (world)
            {
                var root = robot.Body.Root;
                var f = m.ApplyDragForce(root, root.worldCenterOfMass, new Vector3(1000f, 0f, 0f), out var outside);
                Assert.AreEqual(20f, f.magnitude, 1e-3f, "training range: 20 N");
                Assert.IsFalse(outside);
                Assert.That(Vector3.Angle(f, Vector3.right), Is.LessThan(1e-3f), "direction kept");
                m.HardMode = true;
                f = m.ApplyDragForce(root, root.worldCenterOfMass, new Vector3(0f, 0f, -1000f), out outside);
                Assert.AreEqual(300f, f.magnitude, 1e-2f, "hard mode: 300 N");
                Assert.IsTrue(outside);
                f = m.ApplyDragForce(root, root.worldCenterOfMass, new Vector3(0f, 0f, 10f), out outside);
                Assert.AreEqual(10f, f.magnitude, 1e-4f);
                Assert.IsFalse(outside);
            }
        }

        [Test]
        public void Drag_tool_pulls_the_base_toward_the_target_and_logs_one_event()
        {
            var (world, robot, m) = Setup(hard: true);
            using (world)
            {
                var tool = new DragTool { MaxForce = 250f, Stiffness = 400f };
                var root = robot.Body.Root;
                var start = root.transform.position;
                tool.Grab(m, root, root.worldCenterOfMass);
                tool.Target = root.worldCenterOfMass + new Vector3(1f, 0.1f, 0f);
                var dt = world.Interface.Timing.PhysicsDt;
                var peak = 0f;
                world.Run(1f, () => peak = Mathf.Max(peak, tool.Pull(m, dt).magnitude));
                var moved = root.transform.position - start;
                Assert.Greater(moved.x, 0.3f, $"pulled sideways {moved}");
                Assert.LessOrEqual(peak, 250f + 1e-3f, "capped by the tool's max force");
                Assert.Greater(peak, 100f);

                var before = m.Log.Events.Count;
                tool.Release(m);
                Assert.IsFalse(tool.Holding);
                Assert.AreEqual(before + 1, m.Log.Events.Count);
                var e = m.Log.Events.Last();
                Assert.AreEqual("drag", e.Type);
                Assert.IsTrue(e.OutsideTraining, "250 N is beyond the 20 N training mark");
                StringAssert.Contains("base", e.Detail);
            }
        }

        // --- ball ------------------------------------------------------------------------------------------------

        [Test]
        public void Ball_momentum_is_capped_by_lowering_the_speed()
        {
            var (world, robot, m) = Setup();
            using (world)
            {
                var b = m.ThrowBall(new Vector3(0f, 2f, -5f), new Vector3(0f, 0f, 50f), 1f);
                Assert.AreEqual(7.5f, b.linearVelocity.magnitude, 1e-3f, "training: p ≤ 7.5 N·s");
                Assert.AreEqual(1f, b.mass, 1e-6f);
                Assert.AreEqual(b.gameObject.scene, robot.gameObject.scene, "ball lives in the robot's physics scene");
                Assert.AreEqual("ball", m.Log.Events.Last().Type);
                m.HardMode = true;
                b = m.ThrowBall(new Vector3(0f, 2f, -5f), new Vector3(0f, 0f, 50f), 1f);
                Assert.AreEqual(40f, b.linearVelocity.magnitude, 1e-3f, "hard: p ≤ 40 N·s");
                Assert.IsTrue(m.Log.Events.Last().OutsideTraining);
                b = m.ThrowBall(new Vector3(0f, 2f, -5f), new Vector3(0f, 0f, 3f), 0.5f);
                Assert.AreEqual(3f, b.linearVelocity.magnitude, 1e-4f, "below the cap: unchanged");
                Assert.AreEqual(3, m.Balls.Count);
                m.ClearBalls();
                Assert.AreEqual(0, m.Balls.Count);
            }
        }

        [Test]
        public void Aim_velocity_lands_on_the_target()
        {
            var g = -Physics.gravity.y;
            foreach (var (origin, target, speed) in new[]
                     {
                         (new Vector3(0f, 5f, -10f), new Vector3(2f, 0f, 3f), 20f),
                         (new Vector3(1f, 1.5f, 0f), new Vector3(-4f, 0.3f, 6f), 12f),
                     })
            {
                var v = DisturbanceManager.AimVelocity(origin, target, speed);
                Assert.AreEqual(speed, v.magnitude, 1e-3f);
                var d = target - origin;
                var t = Horizontal(d) / Horizontal(v);
                var y = v.y * t - 0.5f * g * t * t;
                Assert.AreEqual(d.y, y, 0.01f, "lands at the target's height");
                Assert.That(Vector2.Angle(new Vector2(v.x, v.z), new Vector2(d.x, d.z)), Is.LessThan(0.01f));
            }
        }

        [Test]
        public void A_heavy_ball_knocks_the_robot()
        {
            var (world, robot, m) = Setup(hard: true);
            using (world)
            {
                var root = robot.Body.Root;
                var origin = root.worldCenterOfMass + new Vector3(-2.5f, 0.3f, 0f);
                var ball = m.ThrowBall(origin, DisturbanceManager.AimVelocity(origin, root.worldCenterOfMass, 20f), 2f);
                var peak = 0f;
                world.Run(0.6f, () => peak = Mathf.Max(peak, root.linearVelocity.x));
                Assert.Greater(peak, 0.3f, "2 kg at 20 m/s (40 N·s) shoves the 15 kg robot along +x");
                Debug.Log($"[BMB] ball test: base peak vx {peak:F2} m/s, hit callback {ball.GetComponent<BallProjectile>().HitRobot}");
            }
        }

        // --- evaluation building blocks ----------------------------------------------------------------------------

        [Test]
        public void Command_sampler_follows_isaacs_heading_control()
        {
            var spec = RobotInterface.Load().Commands;
            var s = new CommandSampler(spec, new System.Random(3)) { StandingProbability = 0f };
            var first = s.Next(0.02f, 0f);
            Assert.That(first.x, Is.InRange(spec.LinVelX[0], spec.LinVelX[1]));
            Assert.That(first.y, Is.InRange(spec.LinVelY[0], spec.LinVelY[1]));
            var target = s.HeadingTarget;
            Assert.AreEqual(0f, s.Next(0.02f, target).z, 1e-5f, "on heading: no turn");
            Assert.AreEqual(spec.HeadingControlStiffness * 0.4f, s.Next(0.02f, target - 0.4f).z, 1e-5f);
            Assert.AreEqual(spec.AngVelZ[1], s.Next(0.02f, target - 3f).z, 1e-5f, "clipped to the wz range");
            Assert.AreEqual(spec.HeadingControlStiffness * -0.2f,
                            s.Next(0.02f, target + 0.2f - 2f * Mathf.PI).z, 1e-4f, "heading error wrapped to [−π, π]");

            var changes = 0;
            var last = s.Current;
            for (var i = 0; i < 1000; i++)  // 20 s more
            {
                var c = s.Next(0.02f, s.HeadingTarget);
                if (c.x != last.x)
                    changes++;
                last = c;
            }
            Assert.AreEqual(2, changes, "resampled every 10 s");

            s.Cap(0.3f, 0.5f);
            s.Reset();
            for (var i = 0; i < 20; i++)
            {
                s.Reset();
                s.Next(0.02f, 0f);  // resample
                var c = s.Next(0.02f, s.HeadingTarget - 3f);
                Assert.That(Mathf.Abs(c.x), Is.LessThanOrEqualTo(0.3f));
                Assert.That(Mathf.Abs(c.y), Is.LessThanOrEqualTo(0.3f));
                Assert.AreEqual(0.5f, c.z, 1e-6f);
            }

            Assert.AreEqual(0f, CommandSampler.Heading(Quaternion.identity), 1e-6f, "Unity +z = Isaac +x");
            Assert.AreEqual(-Mathf.PI / 2f, CommandSampler.Heading(Quaternion.Euler(0f, 90f, 0f)), 1e-5f, "Unity +x = Isaac −y");
        }

        [Test]
        public void Actuator_gain_scales_scale_the_pd_torque()
        {
            var a = RobotInterface.Load().Actuator;
            Assert.AreEqual(DCMotor.Torque(0.1f, 0f, 0f, a), DCMotor.Torque(0.1f, 0f, 0f, a, 1f, 1f), "scale 1 is exact");
            Assert.AreEqual(a.Stiffness * 1.1f * 0.1f, DCMotor.Torque(0.1f, 0f, 0f, a, 1.1f, 1f), 1e-4f);
            Assert.AreEqual(-a.Damping * 0.9f * 2f, DCMotor.Torque(0f, 0f, 2f, a, 1f, 0.9f), 1e-4f);
        }

        [Test]
        public void Foot_friction_can_drop_mid_run()
        {
            var (world, robot, m) = Setup();
            using (world)
            {
                Assert.AreEqual(1f, robot.Body.FootStaticFriction(0), 1e-6f);
                robot.FootFriction.SetFootMu(0.1f);
                for (var i = 0; i < 4; i++)
                    Assert.AreEqual(0.1f, robot.Body.FootStaticFriction(i), 1e-6f);
                Assert.AreEqual(0.1f, robot.FootFriction.EffectiveStatic(2), 1e-6f);
                // Standing still, v2 keeps one foot lifted (~14 cm); the contact test must agree with the foot heights
                // (foot sphere radius 0.022 m, flat ground at y = 0).
                var mask = LayerMask.GetMask(SimSettings.TerrainLayer);
                var touching = 0;
                for (var i = 0; i < 4; i++)
                {
                    var down = robot.Body.FootPosition(i).y < 0.022f + 0.002f;
                    Assert.AreEqual(down, robot.Body.FootTouches(i, world.Physics, mask), $"foot {i} at {robot.Body.FootPosition(i).y:F4} m");
                    touching += down ? 1 : 0;
                }
                Assert.GreaterOrEqual(touching, 3);
            }
        }

        [Test]
        public void Eval_smoke_run_writes_one_row_per_policy()
        {
            var o = new EvalOptions { EnvsPerPolicy = 2, EpisodeSeconds = 2f, TilesPerSide = 1, Border = 6f };
            var rows = UnityEval.RunSync(new[] { EvalScenario.Find("mu_drop"), EvalScenario.Find("pits_deceived") }, o);
            Assert.AreEqual(4, rows.Count);
            foreach (var r in rows)
            {
                Assert.AreEqual(2, r.Envs);
                Assert.AreEqual(100, r.Steps);
                Assert.That(r.SuccessRate, Is.InRange(0f, 1f));
                Assert.That(r.TrackingError, Is.InRange(0f, 3f));
                Assert.That(r.SlipSpeed, Is.InRange(0f, 3f));
                StringAssert.StartsWith("{\"scenario\": ", r.ToJson());
            }
            CollectionAssert.AreEqual(new[] { "v1", "v2", "v1", "v2" }, rows.Select(r => r.Policy).ToArray());

            var path = System.IO.Path.Combine(Application.temporaryCachePath, "bmb_eval_test.jsonl");
            System.IO.File.WriteAllText(path, "");
            UnityEval.AppendResults(path, rows);
            var md = UnityEval.Summary(path, UnityEval.IsaacResults, new[] { EvalPolicy.V1, EvalPolicy.V2 });
            StringAssert.Contains("| mu_drop |", md);
            StringAssert.Contains("| pits_deceived |", md);
        }
    }
}
