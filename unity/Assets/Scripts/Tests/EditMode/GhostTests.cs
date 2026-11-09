using System.Collections.Generic;
using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>
    /// The world model in closed loop and its ghost: the ghost's forward kinematics against Isaac's link poses, an
    /// imagined 0.5 s rollout against what the robot then really does in Unity, and the GhostView component.
    /// </summary>
    public class GhostTests
    {
        HeadlessWorld world;

        [TearDown]
        public void TearDown() => world?.Dispose();

        [Test]
        public void Ghost_kinematics_match_isaac_link_poses()
        {
            world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero);
            var ghost = GhostRobot.Create(robot.Body, new Material(Shader.Find("Sprites/Default")));
            var r = ReferenceRollout.Load("v2");
            float maxPos = 0f, maxAngle = 0f;
            string worst = "";
            for (var t = 0; t < r.Steps; t += 25)
            {
                var (pos, angle, link) = LinkErrors(ghost, r, t, null);
                if (pos > maxPos) (maxPos, worst) = (pos, $"step {t}, {link}");
                maxAngle = Mathf.Max(maxAngle, angle);
            }
            Debug.Log($"[BMB] Ghost kinematics: max link position error {maxPos * 1000:F4} mm ({worst}), max orientation error {maxAngle:F4} deg");
            Assert.That(maxPos, Is.LessThan(1e-4f));
            Assert.That(maxAngle, Is.LessThan(0.05f));
            // negative control: a flipped thigh joint must show
            var (flipped, _, _) = LinkErrors(ghost, r, 250, RobotInterface.Load().JointNames.IndexOf("FL_thigh_joint"));
            Assert.That(flipped, Is.GreaterThan(0.01f));
            Assert.AreEqual(0, ghost.GetComponentsInChildren<Collider>(true).Length, "ghost has no colliders");
            Assert.AreEqual(0, ghost.GetComponentsInChildren<ArticulationBody>(true).Length, "ghost has no articulation");
            Assert.That(ghost.Renderers.Count, Is.GreaterThan(10));
        }

        [Test]
        public void Dream_predicts_where_the_robot_walks()
        {
            // v2 at 0.5 m/s on flat ground. Every 50 policy steps (after 3 s), imagine 25 steps (0.5 s) ahead and
            // compare with where the robot really is 25 steps later in Unity physics.
            world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero, command: new Vector3(0.5f, 0f, 0f), worldModel: true,
                                       surprisal: SurprisalParams.Isaac(RobotInterface.Load(), "v2"));
            var dreamer = new WorldModelDreamer(robot) { Horizon = 25 };
            var pending = new List<(int at, Vector3 start, Vector3 ghost, Quaternion rot, float[] q)>();
            var errors = new List<(float xy, float dz, float yaw, float moved, float ahead, float qRms)>();
            var step = 0;
            var q = new float[12];
            robot.PolicyStepped += c =>
            {
                for (var i = pending.Count - 1; i >= 0; i--)
                {
                    var p = pending[i];
                    if (p.at != step)
                        continue;
                    var real = c.Body.BasePosition;
                    var d = real - p.ghost;
                    for (var j = 0; j < q.Length; j++)
                        q[j] = c.JointPos[j] - p.q[j];
                    var fwd = c.Body.BaseRotation * Vector3.forward;
                    errors.Add((new Vector2(d.x, d.z).magnitude, Mathf.Abs(d.y),
                                Quaternion.Angle(Quaternion.Euler(0f, p.rot.eulerAngles.y, 0f), Quaternion.Euler(0f, c.Body.BaseRotation.eulerAngles.y, 0f)),
                                new Vector2(real.x - p.start.x, real.z - p.start.z).magnitude,
                                Vector3.Dot(p.ghost - p.start, new Vector3(fwd.x, 0f, fwd.z).normalized),
                                Mathf.Sqrt(q.Select(v => v * v).Average())));
                    pending.RemoveAt(i);
                }
                if (step >= 150 && step % 50 == 0 && step <= 400)
                {
                    Assert.IsTrue(dreamer.Begin(), "dream starts");
                    Assert.IsTrue(dreamer.Advance(), "dream finishes");
                    pending.Add((step + 25, c.Body.BasePosition, dreamer.GhostPosition, dreamer.GhostRotation, dreamer.GhostJointPos.ToArray()));
                    Assert.AreEqual(26, dreamer.Path.Length);
                }
                step++;
            };
            world.Run(9f);

            Assert.AreEqual(6, errors.Count, "dreams compared");
            var e = (xy: errors.Average(x => x.xy), dz: errors.Average(x => x.dz), yaw: errors.Average(x => x.yaw),
                     moved: errors.Average(x => x.moved), ahead: errors.Average(x => x.ahead), q: errors.Average(x => x.qRms));
            Debug.Log($"[BMB] Dream vs Unity after 0.5 s (v2, 0.5 m/s, flat; {errors.Count} dreams): base xy error {e.xy * 100:F1} cm " +
                      $"(robot moved {e.moved * 100:F1} cm, ghost {e.ahead * 100:F1} cm ahead), height {e.dz * 100:F1} cm, yaw {e.yaw:F1} deg, " +
                      $"joint RMS {e.q:F3} rad; dream cost {dreamer.Completed} dreams");
            Assert.That(e.xy, Is.LessThan(0.15f), "xy error");
            Assert.That(e.xy, Is.LessThan(0.6f * e.moved), "better than standing still");
            Assert.That(e.ahead, Is.GreaterThan(0.1f), "the ghost is ahead of the robot");
            Assert.That(e.dz, Is.LessThan(0.05f), "base height");
            Assert.That(e.yaw, Is.LessThan(10f), "yaw");
            Assert.That(robot.Falls, Is.EqualTo(0));
        }

        [Test]
        public void Ghost_view_follows_the_robot()
        {
            world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero, command: new Vector3(0.6f, 0f, 0f), worldModel: true);
            var view = world.Add(new GameObject("WorldModel")).AddComponent<GhostView>();
            view.Configure(robot);
            Assert.IsTrue(view.EnsureCreated());
            var dt = robot.Interface.Timing.PolicyDt;
            world.Run(4f, () =>
            {
                if (robot.PhysicsStep % robot.Interface.Timing.Decimation == 0)
                    view.Tick(dt);
            });
            Assert.IsNotNull(view.Ghost);
            Assert.IsTrue(view.Ghost.gameObject.activeSelf, "ghost visible");
            Assert.That(view.Dreamer.Completed, Is.GreaterThan(20));
            var ahead = view.Ghost.Link("base").position - robot.Body.BasePosition;
            var fwd = robot.Body.BaseRotation * Vector3.forward;
            Debug.Log($"[BMB] Ghost view: {view.Dreamer.Completed} dreams in 4 s, ghost {Vector3.Dot(ahead, fwd) * 100:F1} cm ahead, tint {view.Tint}");
            Assert.That(Vector3.Dot(ahead, fwd), Is.InRange(0.05f, 0.6f), "a little ahead");
            Assert.That(view.Tint.b, Is.GreaterThan(view.Tint.r), "blue while walking normally");

            view.Show = false;
            view.Tick(dt);
            Assert.IsFalse(view.Ghost.gameObject.activeSelf, "hidden");
            view.Teardown();
            Assert.IsNull(view.Ghost);
        }

        [Test]
        public void Monitor_runs_in_closed_loop()
        {
            // v2 walking normally in Unity: predictions run every policy step, fall probability stays low. The
            // number of alarms with the Isaac thresholds is only logged; Unity's own calibration is step 4.
            world = HeadlessWorld.Create();
            var robot = world.AddRobot("v2", Vector3.zero, command: new Vector3(0.5f, 0f, 0.2f), worldModel: true,
                                       surprisal: SurprisalParams.Isaac(RobotInterface.Load(), "v2"));
            var m = robot.WorldModel;
            Assert.IsNotNull(m);
            float maxFall = 0f;
            robot.PolicyStepped += c => maxFall = Mathf.Max(maxFall, c.WorldModel.Prediction.FallProb[1]);
            world.Run(10f);
            Debug.Log($"[BMB] World model closed loop (v2, 10 s, flat, Isaac thresholds): {m.Detector.ScoredSteps} scored steps, " +
                      $"{m.Detector.Alarms} alarms, max P(fall 0.5 s) {maxFall:F3}");
            Assert.That(m.Detector.ScoredSteps, Is.InRange(474, 476), "500 policy steps minus the 25-step warm-up");
            Assert.That(maxFall, Is.LessThan(0.2f));
        }

        static (float pos, float angle, string link) LinkErrors(GhostRobot ghost, ReferenceRollout r, int t, int? flipJoint)
        {
            var basePos = IsaacFrame.PolarFromIsaac(new Vector3(r.root_link_pos_w[t, 0], r.root_link_pos_w[t, 1], r.root_link_pos_w[t, 2]));
            var baseRot = IsaacFrame.QuatFromIsaac(r.root_link_quat_w[t, 0], r.root_link_quat_w[t, 1], r.root_link_quat_w[t, 2], r.root_link_quat_w[t, 3]);
            var q = r.joint_pos.Row(t);
            if (flipJoint is int f)
                q[f] = -q[f];
            ghost.SetPose(basePos, baseRot, q);
            float maxPos = 0f, maxAngle = 0f;
            var worst = "";
            var n = r.body_names.Length;
            for (var k = 0; k < n; k++)
            {
                var o = t * n + k;
                var p = r.body_link_pos_w.data;
                var expectedPos = IsaacFrame.PolarFromIsaac(new Vector3(p[3 * o], p[3 * o + 1], p[3 * o + 2]));
                var qd = r.body_link_quat_w.data;
                var expectedRot = IsaacFrame.QuatFromIsaac(qd[4 * o], qd[4 * o + 1], qd[4 * o + 2], qd[4 * o + 3]);
                var link = ghost.Link(r.body_names[k]);
                Assert.IsNotNull(link, $"ghost link {r.body_names[k]}");
                var err = (link.position - expectedPos).magnitude;
                if (err > maxPos) (maxPos, worst) = (err, r.body_names[k]);
                maxAngle = Mathf.Max(maxAngle, Quaternion.Angle(link.rotation, expectedRot));
            }
            return (maxPos, maxAngle, worst);
        }
    }
}
