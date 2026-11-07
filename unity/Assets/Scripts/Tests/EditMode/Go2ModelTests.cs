using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>
    /// Assets/Robots/Go2.prefab (built by BMB/Setup Go2) is the model Isaac Lab simulated (docs/unity_handoff.md §7),
    /// and verification level 1 (§8): with the base pose and joint angles of a reference frame, every link lands
    /// where Isaac had it. Joint angle signs, axes and the Unity ↔ Isaac frame conversion are all exercised.
    /// </summary>
    public class Go2ModelTests
    {
        const float PositionTolerance = 1e-3f;  // m (§8: about 1 mm)
        const float AngleToleranceDeg = 0.1f;

        RobotInterface itf;
        ReferenceRollout reference;
        GameObject robot;
        Go2Body body;
        ArticulationBody[] bodies;

        [OneTimeSetUp]
        public void Load()
        {
            itf = RobotInterface.Load();
            reference = ReferenceRollout.Load("flat");
        }

        [SetUp]
        public void Spawn()
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(Go2Setup.PrefabPath);
            Assert.IsNotNull(prefab, $"{Go2Setup.PrefabPath} missing; run BMB/Setup Go2");
            robot = Object.Instantiate(prefab);
            body = robot.GetComponent<Go2Body>();
            body.ApplyRuntimeSettings();  // Awake does not run in edit mode
            bodies = robot.GetComponentsInChildren<ArticulationBody>();
        }

        [TearDown]
        public void Despawn() => Object.DestroyImmediate(robot);

        ArticulationBody Body(string bodyName) => bodies.Single(b => b.name == bodyName);

        [Test]
        public void Bodies_and_inertials_match_isaac()
        {
            CollectionAssert.AreEquivalent(reference.body_names, bodies.Select(b => b.name));
            Assert.AreEqual(reference.body_mass.data.Sum(), bodies.Sum(b => b.mass), 1e-4f, "total mass");
            for (var k = 0; k < reference.body_names.Length; k++)
            {
                var b = Body(reference.body_names[k]);
                Assert.AreEqual(reference.body_mass.data[k], b.mass, 1e-6f, $"{b.name} mass");
                var com = IsaacFrame.PolarFromIsaac(new Vector3(reference.body_com_pose[k, 0], reference.body_com_pose[k, 1], reference.body_com_pose[k, 2]));
                Assert.That((b.centerOfMass - com).magnitude, Is.LessThan(1e-6f), $"{b.name} COM");

                // Independent conversion of the Isaac tensor: Unity axes (-y, z, x) => I_u[r,c] = s_r s_c I[src_r, src_c].
                int[] src = { 1, 2, 0 };
                float[] sign = { -1, 1, 1 };
                var isaac = reference.body_inertia.Row(k);
                var actual = Inertia.Tensor(b.inertiaTensor, b.inertiaTensorRotation);
                for (var r = 0; r < 3; r++)
                    for (var c = 0; c < 3; c++)
                        Assert.AreEqual(sign[r] * sign[c] * isaac[3 * src[r] + src[c]], actual[3 * r + c], 1e-8f, $"{b.name} inertia[{r},{c}]");
                Assert.IsFalse(b.automaticCenterOfMass || b.automaticInertiaTensor, $"{b.name} must keep the reference inertials");
            }
        }

        [Test]
        public void Joints_and_bodies_add_nothing_of_their_own()
        {
            body.Validate(itf);
            var robotLayer = LayerMask.NameToLayer(SimSettings.RobotLayer);
            foreach (var b in bodies)
            {
                Assert.AreEqual(0f, b.linearDamping, $"{b.name} linearDamping");
                Assert.AreEqual(0f, b.angularDamping, $"{b.name} angularDamping");
                Assert.AreEqual(itf.Actuator.Friction, b.jointFriction, $"{b.name} jointFriction");
                Assert.AreEqual(robotLayer, b.gameObject.layer, $"{b.name} layer");
                Assert.AreEqual(SimSettings.MaxDepenetrationVelocity, b.maxDepenetrationVelocity, $"{b.name} maxDepenetrationVelocity");
            }
            Assert.AreEqual(SimSettings.SolverIterations, body.Root.solverIterations, "solver iterations");
            Assert.AreEqual(SimSettings.SolverVelocityIterations, body.Root.solverVelocityIterations, "solver velocity iterations");
            for (var i = 0; i < body.NumJoints; i++)
            {
                var j = Body(itf.JointNames[i].Replace("_joint", ""));
                Assert.AreEqual(ArticulationJointType.RevoluteJoint, j.jointType);
                Assert.AreEqual(0f, j.xDrive.stiffness, $"{j.name} drive stiffness (torques come from DCMotor)");
                Assert.AreEqual(0f, j.xDrive.damping, $"{j.name} drive damping");
                Assert.AreEqual(reference.joint_pos_limits[i, 0], j.xDrive.lowerLimit * Mathf.Deg2Rad, 1e-5f, $"{j.name} lower limit");
                Assert.AreEqual(reference.joint_pos_limits[i, 1], j.xDrive.upperLimit * Mathf.Deg2Rad, 1e-5f, $"{j.name} upper limit");
                Assert.AreEqual(reference.joint_vel_limits.data[i], j.maxJointVelocity, 1e-5f, $"{j.name} velocity limit");
            }
            foreach (var c in robot.GetComponentsInChildren<Collider>())
            {
                Assert.AreEqual(robotLayer, c.gameObject.layer, $"{c.name} layer");
                Assert.AreEqual(itf.Friction.V1Foot[0], c.sharedMaterial.staticFriction, $"{c.name} static friction");
                Assert.AreEqual(itf.Friction.V1Foot[1], c.sharedMaterial.dynamicFriction, $"{c.name} dynamic friction");
                Assert.AreEqual(itf.Friction.UnityCombineMode(), c.sharedMaterial.frictionCombine, $"{c.name} combine");
            }
        }

        [TestCase("flat")]
        [TestCase("rough")]
        public void Link_poses_match_reference(string variant)
        {
            var r = ReferenceRollout.Load(variant);
            float maxPos = 0f, maxAngle = 0f;
            string worst = "";
            for (var t = 0; t < r.Steps; t += 25)
            {
                PoseFromReference(r, t, null);
                var (pos, angle, link) = LinkErrors(r, t);
                if (pos > maxPos) (maxPos, worst) = (pos, $"step {t}, {link}");
                maxAngle = Mathf.Max(maxAngle, angle);
            }
            Debug.Log($"[BMB] Kinematics {variant}: max link position error {maxPos * 1000:F4} mm ({worst}), max orientation error {maxAngle:F4} deg");
            Assert.That(maxPos, Is.LessThan(PositionTolerance));
            Assert.That(maxAngle, Is.LessThan(AngleToleranceDeg));
        }

        [Test]
        public void Wrong_joint_sign_is_detected()
        {
            // Negative control: the comparison above must be able to fail.
            var t = 250;
            PoseFromReference(reference, t, flipJoint: itf.JointNames.IndexOf("FL_thigh_joint"));
            var (pos, _, link) = LinkErrors(reference, t);
            Assert.That(pos, Is.GreaterThan(0.01f), $"flipping FL_thigh only moved {link} by {pos * 1000:F2} mm");
        }

        [Test]
        public void Positive_torque_increases_joint_angle()
        {
            // Base fixed, no gravity: a small positive effort on one joint must accelerate it in the + direction.
            var q = itf.DefaultJointPos.ToArray();
            var tau = new float[body.NumJoints];
            var qNow = new float[body.NumJoints];
            var qdNow = new float[body.NumJoints];
            var mode = Physics.simulationMode;
            var gravity = Physics.gravity;
            try
            {
                Physics.simulationMode = SimulationMode.Script;
                Physics.gravity = Vector3.zero;
                body.Root.immovable = true;
                for (var i = 0; i < body.NumJoints; i++)
                {
                    body.Teleport(Vector3.up, Quaternion.identity, q);
                    System.Array.Clear(tau, 0, tau.Length);
                    tau[i] = 0.05f;
                    body.SetJointTorques(tau);
                    Physics.Simulate(itf.Timing.PhysicsDt);
                    body.ReadJoints(qNow, qdNow);
                    Assert.That(qdNow[i], Is.GreaterThan(0f), $"{itf.JointNames[i]}: +effort gave qd {qdNow[i]}");
                }
            }
            finally
            {
                Physics.simulationMode = mode;
                Physics.gravity = gravity;
            }
        }

        /// <summary>Base pose and joint angles of frame t (converted to Unity), then one tiny physics step to publish link poses.</summary>
        void PoseFromReference(ReferenceRollout r, int t, int? flipJoint)
        {
            var basePos = IsaacFrame.PolarFromIsaac(new Vector3(r.root_link_pos_w[t, 0], r.root_link_pos_w[t, 1], r.root_link_pos_w[t, 2]));
            var baseRot = IsaacFrame.QuatFromIsaac(r.root_link_quat_w[t, 0], r.root_link_quat_w[t, 1], r.root_link_quat_w[t, 2], r.root_link_quat_w[t, 3]);
            var q = r.joint_pos.Row(t);
            if (flipJoint is int f)
                q[f] = -q[f];

            var mode = Physics.simulationMode;
            var gravity = Physics.gravity;
            try
            {
                Physics.simulationMode = SimulationMode.Script;
                Physics.gravity = Vector3.zero;
                body.Teleport(basePos, baseRot, q);
                Physics.Simulate(1e-6f);
            }
            finally
            {
                Physics.simulationMode = mode;
                Physics.gravity = gravity;
            }
        }

        (float position, float angleDeg, string link) LinkErrors(ReferenceRollout r, int t)
        {
            var nBodies = r.body_names.Length;
            float maxPos = 0f, maxAngle = 0f;
            var worst = "";
            for (var k = 0; k < nBodies; k++)
            {
                var o = t * nBodies + k;
                var p = r.body_link_pos_w.data;
                var expectedPos = IsaacFrame.PolarFromIsaac(new Vector3(p[3 * o], p[3 * o + 1], p[3 * o + 2]));
                var qd = r.body_link_quat_w.data;
                var expectedRot = IsaacFrame.QuatFromIsaac(qd[4 * o], qd[4 * o + 1], qd[4 * o + 2], qd[4 * o + 3]);
                var b = Body(r.body_names[k]).transform;
                var pos = (b.position - expectedPos).magnitude;
                if (pos > maxPos) (maxPos, worst) = (pos, r.body_names[k]);
                maxAngle = Mathf.Max(maxAngle, Quaternion.Angle(b.rotation, expectedRot));
            }
            return (maxPos, maxAngle, worst);
        }
    }
}
