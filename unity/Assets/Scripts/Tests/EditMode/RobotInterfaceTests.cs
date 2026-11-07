using System.Linq;
using NUnit.Framework;

namespace BMB.Tests
{
    /// <summary>The yaml is internally consistent and agrees with what Isaac Lab actually simulated.</summary>
    public class RobotInterfaceTests
    {
        RobotInterface itf;

        [OneTimeSetUp]
        public void Load() => itf = RobotInterface.Load();

        [Test]
        public void Observation_dims_add_up()
        {
            var terms = itf.Observation.Terms;
            Assert.AreEqual(itf.Observation.Dim["flat"], terms.Where(t => !t.RoughOnly).Sum(t => t.Dim));
            Assert.AreEqual(itf.Observation.Dim["rough"], terms.Sum(t => t.Dim));
            Assert.AreEqual(itf.Observation.Dim["flat"], itf.ObsDim("flat"));
            Assert.AreEqual(itf.Observation.Dim["rough"], itf.ObsDim("rough"));
            Assert.AreEqual(itf.NumJoints, itf.Observation.Term("joint_pos").Dim);
            Assert.AreEqual(itf.NumJoints, itf.Observation.Term("actions").Dim);
        }

        [Test]
        public void V2_obs_dim_matches_its_groups()
        {
            var v2 = itf.Policies["v2"];
            var dim = v2.Groups.Sum(g => System.Math.Max(1, g.HistoryLength) * g.Terms.Sum(n => itf.Observation.Term(n).Dim));
            Assert.AreEqual(v2.ObsDim, dim);
        }

        [Test]
        public void Height_scan_grid_matches_term()
        {
            var hs = itf.HeightScan;
            Assert.AreEqual(itf.Observation.Term("height_scan").Dim, hs.GridShape[0] * hs.GridShape[1]);
            Assert.AreEqual(hs.GridSize[0], (hs.GridShape[0] - 1) * hs.Resolution, 1e-5);
            Assert.AreEqual(hs.GridSize[1], (hs.GridShape[1] - 1) * hs.Resolution, 1e-5);
        }

        [TestCase("flat")]
        [TestCase("rough")]
        [TestCase("v2")]
        public void Reference_rollout_matches_interface(string variant)
        {
            var r = ReferenceRollout.Load(variant);
            CollectionAssert.AreEqual(itf.JointNames, r.joint_names);
            for (var i = 0; i < itf.NumJoints; i++)
            {
                Assert.AreEqual(itf.DefaultJointPos[i], r.default_joint_pos.data[i], 1e-6, "default_joint_pos");
                Assert.AreEqual(itf.Actuator.Stiffness, r.stiffness.data[i], 1e-6, "stiffness");
                Assert.AreEqual(itf.Actuator.Damping, r.damping.data[i], 1e-6, "damping");
                Assert.AreEqual(itf.Actuator.EffortLimit, r.effort_limit.data[i], 1e-6, "effort_limit");
                Assert.AreEqual(itf.Actuator.VelocityLimit, r.velocity_limit.data[i], 1e-6, "velocity_limit");
            }
            Assert.AreEqual(itf.Actuator.SaturationEffort, r.saturation_effort, 1e-6);
            Assert.AreEqual(itf.Timing.PhysicsDt, r.physics_dt, 1e-6);
            Assert.AreEqual(itf.Timing.PolicyDt, r.step_dt, 1e-6);
            Assert.AreEqual(itf.ObsDim(variant), r.obs.RowSize);
        }
    }
}
