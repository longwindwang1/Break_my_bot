using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>Project settings match robot_interface.yaml / SimSettings. Fix with BMB/Apply interface settings.</summary>
    public class ProjectSettingsTests
    {
        [Test]
        public void Physics_step_matches_interface()
        {
            // Exact to float precision (Unity 6 stores the step as a rational; see ApplyInterfaceSettings).
            Assert.AreEqual(RobotInterface.Load().Timing.PhysicsDt, Time.fixedDeltaTime, 1e-9f);
            Assert.AreEqual(SimulationMode.FixedUpdate, Physics.simulationMode);
        }

        [Test]
        public void Solver_iterations()
        {
            Assert.AreEqual(SimSettings.SolverIterations, Physics.defaultSolverIterations);
            Assert.AreEqual(SimSettings.SolverVelocityIterations, Physics.defaultSolverVelocityIterations);
        }

        [Test]
        public void Layers_exist_and_robot_does_not_self_collide()
        {
            var terrain = LayerMask.NameToLayer(SimSettings.TerrainLayer);
            var robot = LayerMask.NameToLayer(SimSettings.RobotLayer);
            Assert.GreaterOrEqual(terrain, 0, "Terrain layer missing");
            Assert.GreaterOrEqual(robot, 0, "Robot layer missing");
            Assert.IsTrue(Physics.GetIgnoreLayerCollision(robot, robot), "Robot×Robot collisions must be off");
            Assert.IsFalse(Physics.GetIgnoreLayerCollision(robot, terrain), "Robot must collide with Terrain");
            var scanOnly = LayerMask.NameToLayer(SimSettings.ScanOnlyLayer);
            Assert.GreaterOrEqual(scanOnly, 0, "ScanOnly layer missing");
            for (var other = 0; other < 32; other++)
                Assert.IsTrue(Physics.GetIgnoreLayerCollision(scanOnly, other), $"ScanOnly must not collide with layer {other}");
        }
    }
}
