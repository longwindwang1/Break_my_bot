using NUnit.Framework;

namespace BMB.Tests
{
    /// <summary>
    /// DC motor torque-speed curve (robot_interface.yaml `actuator`). Cases are written for the current Go2 values
    /// (Kp 25, Kd 0.5, effort = saturation = 23.5, velocity limit 30); the first test pins those values so a yaml
    /// change shows up here before the expected numbers silently go stale.
    /// </summary>
    public class DCMotorTests
    {
        ActuatorSpec a;

        [OneTimeSetUp]
        public void Load() => a = RobotInterface.Load().Actuator;

        [Test]
        public void Actuator_values_are_the_ones_these_cases_assume()
        {
            Assert.AreEqual("dc_motor", a.Model);
            Assert.AreEqual(25f, a.Stiffness);
            Assert.AreEqual(0.5f, a.Damping);
            Assert.AreEqual(23.5f, a.EffortLimit);
            Assert.AreEqual(23.5f, a.SaturationEffort);
            Assert.AreEqual(30f, a.VelocityLimit);
        }

        // qTarget, q, qd -> expected torque
        [TestCase(0.1f, 0f, 0f, 2.5f, TestName = "PD only, small error")]
        [TestCase(0f, 0f, 2f, -1f, TestName = "PD only, damping")]
        [TestCase(10f, 0f, 0f, 23.5f, TestName = "Stall: clipped to effort limit")]
        [TestCase(-10f, 0f, 0f, -23.5f, TestName = "Stall: clipped to minus effort limit")]
        [TestCase(10f, 0f, 15f, 11.75f, TestName = "Half speed: max torque halves")]
        [TestCase(-10f, 0f, -15f, -11.75f, TestName = "Half reverse speed: min torque halves")]
        [TestCase(-10f, 0f, 15f, -23.5f, TestName = "Braking against motion: full effort")]
        [TestCase(10f, 0f, 40f, -7.833333f, TestName = "Above velocity limit: only braking torque")]
        [TestCase(0f, 0f, 100f, -23.5f, TestName = "Beyond v at effort limit: speed clipped to 60")]
        public void Torque(float qTarget, float q, float qd, float expected)
        {
            Assert.AreEqual(expected, DCMotor.Torque(qTarget, q, qd, a), 1e-4f);
        }
    }
}
