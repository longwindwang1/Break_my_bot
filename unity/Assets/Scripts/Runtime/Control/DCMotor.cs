using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Explicit DC motor of Isaac Lab (isaaclab.actuators.DCMotor), formula from robot_interface.yaml `actuator`:
    /// PD torque, then a torque-speed saturation curve. Computed every physics step with the current q, qd;
    /// the ArticulationBody drives stay at zero stiffness/damping so PhysX adds no PD of its own.
    /// </summary>
    public static class DCMotor
    {
        public static float Torque(float qTarget, float q, float qd, ActuatorSpec a)
        {
            var tau = a.Stiffness * (qTarget - q) - a.Damping * qd;
            var vAtEffortLimit = a.VelocityLimit * (1f + a.EffortLimit / a.SaturationEffort);
            var qdClipped = Mathf.Clamp(qd, -vAtEffortLimit, vAtEffortLimit);
            var tauMax = Mathf.Min(a.SaturationEffort * (1f - qdClipped / a.VelocityLimit), a.EffortLimit);
            var tauMin = Mathf.Max(a.SaturationEffort * (-1f - qdClipped / a.VelocityLimit), -a.EffortLimit);
            return Mathf.Clamp(tau, tauMin, tauMax);
        }

        /// <summary>All joints at once (arrays in policy joint order).</summary>
        public static void Torques(float[] qTarget, float[] q, float[] qd, ActuatorSpec a, float[] tauOut)
        {
            for (var i = 0; i < tauOut.Length; i++)
                tauOut[i] = Torque(qTarget[i], q[i], qd[i], a);
        }
    }
}
