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
        public static float Torque(float qTarget, float q, float qd, ActuatorSpec a) => Torque(qTarget, q, qd, a, 1f, 1f);

        /// <summary>
        /// With the gains scaled (v2's randomize_actuator_gains, operation "scale": Kp·kpScale, Kd·kdScale);
        /// a scale of 1 gives exactly the unscaled torque.
        /// </summary>
        public static float Torque(float qTarget, float q, float qd, ActuatorSpec a, float kpScale, float kdScale)
        {
            var tau = a.Stiffness * kpScale * (qTarget - q) - a.Damping * kdScale * qd;
            var vAtEffortLimit = a.VelocityLimit * (1f + a.EffortLimit / a.SaturationEffort);
            var qdClipped = Mathf.Clamp(qd, -vAtEffortLimit, vAtEffortLimit);
            var tauMax = Mathf.Min(a.SaturationEffort * (1f - qdClipped / a.VelocityLimit), a.EffortLimit);
            var tauMin = Mathf.Max(a.SaturationEffort * (-1f - qdClipped / a.VelocityLimit), -a.EffortLimit);
            return Mathf.Clamp(tau, tauMin, tauMax);
        }

        /// <summary>All joints at once (arrays in policy joint order); null scales = 1.</summary>
        public static void Torques(float[] qTarget, float[] q, float[] qd, ActuatorSpec a, float[] tauOut,
                                   float[] kpScale = null, float[] kdScale = null)
        {
            for (var i = 0; i < tauOut.Length; i++)
                tauOut[i] = Torque(qTarget[i], q[i], qd[i], a, kpScale?[i] ?? 1f, kdScale?[i] ?? 1f);
        }
    }
}
