using UnityEngine;

namespace BMB
{
    /// <summary>Base pose on the ground plane, Isaac world axes: x, y (m), yaw (rad, counter-clockwise from +x).</summary>
    public struct PilotPose
    {
        public Vector2 Position;
        public float Yaw;

        public PilotPose(Vector2 position, float yaw)
        {
            Position = position;
            Yaw = yaw;
        }

        public static PilotPose FromUnity(Vector3 basePosition, Quaternion baseRotation)
        {
            var p = IsaacFrame.PolarToIsaac(basePosition);
            return new PilotPose(new Vector2(p.x, p.y), CommandSampler.Heading(baseRotation));
        }

        public static PilotPose Of(Go2Controller c) => FromUnity(c.Body.BasePosition, c.Body.BaseRotation);

        /// <summary>Bearing of a world point from the base: angle left of the heading (rad, −π..π).</summary>
        public float BearingTo(Vector2 target)
        {
            var d = target - Position;
            return PilotDrive.Wrap(Mathf.Atan2(d.y, d.x) - Yaw);
        }

        public float DistanceTo(Vector2 target) => (target - Position).magnitude;
    }

    /// <summary>
    /// The pilot's low-level driving: turns a high-level order into a velocity command (Isaac base frame vx, vy, wz)
    /// for the locomotion policy, every policy step. Pure functions, shared by the real robot and the world-model
    /// imagination, so an imagined order is driven exactly like the real one.
    /// </summary>
    public static class PilotDrive
    {
        /// <summary>Distance at which a walk-to target counts as reached (m).</summary>
        public const float ArriveRadius = 0.3f;
        /// <summary>Beyond this heading error the robot turns on the spot before walking.</summary>
        public const float TurnInPlaceRad = 60f * Mathf.Deg2Rad;
        /// <summary>Yaw rate per radian of heading error (rad/s per rad), clamped to the command range.</summary>
        public const float HeadingGain = 1.5f;
        /// <summary>Within this distance the forward speed ramps down linearly (m).</summary>
        public const float SlowRadius = 1.0f;
        /// <summary>Lowest forward speed while approaching a target (m/s), so the robot does not stall short of it.</summary>
        public const float MinApproachSpeed = 0.2f;

        public static float Wrap(float a) => Mathf.Repeat(a + Mathf.PI, 2f * Mathf.PI) - Mathf.PI;

        /// <summary>Command that walks toward <paramref name="target"/> at up to <paramref name="speed"/>; zero once there.</summary>
        public static Vector3 WalkTo(PilotPose pose, Vector2 target, float speed, CommandsSpec ranges)
        {
            var distance = pose.DistanceTo(target);
            if (distance <= ArriveRadius)
                return Vector3.zero;
            var bearing = pose.BearingTo(target);
            var wz = HeadingGain * bearing;
            var forward = 0f;
            if (Mathf.Abs(bearing) < TurnInPlaceRad)
            {
                var approach = Mathf.Max(MinApproachSpeed, speed * Mathf.Clamp01(distance / SlowRadius));
                forward = Mathf.Min(speed, approach) * Mathf.Cos(bearing);
            }
            return Clamp(new Vector3(forward, 0f, wz), ranges);
        }

        /// <summary>Clamps each component to the yaml's command ranges (the training ranges).</summary>
        public static Vector3 Clamp(Vector3 command, CommandsSpec ranges) => new Vector3(
            Mathf.Clamp(command.x, ranges.LinVelX[0], ranges.LinVelX[1]),
            Mathf.Clamp(command.y, ranges.LinVelY[0], ranges.LinVelY[1]),
            Mathf.Clamp(command.z, ranges.AngVelZ[0], ranges.AngVelZ[1]));

        /// <summary>
        /// The cautious gait (policies.v2.cautious_gait): |(vx, vy)| scaled down to <paramref name="maxLin"/>,
        /// |wz| clamped to <paramref name="maxAng"/>.
        /// </summary>
        public static Vector3 Cautious(Vector3 command, float maxLin, float maxAng)
        {
            var v = new Vector2(command.x, command.y);
            if (v.magnitude > maxLin)
                v = v.normalized * maxLin;
            return new Vector3(v.x, v.y, Mathf.Clamp(command.z, -maxAng, maxAng));
        }
    }
}
