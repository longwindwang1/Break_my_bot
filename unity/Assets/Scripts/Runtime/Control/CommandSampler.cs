using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Isaac Lab's UniformVelocityCommand in heading mode (robot_interface.yaml `commands`): every resampling period
    /// draw vx, vy uniformly from their ranges and a heading target uniformly in [−π, π] (2 % of draws stand still);
    /// every policy step wz = clip(stiffness · wrap(heading_target − heading), ang_vel_z range).
    /// Used by the headless evaluation; same distributions as numpy, different samples.
    /// </summary>
    public sealed class CommandSampler
    {
        readonly System.Random rng;
        float timeLeft;
        Vector3 command;
        float headingTarget;

        public Vector2 LinVelX { get; set; }
        public Vector2 LinVelY { get; set; }
        public Vector2 AngVelZ { get; set; }
        public float Stiffness { get; set; }
        public float ResampleSeconds { get; set; }
        /// <summary>Fraction of draws that command zero velocity (Isaac's rel_standing_envs).</summary>
        public float StandingProbability { get; set; } = 0.02f;
        public bool Standing { get; private set; }
        public float HeadingTarget => headingTarget;
        /// <summary>The command returned by the last <see cref="Next"/> (Isaac base frame: vx, vy, wz).</summary>
        public Vector3 Current => Standing ? Vector3.zero : command;

        public CommandSampler(CommandsSpec spec, System.Random random)
        {
            rng = random;
            LinVelX = new Vector2(spec.LinVelX[0], spec.LinVelX[1]);
            LinVelY = new Vector2(spec.LinVelY[0], spec.LinVelY[1]);
            AngVelZ = new Vector2(spec.AngVelZ[0], spec.AngVelZ[1]);
            Stiffness = spec.HeadingControlStiffness;
            ResampleSeconds = spec.ResamplingTimeS;
        }

        /// <summary>The cautious gait: |v| ≤ maxLin on both axes, |wz| ≤ maxAng (policies.v2.cautious_gait).</summary>
        public void Cap(float maxLin, float maxAng)
        {
            LinVelX = new Vector2(-maxLin, maxLin);
            LinVelY = new Vector2(-maxLin, maxLin);
            AngVelZ = new Vector2(-maxAng, maxAng);
        }

        /// <summary>Forces a new draw on the next call (episode reset).</summary>
        public void Reset() => timeLeft = 0f;

        /// <summary>Advances by <paramref name="dt"/> (one policy step) and returns the command for the robot's yaw.</summary>
        public Vector3 Next(float dt, float headingIsaac)
        {
            timeLeft -= dt;
            if (timeLeft <= 0f)
            {
                Resample();
                timeLeft += ResampleSeconds;
                if (timeLeft <= 0f)
                    timeLeft = ResampleSeconds;
            }
            var error = Mathf.Repeat(headingTarget - headingIsaac + Mathf.PI, 2f * Mathf.PI) - Mathf.PI;
            command.z = Mathf.Clamp(Stiffness * error, AngVelZ.x, AngVelZ.y);
            return Current;
        }

        void Resample()
        {
            command.x = Uniform(LinVelX);
            command.y = Uniform(LinVelY);
            command.z = Uniform(AngVelZ);
            headingTarget = Uniform(new Vector2(-Mathf.PI, Mathf.PI));
            Standing = rng.NextDouble() <= StandingProbability;
        }

        float Uniform(Vector2 range) => range.x + (float)rng.NextDouble() * (range.y - range.x);

        /// <summary>Isaac's heading_w: yaw of the base x axis in the Isaac world frame (rad).</summary>
        public static float Heading(Quaternion unityBaseRotation)
        {
            var f = IsaacFrame.PolarToIsaac(unityBaseRotation * Vector3.forward);
            return Mathf.Atan2(f.y, f.x);
        }
    }
}
