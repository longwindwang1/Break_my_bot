using System;
using System.Collections.Generic;
using UnityEngine;

namespace BMB
{
    /// <summary>
    /// "Think before you walk": imagines an order in the world model with the policy in the loop (a
    /// <see cref="WorldModelDreamer"/> whose command comes from <see cref="PilotDrive"/> at every imagined pose, exactly
    /// as the real executor drives), synchronously, and reports where the robot ends up and the highest predicted fall
    /// risk. Used by the pilot's imagine tool and by the veto that rejects orders the world model expects to end in a
    /// fall. The world model only sees geometry (height scan), not friction: it cannot foresee ice.
    /// </summary>
    public sealed class PilotImagination
    {
        /// <summary>Longest imagination (s); accuracy was measured up to 2 s (experiments/claude_pilot).</summary>
        public const float MaxSeconds = 3f;
        /// <summary>Spacing of the reported path points (s).</summary>
        public const float PathSpacing = 0.25f;

        readonly ClaudePilot pilot;
        WorldModelDreamer dreamer;

        public PilotImagination(ClaudePilot owner) => pilot = owner;

        public bool Available
        {
            get
            {
                var c = pilot.Controller;
                return c != null && c.Initialized && c.WorldModel != null && c.WorldModel.HasPrediction;
            }
        }

        /// <summary>The imagined base path of the last run (Unity frame), for drawing.</summary>
        public Vector3[] LastPath => dreamer?.Path ?? Array.Empty<Vector3>();

        /// <summary>Imagines <paramref name="order"/> for <paramref name="seconds"/>; null if no world model / prediction.</summary>
        public PilotImagined Run(PilotOrder order, float seconds)
        {
            if (!Available)
                return null;
            var c = pilot.Controller;
            dreamer ??= new WorldModelDreamer(c);
            var dt = c.Interface.Timing.PolicyDt;
            seconds = Mathf.Clamp(seconds, dt, MaxSeconds);
            dreamer.Horizon = Mathf.Max(1, Mathf.RoundToInt(seconds / dt));
            var cautious = pilot.GuardActive;
            var driver = new DreamDriver(pilot, order, cautious, 0f);
            if (!dreamer.Begin(driver.Command))
                return null;
            dreamer.Advance();
            var arrived = driver.Arrived;

            var end = IsaacFrame.PolarToIsaac(dreamer.GhostPosition);
            var endPose = new PilotPose(new Vector2(end.x, end.y), dreamer.GhostYaw);
            var result = new PilotImagined
            {
                order = order.Describe(),
                seconds = R(dreamer.Horizon * dt),
                end_x = R(end.x),
                end_y = R(end.y),
                end_yaw_deg = R(dreamer.GhostYaw * Mathf.Rad2Deg, 1),
                max_fall_prob = Array.ConvertAll(dreamer.MaxFallProbs, p => R(p, 3)),
                cautious = cautious,
            };
            if (order.Kind == PilotOrderKind.WalkTo)
            {
                result.arrived = arrived || endPose.DistanceTo(order.Target) <= PilotDrive.ArriveRadius;
                result.distance_left = R(endPose.DistanceTo(order.Target));
            }
            var path = new List<PilotPoint>();
            var every = Mathf.Max(1, Mathf.RoundToInt(PathSpacing / dt));
            for (var i = every; i < dreamer.Path.Length; i += every)
            {
                var q = IsaacFrame.PolarToIsaac(dreamer.Path[i]);
                path.Add(new PilotPoint { t = R(i * dt), x = R(q.x), y = R(q.y) });
            }
            result.path = path.ToArray();
            return result;
        }

        static float R(float v, int digits = 2)
        {
            if (!float.IsFinite(v))
                return 0f;
            var s = Mathf.Pow(10f, digits);
            return Mathf.Round(v * s) / s;
        }
    }

    /// <summary>
    /// Drives an imagined robot like <see cref="ClaudePilot"/> drives the real one: the command for each imagined step
    /// from the imagined pose (Unity position, Isaac yaw), for a <see cref="WorldModelDreamer"/>.
    /// </summary>
    public sealed class DreamDriver
    {
        readonly ClaudePilot pilot;
        readonly PilotOrder order;
        readonly bool cautious;
        readonly float elapsedAtStart;
        readonly float dt;
        readonly CommandsSpec ranges;
        int k;

        /// <summary>The imagined robot reached the walk-to target.</summary>
        public bool Arrived { get; private set; }

        /// <param name="elapsedAtStart">Seconds the order had already run when the dream starts (move orders).</param>
        public DreamDriver(ClaudePilot pilot, PilotOrder order, bool cautious, float elapsedAtStart)
        {
            this.pilot = pilot;
            this.order = order;
            this.cautious = cautious;
            this.elapsedAtStart = elapsedAtStart;
            dt = pilot.Controller.Interface.Timing.PolicyDt;
            ranges = pilot.Controller.Interface.Commands;
        }

        public Vector3 Command(Vector3 position, float yaw)
        {
            k++;
            var p = IsaacFrame.PolarToIsaac(position);
            var pose = new PilotPose(new Vector2(p.x, p.y), yaw);
            var cmd = Vector3.zero;
            switch (order.Kind)
            {
                case PilotOrderKind.WalkTo:
                    Arrived |= pose.DistanceTo(order.Target) <= PilotDrive.ArriveRadius;
                    if (!Arrived)
                        cmd = PilotDrive.WalkTo(pose, order.Target, order.Speed, ranges);
                    break;
                case PilotOrderKind.Move:
                    if (elapsedAtStart + k * dt <= order.Seconds + 1e-4f)
                        cmd = order.Velocity;
                    break;
            }
            return cautious ? pilot.CautiousCap(cmd) : cmd;
        }
    }

    [Serializable]
    public sealed class PilotImagined
    {
        public string order;
        public float seconds;
        public float end_x;
        public float end_y;
        public float end_yaw_deg;
        /// <summary>Walk-to only: reached within the imagined time, and the distance left at its end.</summary>
        public bool arrived;
        public float distance_left;
        /// <summary>Highest predicted P(fall within 0.2 / 0.5 / 1 s) along the imagined path.</summary>
        public float[] max_fall_prob = Array.Empty<float>();
        /// <summary>Imagined under the cautious gait cap (a surprise alarm was recent).</summary>
        public bool cautious;
        public PilotPoint[] path = Array.Empty<PilotPoint>();
    }

    [Serializable]
    public sealed class PilotPoint
    {
        public float t;
        public float x;
        public float y;
    }
}
