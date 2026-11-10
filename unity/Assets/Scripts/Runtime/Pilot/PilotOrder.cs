using UnityEngine;

namespace BMB
{
    public enum PilotOrderKind
    {
        /// <summary>Stand still (zero command).</summary>
        Stand,
        /// <summary>Walk to a point on the ground (Isaac world x, y).</summary>
        WalkTo,
        /// <summary>Hold a base-frame velocity command (vx, vy, wz) for some seconds.</summary>
        Move,
    }

    public enum PilotOrderState
    {
        Running,
        /// <summary>Walk-to target reached.</summary>
        Arrived,
        /// <summary>Move order ran for its duration.</summary>
        Done,
        /// <summary>Stand order (immediately settled).</summary>
        Standing,
        /// <summary>A newer order took over.</summary>
        Replaced,
        Fell,
        /// <summary>The robot was reset to its spawn pose (player, R key).</summary>
        Reset,
        /// <summary>Interrupted by a world-model surprise alarm.</summary>
        Alarm,
        /// <summary>The pilot handed control back to the player.</summary>
        Released,
    }

    /// <summary>One high-level order from the AI pilot and how it went.</summary>
    public sealed class PilotOrder
    {
        public int Id;
        public PilotOrderKind Kind;
        /// <summary>Walk-to target, Isaac world (x, y) m.</summary>
        public Vector2 Target;
        /// <summary>Walk-to speed (m/s).</summary>
        public float Speed;
        /// <summary>Move command, Isaac base frame (vx, vy m/s, wz rad/s), already clamped.</summary>
        public Vector3 Velocity;
        /// <summary>Move duration (s).</summary>
        public float Seconds;
        /// <summary>The pilot's reason, shown on the HUD.</summary>
        public string Note;
        public double IssuedAt;
        public PilotOrderState State = PilotOrderState.Running;
        public string Detail = "";
        public double FinishedAt = double.NaN;

        public bool Running => State == PilotOrderState.Running;

        public float Elapsed(double now) => (float)((Running ? now : FinishedAt) - IssuedAt);

        public static string KindName(PilotOrderKind k) => k switch
        {
            PilotOrderKind.WalkTo => "walk_to",
            PilotOrderKind.Move => "move",
            _ => "stand",
        };

        public static string StateName(PilotOrderState s) => s.ToString().ToLowerInvariant();

        public string Describe() => Kind switch
        {
            PilotOrderKind.WalkTo => $"walk to ({Target.x:F1}, {Target.y:F1}) at {Speed:F1} m/s",
            PilotOrderKind.Move => $"move vx {Velocity.x:F2} vy {Velocity.y:F2} wz {Velocity.z:F2} for {Seconds:F1} s",
            _ => "stand",
        };
    }
}
