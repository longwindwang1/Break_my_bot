using System;

// Data sent to the AI pilot (serialized to JSON by the MCP tools / the API agent). Field names are the JSON keys,
// so they are snake_case. Positions are Isaac world (x, y) in metres; yaw and bearings in degrees, counter-clockwise
// (a positive bearing is to the robot's left); velocities in the robot's base frame (vx forward, vy left, wz left turn).
namespace BMB
{
    [Serializable]
    public sealed class PilotObservation
    {
        public float time_s;
        public float x;
        public float y;
        public float yaw_deg;
        public float vx;
        public float vy;
        public float wz;
        public float base_height;
        public int falls;
        public bool engaged;
        /// <summary>The safety guard holds the cautious gait (after a surprise alarm) for this many more seconds.</summary>
        public bool guard_active;
        public float guard_seconds_left;

        public bool has_goal;
        public float goal_x;
        public float goal_y;
        public float goal_distance;
        public float goal_bearing_deg;
        /// <summary>A challenge round is running, with this much time left (s).</summary>
        public bool challenge_running;
        public float time_left_s;

        /// <summary>World model: P(fall within 0.2 / 0.5 / 1 s); empty without a world model.</summary>
        public float[] fall_prob = Array.Empty<float>();
        /// <summary>World model: surprise score (σ) of the last step and CUSUM g / h (alarm at 1).</summary>
        public float surprise_sigma;
        public float cusum_ratio;
        public int surprise_alarms;

        public PilotOrderInfo order;
        /// <summary>Ground μ everywhere nothing is painted (1 = normal ground).</summary>
        public float ground_mu = 1f;
        public PilotObstacle[] obstacles = Array.Empty<PilotObstacle>();
        public PilotIce[] ice = Array.Empty<PilotIce>();
        /// <summary>Everything logged since the previous observation (pushes, balls, ice, obstacles, alarms, …).</summary>
        public PilotEvent[] events = Array.Empty<PilotEvent>();
    }

    [Serializable]
    public sealed class PilotOrderInfo
    {
        public int id;
        public string kind = "none";
        public string state = "none";
        public string detail = "";
        public string note = "";
        public float target_x;
        public float target_y;
        public float speed;
        public float vx;
        public float vy;
        public float wz;
        public float seconds;
        public float elapsed_s;
        public float distance_left;
    }

    [Serializable]
    public sealed class PilotObstacle
    {
        public string type;
        public string description;
        public float x;
        public float y;
        /// <summary>Direction of the obstacle's length axis.</summary>
        public float heading_deg;
        public float length;
        public float width;
        /// <summary>Height (box, step, ramp rise) or depth (pit), m.</summary>
        public float height;
        /// <summary>From the robot to the obstacle's footprint (0 = on it).</summary>
        public float distance;
        public float bearing_deg;
    }

    [Serializable]
    public sealed class PilotIce
    {
        public float x;
        public float y;
        public float radius;
        public float min_mu;
        public float distance;
        public float bearing_deg;
    }

    [Serializable]
    public sealed class PilotEvent
    {
        public float time_s;
        public string type;
        public string detail;
    }

    [Serializable]
    public sealed class PilotActionResult
    {
        /// <summary>False if the order was rejected (bad arguments, world-model veto, not playing).</summary>
        public bool accepted;
        /// <summary>
        /// Why the call returned: rejected, vetoed, imagined, issued (did not wait), arrived, done, halted (lookahead),
        /// alarm, fell, reset, disturbed (the player did something), released, replaced, max_wait (still running).
        /// </summary>
        public string outcome;
        public string detail;
        public PilotObservation observation;
        /// <summary>The world model's imagination of the order (first second for the veto, or the imagine tool's span).</summary>
        public PilotImagined imagined;
    }
}
