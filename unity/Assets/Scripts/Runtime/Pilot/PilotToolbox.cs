using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace BMB
{
    /// <summary>A pending "wait until something worth a decision happens" for one tool call.</summary>
    public struct PilotWait
    {
        public long Attention;
        public double Start;
        public float MaxWait;
        public int OrderId;
    }

    /// <summary>
    /// The AI pilot's tools, independent of how they are called (MCP tools for Claude Code now, the in-game Claude API
    /// agent later): observe, walk_to, move, stop, plus the waiting logic. Orders are checked and clamped here and
    /// executed by <see cref="ClaudePilot"/>. A waiting call returns at the first attention event (arrival, end of a move,
    /// fall, reset, a player disturbance, a surprise alarm) or after its maximum wait, with a fresh observation.
    /// </summary>
    public sealed class PilotToolbox
    {
        /// <summary>Obstacles and ice are reported within this distance of the robot (m).</summary>
        public const float SenseRadius = 8f;
        public const float MaxMoveSeconds = 10f;
        public const float MaxWaitSeconds = 60f;
        public const float DefaultSpeed = 0.6f;
        /// <summary>Painted ground below this μ is reported as ice.</summary>
        public const float IceMu = 0.5f;
        const int MaxPendingEvents = 30;

        readonly ClaudePilot pilot;
        readonly List<PilotEvent> pending = new List<PilotEvent>();
        DisturbanceManager subscribedTo;

        public PilotToolbox(ClaudePilot owner) => pilot = owner;

        Go2Controller Controller => pilot.Controller;
        double Now => Controller != null && Controller.Initialized ? Controller.TotalTime : 0d;

        internal void Subscribe()
        {
            if (subscribedTo != null || pilot.Manager == null)
                return;
            subscribedTo = pilot.Manager;
            subscribedTo.Log.Added += Queue;
        }

        internal void Unsubscribe()
        {
            if (subscribedTo != null)
                subscribedTo.Log.Added -= Queue;
            subscribedTo = null;
        }

        void Queue(DisturbanceEvent e)
        {
            if (e.Type == "pilot")
                return;
            pending.Add(new PilotEvent { time_s = R((float)e.Time), type = e.Type, detail = PilotDetail(e) });
            if (pending.Count > MaxPendingEvents)
                pending.RemoveAt(0);
        }

        /// <summary>
        /// What the pilot is told about a logged event. Placing a hidden pit or breakable ice is a trap: the pilot learns
        /// that one was set, not where (the robot's height scan cannot see it either). Everything else is passed on.
        /// </summary>
        public static string PilotDetail(DisturbanceEvent e)
        {
            if (e.Type == "deceptive" && e.Detail != null && (e.Detail.StartsWith("hidden pit") || e.Detail.StartsWith("breakable ice")))
                return "the player set a hidden trap somewhere";
            return e.Detail;
        }

        // --- orders -------------------------------------------------------------------------------------------

        public PilotActionResult WalkTo(float x, float y, float speed = DefaultSpeed, string note = null)
        {
            if (!Ready(out var why))
                return Rejected(why);
            if (!float.IsFinite(x) || !float.IsFinite(y))
                return Rejected("target must be finite numbers");
            if (!float.IsFinite(speed) || speed <= 0f)
                return Rejected("speed must be > 0 m/s");
            var maxSpeed = Controller.Interface.Commands.LinVelX[1];
            speed = Mathf.Min(speed, maxSpeed);
            if (!InsideLevel(new Vector2(x, y), out why))
                return Rejected(why);
            var order = pilot.Issue(new PilotOrder { Kind = PilotOrderKind.WalkTo, Target = new Vector2(x, y), Speed = speed, Note = note });
            return Issued(order);
        }

        public PilotActionResult Move(float vx, float vy, float wz, float seconds, string note = null)
        {
            if (!Ready(out var why))
                return Rejected(why);
            if (!float.IsFinite(vx) || !float.IsFinite(vy) || !float.IsFinite(wz) || !float.IsFinite(seconds))
                return Rejected("vx, vy, wz and seconds must be finite numbers");
            if (seconds <= 0f || seconds > MaxMoveSeconds)
                return Rejected($"seconds must be in (0, {MaxMoveSeconds:0}]");
            var v = PilotDrive.Clamp(new Vector3(vx, vy, wz), Controller.Interface.Commands);
            var order = pilot.Issue(new PilotOrder { Kind = PilotOrderKind.Move, Velocity = v, Seconds = seconds, Note = note });
            var result = Issued(order);
            if (v != new Vector3(vx, vy, wz))
                result.detail += $" (clamped to the training ranges: vx {v.x:F2}, vy {v.y:F2}, wz {v.z:F2})";
            return result;
        }

        public PilotActionResult Stop(string note = null)
        {
            if (!Ready(out var why))
                return Rejected(why);
            var order = pilot.Issue(new PilotOrder { Kind = PilotOrderKind.Stand, Note = note });
            return Issued(order);
        }

        bool Ready(out string why)
        {
            why = null;
            if (Controller == null || !Controller.Initialized)
                why = "the robot is not running (enter Play mode in the Playground scene)";
            return why == null;
        }

        bool InsideLevel(Vector2 target, out string why)
        {
            why = null;
            var level = pilot.Manager != null ? pilot.Manager.Level : null;
            if (level == null)
                return true;
            var u = IsaacFrame.PolarFromIsaac(new Vector3(target.x, target.y, 0f)) - level.transform.position;
            var half = level.Size / 2f - Vector2.one * 0.5f;
            if (Mathf.Abs(u.x) <= half.x && Mathf.Abs(u.z) <= half.y)
                return true;
            var c = IsaacFrame.PolarToIsaac(level.transform.position);
            why = $"target is outside the level: x in [{c.x - half.y:F1}, {c.x + half.y:F1}], y in [{c.y - half.x:F1}, {c.y + half.x:F1}]";
            return false;
        }

        PilotActionResult Issued(PilotOrder order) => new PilotActionResult
        {
            accepted = true,
            outcome = "issued",
            detail = $"order #{order.Id}: {order.Describe()}",
        };

        PilotActionResult Rejected(string why) => new PilotActionResult
        {
            accepted = false,
            outcome = "rejected",
            detail = why,
            observation = Controller != null && Controller.Initialized ? Observe(false) : null,
        };

        // --- waiting ------------------------------------------------------------------------------------------

        /// <summary>Starts waiting after an accepted order (simulated seconds, capped at <see cref="MaxWaitSeconds"/>).</summary>
        public PilotWait BeginWait(float maxWaitSeconds) => new PilotWait
        {
            Attention = pilot.Attention,
            Start = Now,
            MaxWait = Mathf.Clamp(float.IsFinite(maxWaitSeconds) ? maxWaitSeconds : 0f, 0f, MaxWaitSeconds),
            OrderId = pilot.Order?.Id ?? 0,
        };

        /// <summary>True once something happened, the order ended, or the wait ran out.</summary>
        public bool Settled(PilotWait w)
        {
            if (pilot.Attention != w.Attention)
                return true;
            var o = pilot.Order;
            if (o == null || o.Id != w.OrderId || !o.Running)
                return true;
            return Now - w.Start >= w.MaxWait;
        }

        /// <summary>The result of a finished wait: why it returned, and a fresh observation.</summary>
        public PilotActionResult Conclude(PilotWait w, PilotActionResult issued)
        {
            var result = issued ?? new PilotActionResult { accepted = true };
            var o = pilot.Order;
            if (pilot.Attention != w.Attention)
            {
                result.outcome = pilot.AttentionCode;
                result.detail = pilot.AttentionDetail;
            }
            else if (o != null && o.Id == w.OrderId && !o.Running)
            {
                result.outcome = PilotOrder.StateName(o.State);
                result.detail = o.Detail;
            }
            else
            {
                result.outcome = "max_wait";
                result.detail = o != null && o.Running
                    ? $"order #{o.Id} still running after {Now - w.Start:F1} s"
                    : "nothing happened";
            }
            result.observation = Observe();
            return result;
        }

        // --- observation --------------------------------------------------------------------------------------

        /// <summary>Structured observation; <paramref name="drainEvents"/> clears the events it reports.</summary>
        public PilotObservation Observe(bool drainEvents = true)
        {
            var c = Controller;
            var obs = new PilotObservation { engaged = pilot.Engaged };
            if (c == null || !c.Initialized)
                return obs;

            var body = c.Body;
            var pose = PilotPose.Of(c);
            var v = IsaacFrame.PolarInBase(body.BaseRotation, body.BaseComLinearVelocity);
            var w = IsaacFrame.AxialInBase(body.BaseRotation, body.BaseAngularVelocity);
            obs.time_s = R((float)c.TotalTime);
            obs.x = R(pose.Position.x);
            obs.y = R(pose.Position.y);
            obs.yaw_deg = R(pose.Yaw * Mathf.Rad2Deg, 1);
            obs.vx = R(v.x);
            obs.vy = R(v.y);
            obs.wz = R(w.z);
            obs.base_height = R(c.BaseHeight, 3);
            obs.falls = c.Falls;

            if (pilot.Goal is Vector2 goal)
            {
                obs.has_goal = true;
                obs.goal_x = R(goal.x);
                obs.goal_y = R(goal.y);
                obs.goal_distance = R(pose.DistanceTo(goal));
                obs.goal_bearing_deg = R(pose.BearingTo(goal) * Mathf.Rad2Deg, 1);
            }

            var m = c.WorldModel;
            if (m != null && m.HasPrediction)
            {
                obs.fall_prob = m.Prediction.FallProb.Select(p => R(p, 3)).ToArray();
                var d = m.Detector;
                obs.surprise_sigma = R((float)d.X);
                obs.cusum_ratio = R((float)(d.GPeak / d.Params.H), 3);
                obs.surprise_alarms = d.Alarms;
            }

            obs.order = OrderInfo(pose);
            var zones = c.Zones;
            if (zones != null)
            {
                obs.ground_mu = R(zones.GlobalMu);
                obs.ice = zones.SamplePatches(body.BasePosition, SenseRadius, IceMu)
                    .Select(p => Ice(p, pose)).OrderBy(i => i.distance).ToArray();
            }
            var level = pilot.Manager != null ? pilot.Manager.Level : null;
            if (level != null)
                obs.obstacles = Obstacles(level, body.BasePosition, pose);
            obs.events = pending.ToArray();
            if (drainEvents)
                pending.Clear();
            return obs;
        }

        PilotOrderInfo OrderInfo(PilotPose pose)
        {
            var o = pilot.Order;
            if (o == null)
                return new PilotOrderInfo();
            return new PilotOrderInfo
            {
                id = o.Id,
                kind = PilotOrder.KindName(o.Kind),
                state = PilotOrder.StateName(o.State),
                detail = o.Detail ?? "",
                note = o.Note ?? "",
                target_x = R(o.Target.x),
                target_y = R(o.Target.y),
                speed = R(o.Speed),
                vx = R(o.Velocity.x),
                vy = R(o.Velocity.y),
                wz = R(o.Velocity.z),
                seconds = R(o.Seconds),
                elapsed_s = R(o.Elapsed(Now)),
                distance_left = o.Kind == PilotOrderKind.WalkTo ? R(pose.DistanceTo(o.Target)) : 0f,
            };
        }

        static PilotIce Ice(FrictionPatch p, PilotPose pose)
        {
            var c = IsaacFrame.PolarToIsaac(p.Centre);
            var at = new Vector2(c.x, c.y);
            return new PilotIce
            {
                x = R(at.x),
                y = R(at.y),
                radius = R(p.Radius),
                min_mu = R(p.MinMu),
                distance = R(Mathf.Max(0f, pose.DistanceTo(at) - p.Radius)),
                bearing_deg = R(pose.BearingTo(at) * Mathf.Rad2Deg, 1),
            };
        }

        /// <summary>
        /// Visible ground features near the robot. Hidden pits and intact breakable ice are traps the robot cannot see
        /// (its height scan reads flat ground), so they are left out; a broken ice slab shows as a pit.
        /// </summary>
        static PilotObstacle[] Obstacles(LevelGround level, Vector3 basePosition, PilotPose pose)
        {
            var origin = level.transform.position;
            var robot = new Vector2(basePosition.x - origin.x, basePosition.z - origin.z);
            var list = new List<PilotObstacle>();
            foreach (var f in level.Features)
            {
                var broken = f.Covered && level.Covers.Any(cv => cv.Feature == f && cv.Broken);
                if (f.Covered && !broken)
                    continue;
                var l = f.ToLocal(robot);
                var dx = Mathf.Max(0f, Mathf.Abs(l.x) - f.size.x / 2f);
                var dy = Mathf.Max(0f, Mathf.Abs(l.y) - f.size.y / 2f);
                var distance = Mathf.Sqrt(dx * dx + dy * dy);
                if (distance > SenseRadius)
                    continue;
                var c = IsaacFrame.PolarToIsaac(origin + new Vector3(f.centre.x, 0f, f.centre.y));
                var at = new Vector2(c.x, c.y);
                list.Add(new PilotObstacle
                {
                    type = broken ? "pit" : TypeName(f.type),
                    description = broken ? $"pit {f.size.x:F1}×{f.size.y:F1} m, {f.height * 100:F0} cm deep (broken ice)" : f.Describe(),
                    x = R(at.x),
                    y = R(at.y),
                    heading_deg = R(CommandSampler.Heading(Quaternion.Euler(0f, f.yawDeg, 0f)) * Mathf.Rad2Deg, 1),
                    length = R(f.size.y),
                    width = R(f.size.x),
                    height = R(f.height),
                    distance = R(distance),
                    bearing_deg = R(pose.BearingTo(at) * Mathf.Rad2Deg, 1),
                });
            }
            return list.OrderBy(o => o.distance).ToArray();
        }

        static string TypeName(GroundFeatureType t) => t switch
        {
            GroundFeatureType.IsaacTile => "isaac_tile",
            _ => t.ToString().ToLowerInvariant(),
        };

        static float R(float v, int digits = 2)
        {
            if (!float.IsFinite(v))
                return 0f;
            var s = Mathf.Pow(10f, digits);
            return Mathf.Round(v * s) / s;
        }
    }
}
