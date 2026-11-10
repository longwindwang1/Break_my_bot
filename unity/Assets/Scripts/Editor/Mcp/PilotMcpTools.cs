using System;
using System.Threading.Tasks;
using MCPForUnity.Editor.Helpers;
using MCPForUnity.Editor.Tools;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEngine;

// MCP tools that let Claude Code drive the Go2 in the Playground (Play mode) as an AI pilot. Thin wrappers around
// BMB.PilotToolbox (Runtime), which the in-game Claude API agent will share. Registered with MCP for Unity through
// [McpForUnityTool]; the server picks up their parameter schemas at start-up (restart it after changing them).
namespace BMB.Editor.Mcp
{
    /// <summary>Finding the pilot, reading parameters, waiting for the pilot's attention events.</summary>
    public static class PilotMcp
    {
        public const string Frames =
            "Coordinates: Isaac world frame, x and y in metres on the ground plane, yaw and bearings in degrees counter-clockwise " +
            "(a positive bearing is to the robot's LEFT). Velocities vx (forward), vy (left) in m/s and wz (turn left) in rad/s, " +
            "robot base frame, clamped to the training ranges ±1.";

        /// <summary>For tests: a pilot to use outside Play mode (headless world).</summary>
        public static ClaudePilot Override { get; set; }

        public static ClaudePilot Find(out object error)
        {
            error = null;
            if (Override != null)
                return Override;
            if (!EditorApplication.isPlaying)
            {
                error = new ErrorResponse("not_playing", new { hint = "Open Assets/Scenes/Playground.unity and enter Play mode (BMB/Build Playground scene if it has no Pilot object)." });
                return null;
            }
            var pilot = ClaudePilot.Active != null ? ClaudePilot.Active : UnityEngine.Object.FindFirstObjectByType<ClaudePilot>();
            if (pilot == null || pilot.Controller == null || !pilot.Controller.Initialized)
                error = new ErrorResponse("no_pilot", new { hint = "The scene has no running ClaudePilot; rebuild the Playground scene (BMB/Build Playground scene)." });
            return error == null ? pilot : null;
        }

        public static float? F(JObject p, string key)
        {
            var t = p?[key];
            if (t == null || t.Type == JTokenType.Null)
                return null;
            return t.Type is JTokenType.Float or JTokenType.Integer ? t.Value<float>()
                : float.TryParse(t.ToString(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : float.NaN;
        }

        public static int? I(JObject p, string key)
        {
            var f = F(p, key);
            return f.HasValue && float.IsFinite(f.Value) ? Mathf.RoundToInt(f.Value) : null;
        }

        public static bool? B(JObject p, string key)
        {
            var t = p?[key];
            if (t == null || t.Type == JTokenType.Null)
                return null;
            return t.Type == JTokenType.Boolean ? t.Value<bool>() : bool.TryParse(t.ToString(), out var b) && b;
        }

        public static string S(JObject p, string key) => p?[key]?.Type == JTokenType.String ? p[key].Value<string>() : p?[key]?.ToString();

        public static object Respond(PilotActionResult r) =>
            r.accepted ? new SuccessResponse($"{r.outcome}: {r.detail}", r) : new ErrorResponse(r.outcome, r);

        /// <summary>
        /// After an accepted order, waits (simulated seconds, also capped in wall-clock time below the MCP timeout) for
        /// the pilot's next attention event and returns why it stopped waiting plus a fresh observation.
        /// </summary>
        public static async Task<object> WaitAndRespond(ClaudePilot pilot, PilotActionResult issued, JObject p)
        {
            if (!issued.accepted)
                return Respond(issued);
            var maxWait = F(p, "max_wait_s") ?? 8f;
            if (!(maxWait > 0f))
            {
                issued.observation = pilot.Toolbox.Observe();
                return Respond(issued);
            }
            var timeout = F(p, "timeout_seconds") ?? 30f;
            var wallCap = Mathf.Min(maxWait * 2f + 2f, Mathf.Max(5f, timeout - 4f));
            var tools = pilot.Toolbox;
            var w = tools.BeginWait(maxWait);
            await Until(() => tools.Settled(w), wallCap);
            return Respond(tools.Conclude(w, issued));
        }

        /// <summary>Completes when <paramref name="done"/> is true (polled every editor update) or after a wall-clock cap.</summary>
        public static Task<bool> Until(Func<bool> done, double wallSeconds)
        {
            var tcs = new TaskCompletionSource<bool>();
            var start = EditorApplication.timeSinceStartup;
            EditorApplication.CallbackFunction tick = null;
            tick = () =>
            {
                bool finished;
                try
                {
                    finished = done();
                }
                catch (Exception)
                {
                    finished = true;
                }
                if (finished || !EditorApplication.isPlaying || EditorApplication.timeSinceStartup - start > wallSeconds)
                {
                    EditorApplication.update -= tick;
                    tcs.TrySetResult(finished);
                }
            };
            EditorApplication.update += tick;
            EditorApplication.QueuePlayerLoopUpdate();
            return tcs.Task;
        }
    }

    [McpForUnityTool("robot_observe", Description =
        "Break My Bot AI pilot: observe the Go2 robot dog (Unity Play mode, Playground scene). Returns its pose, base-frame " +
        "velocity, the goal (distance and bearing) and time left if a challenge round runs, the world model's predicted " +
        "P(fall within 0.2/0.5/1 s), its surprise score and alarm state, the safety guard state, the current order, visible " +
        "obstacles and painted ice patches within 8 m, and the player's actions since the last observation (hidden traps " +
        "are reported as set, not where). " + PilotMcp.Frames)]
    public static class RobotObserve
    {
        public class Parameters
        {
        }

        public static object HandleCommand(JObject @params)
        {
            var pilot = PilotMcp.Find(out var error);
            if (pilot == null)
                return error;
            return new SuccessResponse("observation", pilot.Toolbox.Observe());
        }
    }

    [McpForUnityTool("robot_walk_to", Description =
        "Break My Bot AI pilot: walk the Go2 to a point (x, y). The learned locomotion policy does the walking; this sets " +
        "the goal of a heading controller (turns on the spot first if the point is more than 60° off, slows down within 1 m, " +
        "arrives within 0.3 m). Before it starts, the world model imagines the first second and VETOES the order if it " +
        "predicts a likely fall (P(fall within 1 s) > 0.5); while it runs, a lookahead keeps imagining 1 s ahead and HALTS " +
        "the robot before predicted falls. After a surprise alarm (ice, push, trap) a guard caps the speed at 0.3 m/s for 2 s. " +
        "The call then WAITS until something worth a new decision happens (arrived, halted, alarm, fell, the player did " +
        "something, goal reached, time up) or max_wait_s passes, and returns the outcome with a fresh observation. " +
        PilotMcp.Frames)]
    public static class RobotWalkTo
    {
        public class Parameters
        {
            [ToolParameter("Target x (m, Isaac world frame)")] public float x { get; set; }
            [ToolParameter("Target y (m, Isaac world frame)")] public float y { get; set; }
            [ToolParameter("Walking speed in m/s (0.1–1.0)", Required = false, DefaultValue = "0.6")] public float speed { get; set; }
            [ToolParameter("One short sentence: why this order (shown to the player on the HUD)", Required = false)] public string note { get; set; }
            [ToolParameter("Simulated seconds to wait for an outcome; 0 = return at once (default 8, max 60)", Required = false, DefaultValue = "8")] public float max_wait_s { get; set; }
            [ToolParameter("MCP timeout in seconds; raise it above 30 when max_wait_s is above 12", Required = false)] public int timeout_seconds { get; set; }
        }

        public static async Task<object> HandleCommand(JObject @params)
        {
            var pilot = PilotMcp.Find(out var error);
            if (pilot == null)
                return error;
            var x = PilotMcp.F(@params, "x");
            var y = PilotMcp.F(@params, "y");
            if (x == null || y == null)
                return new ErrorResponse("x and y are required");
            var issued = pilot.Toolbox.WalkTo(x.Value, y.Value, PilotMcp.F(@params, "speed") ?? PilotToolbox.DefaultSpeed, PilotMcp.S(@params, "note"));
            return await PilotMcp.WaitAndRespond(pilot, issued, @params);
        }
    }

    [McpForUnityTool("robot_move", Description =
        "Break My Bot AI pilot: hold a velocity command for some seconds, then stand. vx forward, vy left (m/s), wz turn " +
        "left (rad/s), each clamped to ±1 (the training range). Same world-model veto, lookahead and guard as robot_walk_to, " +
        "and the same waiting: returns when the move is done or something happens first. " + PilotMcp.Frames)]
    public static class RobotMove
    {
        public class Parameters
        {
            [ToolParameter("Forward velocity (m/s)")] public float vx { get; set; }
            [ToolParameter("Leftward velocity (m/s)")] public float vy { get; set; }
            [ToolParameter("Turn rate, positive = left (rad/s)")] public float wz { get; set; }
            [ToolParameter("Duration (s, up to 10)")] public float seconds { get; set; }
            [ToolParameter("One short sentence: why this order (shown on the HUD)", Required = false)] public string note { get; set; }
            [ToolParameter("Simulated seconds to wait for an outcome; 0 = return at once (default 8)", Required = false, DefaultValue = "8")] public float max_wait_s { get; set; }
            [ToolParameter("MCP timeout in seconds; raise it above 30 when max_wait_s is above 12", Required = false)] public int timeout_seconds { get; set; }
        }

        public static async Task<object> HandleCommand(JObject @params)
        {
            var pilot = PilotMcp.Find(out var error);
            if (pilot == null)
                return error;
            var vx = PilotMcp.F(@params, "vx");
            var vy = PilotMcp.F(@params, "vy");
            var wz = PilotMcp.F(@params, "wz");
            var seconds = PilotMcp.F(@params, "seconds");
            if (vx == null || vy == null || wz == null || seconds == null)
                return new ErrorResponse("vx, vy, wz and seconds are required");
            var issued = pilot.Toolbox.Move(vx.Value, vy.Value, wz.Value, seconds.Value, PilotMcp.S(@params, "note"));
            return await PilotMcp.WaitAndRespond(pilot, issued, @params);
        }
    }

    [McpForUnityTool("robot_stop", Description = "Break My Bot AI pilot: stop and stand still (cancels the running order). Returns at once with an observation.")]
    public static class RobotStop
    {
        public class Parameters
        {
            [ToolParameter("One short sentence: why (shown on the HUD)", Required = false)] public string note { get; set; }
        }

        public static object HandleCommand(JObject @params)
        {
            var pilot = PilotMcp.Find(out var error);
            if (pilot == null)
                return error;
            var r = pilot.Toolbox.Stop(PilotMcp.S(@params, "note"));
            if (r.accepted)
                r.observation = pilot.Toolbox.Observe();
            return PilotMcp.Respond(r);
        }
    }

    [McpForUnityTool("robot_wait", Description =
        "Break My Bot AI pilot: keep the current order (or standing) and wait for the next event worth a decision " +
        "(arrival, halt, alarm, fall, player action, goal reached, time up), at most max_wait_s simulated seconds.")]
    public static class RobotWait
    {
        public class Parameters
        {
            [ToolParameter("Simulated seconds to wait (default 5, max 60)", Required = false, DefaultValue = "5")] public float max_wait_s { get; set; }
            [ToolParameter("MCP timeout in seconds; raise it above 30 when max_wait_s is above 12", Required = false)] public int timeout_seconds { get; set; }
        }

        public static async Task<object> HandleCommand(JObject @params)
        {
            var pilot = PilotMcp.Find(out var error);
            if (pilot == null)
                return error;
            var tools = pilot.Toolbox;
            var maxWait = PilotMcp.F(@params, "max_wait_s") ?? 5f;
            var timeout = PilotMcp.F(@params, "timeout_seconds") ?? 30f;
            var w = tools.BeginWait(maxWait);
            // Standing still is not an order: wait for attention or the time only.
            await PilotMcp.Until(() => pilot.Attention != w.Attention || pilot.Controller.TotalTime - w.Start >= w.MaxWait,
                                 Mathf.Min(maxWait * 2f + 2f, Mathf.Max(5f, timeout - 4f)));
            return PilotMcp.Respond(tools.Conclude(w, new PilotActionResult { accepted = true }));
        }
    }

    [McpForUnityTool("robot_imagine", Description =
        "Break My Bot AI pilot: imagine an order in the world model WITHOUT executing it (the learned dynamics model rolls " +
        "the robot forward with the locomotion policy in the loop). Give x, y (and speed) to imagine walking there, or vx, " +
        "vy, wz to imagine a velocity command. Returns the imagined end pose, path (every 0.25 s) and the highest predicted " +
        "P(fall within 0.2/0.5/1 s); the veto rejects orders whose 1 s risk exceeds 0.5. The world model sees geometry " +
        "(height scan) but not friction: it cannot foresee ice. Accurate to a few cm over 2 s on flat ground. " + PilotMcp.Frames)]
    public static class RobotImagine
    {
        public class Parameters
        {
            [ToolParameter("Target x for an imagined walk (m)", Required = false)] public float x { get; set; }
            [ToolParameter("Target y for an imagined walk (m)", Required = false)] public float y { get; set; }
            [ToolParameter("Walking speed (m/s)", Required = false, DefaultValue = "0.6")] public float speed { get; set; }
            [ToolParameter("Forward velocity for an imagined move (m/s)", Required = false)] public float vx { get; set; }
            [ToolParameter("Leftward velocity (m/s)", Required = false)] public float vy { get; set; }
            [ToolParameter("Turn rate, positive = left (rad/s)", Required = false)] public float wz { get; set; }
            [ToolParameter("How far to imagine (s, up to 3; default 2)", Required = false, DefaultValue = "2")] public float seconds { get; set; }
        }

        public static object HandleCommand(JObject @params)
        {
            var pilot = PilotMcp.Find(out var error);
            if (pilot == null)
                return error;
            var seconds = PilotMcp.F(@params, "seconds") ?? 2f;
            var x = PilotMcp.F(@params, "x");
            var y = PilotMcp.F(@params, "y");
            PilotActionResult r;
            if (x != null && y != null)
                r = pilot.Toolbox.ImagineWalkTo(x.Value, y.Value, PilotMcp.F(@params, "speed") ?? PilotToolbox.DefaultSpeed, seconds);
            else if (PilotMcp.F(@params, "vx") != null || PilotMcp.F(@params, "vy") != null || PilotMcp.F(@params, "wz") != null)
                r = pilot.Toolbox.ImagineMove(PilotMcp.F(@params, "vx") ?? 0f, PilotMcp.F(@params, "vy") ?? 0f, PilotMcp.F(@params, "wz") ?? 0f, seconds);
            else
                return new ErrorResponse("give x and y (a walk) or vx / vy / wz (a move)");
            return PilotMcp.Respond(r);
        }
    }

    [McpForUnityTool("challenge_start", Description =
        "Break My Bot: start a 'Claude vs player' round. The AI pilot must bring the Go2 within 0.5 m of the goal before the " +
        "time runs out while the player disturbs it (pushes, balls, ice, obstacles, traps). Give goal_x and goal_y, or neither " +
        "for a random goal 6–12 m away (seed for repeatable goals). Scored like RoboDojo: success, and progress 0–100. " +
        "turn_based = true pauses the simulation whenever no order runs (while you think); default is real time. " +
        "Returns the round status; then observe and give orders. " + PilotMcp.Frames)]
    public static class ChallengeStart
    {
        public class Parameters
        {
            [ToolParameter("Goal x (m)", Required = false)] public float goal_x { get; set; }
            [ToolParameter("Goal y (m)", Required = false)] public float goal_y { get; set; }
            [ToolParameter("Seed for a random goal", Required = false)] public int seed { get; set; }
            [ToolParameter("Time limit (s, 10–600, default 90)", Required = false, DefaultValue = "90")] public float time_limit_s { get; set; }
            [ToolParameter("Pause the simulation while no order runs (default false)", Required = false, DefaultValue = "false")] public bool turn_based { get; set; }
        }

        public static object HandleCommand(JObject @params)
        {
            var pilot = PilotMcp.Find(out var error);
            if (pilot == null)
                return error;
            var status = pilot.Toolbox.StartChallenge(PilotMcp.F(@params, "goal_x"), PilotMcp.F(@params, "goal_y"), PilotMcp.I(@params, "seed"),
                                                      PilotMcp.F(@params, "time_limit_s") ?? PilotChallenge.DefaultTimeLimit,
                                                      PilotMcp.B(@params, "turn_based") ?? false, out var why);
            if (status == null)
                return new ErrorResponse(why);
            return new SuccessResponse($"round {status.round}: goal at ({status.goal_x}, {status.goal_y}), {status.distance} m away, {status.time_limit_s} s",
                                       new { status, observation = pilot.Toolbox.Observe() });
        }
    }

    [McpForUnityTool("challenge_status", Description = "Break My Bot: the current 'Claude vs player' round (goal, time left, distance, score) and the last result.")]
    public static class ChallengeStatus
    {
        public class Parameters
        {
        }

        public static object HandleCommand(JObject @params)
        {
            var pilot = PilotMcp.Find(out var error);
            if (pilot == null)
                return error;
            var status = pilot.Toolbox.ChallengeStatus();
            return status == null ? new ErrorResponse("no challenge in this scene") : new SuccessResponse(status.running ? "round running" : "no round running", status);
        }
    }

    [McpForUnityTool("pilot_config", Description =
        "Break My Bot AI pilot settings (for experiments): veto = world-model veto + lookahead on/off, guard = cautious gait " +
        "after surprise alarms on/off, veto_fall_prob = the veto threshold on P(fall within 1 s); release = hand the robot " +
        "back to the player. Omitted settings are unchanged; returns the settings.")]
    public static class PilotConfig
    {
        public class Parameters
        {
            [ToolParameter("World-model veto and lookahead", Required = false)] public bool veto { get; set; }
            [ToolParameter("Cautious gait after surprise alarms", Required = false)] public bool guard { get; set; }
            [ToolParameter("Veto threshold on P(fall within 1 s), 0–1 (default 0.5)", Required = false)] public float veto_fall_prob { get; set; }
            [ToolParameter("Hand the robot back to the player", Required = false)] public bool release { get; set; }
        }

        public static object HandleCommand(JObject @params)
        {
            var pilot = PilotMcp.Find(out var error);
            if (pilot == null)
                return error;
            if (PilotMcp.B(@params, "veto") is bool veto)
                pilot.Veto = veto;
            if (PilotMcp.B(@params, "guard") is bool guard)
                pilot.Guard = guard;
            if (PilotMcp.F(@params, "veto_fall_prob") is float p && float.IsFinite(p))
                pilot.VetoFallProb = Mathf.Clamp01(p);
            if (PilotMcp.B(@params, "release") == true)
                pilot.Release();
            return new SuccessResponse("pilot settings", new { veto = pilot.Veto, guard = pilot.Guard, veto_fall_prob = pilot.VetoFallProb, engaged = pilot.Engaged });
        }
    }
}
