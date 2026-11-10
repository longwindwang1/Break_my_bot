using System;
using System.IO;
using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Claude vs the player: the AI pilot has to bring the Go2 to a goal within a time limit while the player disturbs
    /// it with the tools. Scored like RoboDojo: success (reached within <see cref="GoalRadius"/>) and a progress score
    /// 100 · max(0, 1 − distance left / start distance); falls, disturbances, vetoes, lookahead halts and surprise
    /// alarms are counted. Each finished round appends one JSON line to <see cref="RepoPaths.PilotResults"/>.
    /// Turn-based mode pauses the simulation (Time.timeScale 0) whenever the pilot has no running order, i.e. while
    /// Claude is deciding, like the paused simulator in Anthropic's robotics study; real time is the default.
    /// </summary>
    public sealed class PilotChallenge : MonoBehaviour
    {
        public const float GoalRadius = 0.5f;
        public const float MinGoalDistance = 6f;
        public const float MaxGoalDistance = 12f;
        public const float DefaultTimeLimit = 90f;

        [SerializeField] ClaudePilot pilot;
        [Tooltip("Append each finished round to experiments/claude_pilot/results.jsonl (editor)")]
        [SerializeField] bool writeResults = true;

        bool subscribed;
        DisturbanceManager subscribedManager;
        int fallsAtStart, ordersAtStart, vetoesAtStart, haltsAtStart, alarmsAtStart;

        public ClaudePilot Pilot => pilot;
        public bool Running { get; private set; }
        public Vector2 Goal { get; private set; }
        public float TimeLimit { get; private set; } = DefaultTimeLimit;
        public bool TurnBased { get; private set; }
        public double StartTime { get; private set; }
        public float StartDistance { get; private set; }
        public int Round { get; private set; }
        public int Disturbances { get; private set; }
        /// <summary>Who is piloting, written to the results (e.g. "claude-code (mcp)").</summary>
        public string PilotName { get; set; } = "claude-code (mcp)";
        public PilotChallengeResult LastResult { get; private set; }
        public bool WriteResults
        {
            get => writeResults;
            set => writeResults = value;
        }

        public event Action<PilotChallenge> Started;
        public event Action<PilotChallenge, PilotChallengeResult> Finished;

        Go2Controller Controller => pilot != null ? pilot.Controller : null;
        double Now => Controller != null && Controller.Initialized ? Controller.TotalTime : 0d;
        public float Elapsed => Running ? (float)(Now - StartTime) : LastResult?.time_s ?? 0f;
        public float TimeLeft => Running ? Mathf.Max(0f, TimeLimit - Elapsed) : 0f;

        public float DistanceLeft => Controller != null && Controller.Initialized ? PilotPose.Of(Controller).DistanceTo(Goal) : float.NaN;

        /// <summary>Progress score 0–100 (100 once reached).</summary>
        public float Score(float distanceLeft) =>
            StartDistance <= 0f ? 0f : 100f * Mathf.Clamp01(1f - Mathf.Max(0f, distanceLeft - GoalRadius) / Mathf.Max(1e-3f, StartDistance - GoalRadius));

        public void Configure(ClaudePilot owner)
        {
            Unsubscribe();
            pilot = owner;
            Subscribe();
        }

        void OnEnable() => Subscribe();

        void OnDisable()
        {
            if (Running)
                Finish("cancelled", "challenge object disabled");
            Unsubscribe();
            if (Application.isPlaying && TurnBased)
                Time.timeScale = 1f;
        }

        void Subscribe()
        {
            if (subscribed || Controller == null)
                return;
            pilot.Challenge = this;
            Controller.PolicyStepped += OnPolicyStepped;
            subscribedManager = pilot.Manager;
            if (subscribedManager != null)
                subscribedManager.Log.Added += OnLogAdded;
            subscribed = true;
        }

        void Unsubscribe()
        {
            if (!subscribed)
                return;
            if (Controller != null)
                Controller.PolicyStepped -= OnPolicyStepped;
            if (subscribedManager != null)
                subscribedManager.Log.Added -= OnLogAdded;
            subscribedManager = null;
            subscribed = false;
        }

        /// <summary>
        /// Starts a round: the goal given (Isaac world x, y), or a random one 6–12 m from the robot (seeded), on free
        /// ground inside the level. A running round is cancelled first.
        /// </summary>
        public PilotChallengeStatus Begin(Vector2? goal = null, int? seed = null, float timeLimit = DefaultTimeLimit, bool turnBased = false)
        {
            if (Controller == null || !Controller.Initialized)
                throw new InvalidOperationException("the robot is not running");
            if (Running)
                Finish("cancelled", "a new round started");
            Subscribe();
            var pose = PilotPose.Of(Controller);
            Goal = goal ?? RandomGoal(pose.Position, seed ?? Environment.TickCount);
            TimeLimit = Mathf.Clamp(float.IsFinite(timeLimit) ? timeLimit : DefaultTimeLimit, 10f, 600f);
            TurnBased = turnBased;
            StartTime = Now;
            StartDistance = pose.DistanceTo(Goal);
            Disturbances = 0;
            fallsAtStart = Controller.Falls;
            ordersAtStart = pilot.OrdersIssued;
            vetoesAtStart = pilot.Vetoes;
            haltsAtStart = pilot.Halts;
            alarmsAtStart = pilot.Alarms;
            Round++;
            Running = true;
            pilot.Goal = Goal;
            pilot.Engage();
            pilot.Manager?.Record("goal", $"round {Round}: goal at ({Goal.x:F1}, {Goal.y:F1}), {StartDistance:F1} m away, {TimeLimit:0} s{(turnBased ? ", turn-based" : "")}", false);
            Started?.Invoke(this);
            return Status();
        }

        /// <summary>Ends the current round (e.g. the player gave up on it).</summary>
        public void Cancel(string why = "cancelled")
        {
            if (Running)
                Finish("cancelled", why);
        }

        Vector2 RandomGoal(Vector2 from, int seed)
        {
            var rng = new System.Random(seed);
            var level = pilot.Manager != null ? pilot.Manager.Level : null;
            var candidate = from + new Vector2(MinGoalDistance, 0f);
            for (var attempt = 0; attempt < 50; attempt++)
            {
                var angle = (float)(rng.NextDouble() * 2.0 * Math.PI);
                var distance = MinGoalDistance + (float)rng.NextDouble() * (MaxGoalDistance - MinGoalDistance);
                candidate = from + distance * new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
                if (level == null)
                    break;
                var world = level.transform.position + IsaacFrame.PolarFromIsaac(new Vector3(candidate.x, candidate.y, 0f));
                var local = world - level.transform.position;
                var inside = Mathf.Abs(local.x) < level.Size.x / 2f - 2f && Mathf.Abs(local.z) < level.Size.y / 2f - 2f;
                if (inside && level.FeatureAt(world) == null)
                    break;
            }
            return candidate;
        }

        void OnPolicyStepped(Go2Controller c)
        {
            if (!Running)
                return;
            var d = PilotPose.Of(c).DistanceTo(Goal);
            if (d <= GoalRadius)
                Finish("reached", $"reached the goal in {Elapsed:0.0} s");
            else if (Elapsed >= TimeLimit)
                Finish("timeout", $"time is up, {d:0.0} m from the goal");
        }

        void Update()
        {
            if (Application.isPlaying)
                UpdateTimeScale();
        }

        /// <summary>Turn-based rounds: the world only moves while an order runs (Time.timeScale 1, else 0).</summary>
        public void UpdateTimeScale()
        {
            if (!Running || !TurnBased)
                return;
            Time.timeScale = pilot.Engaged && pilot.Order != null && pilot.Order.Running ? 1f : 0f;
        }

        static bool IsDisturbance(DisturbanceEvent e) =>
            e.Type is "push" or "payload" or "drag" or "friction" or "obstacle" or "deceptive"
            || (e.Type == "ball" && e.Detail != null && e.Detail.StartsWith("thrown"));

        void OnLogAdded(DisturbanceEvent e)
        {
            if (Running && IsDisturbance(e))
                Disturbances++;
        }

        void Finish(string outcome, string detail)
        {
            var c = Controller;
            var distance = DistanceLeft;
            var reached = outcome == "reached";
            var result = new PilotChallengeResult
            {
                round = Round,
                outcome = outcome,
                success = reached,
                score = R(reached ? 100f : Score(distance), 1),
                time_s = R((float)(Now - StartTime), 1),
                time_limit_s = TimeLimit,
                start_distance = R(StartDistance),
                final_distance = R(distance),
                goal_x = R(Goal.x),
                goal_y = R(Goal.y),
                falls = c != null ? c.Falls - fallsAtStart : 0,
                disturbances = Disturbances,
                orders = pilot.OrdersIssued - ordersAtStart,
                vetoes = pilot.Vetoes - vetoesAtStart,
                halts = pilot.Halts - haltsAtStart,
                alarms = pilot.Alarms - alarmsAtStart,
                turn_based = TurnBased,
                veto = pilot.Veto,
                guard = pilot.Guard,
                pilot = PilotName,
                policy = c != null ? c.Variant : "",
                wall_time = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
            };
            Running = false;
            LastResult = result;
            if (TurnBased)
                Time.timeScale = 1f;
            pilot.Manager?.Record("goal", $"round {Round} {outcome}: {detail}; score {result.score:0}, falls {result.falls}, disturbances {result.disturbances}", false);
            pilot.Raise(reached ? "goal_reached" : outcome == "timeout" ? "timeout" : "cancelled",
                        $"round {Round} {outcome}: {detail} (score {result.score:0})");
            if (writeResults)
                Append(result, ResultsPath ?? RepoPaths.PilotResults);
            Finished?.Invoke(this, result);
        }

        /// <summary>Where finished rounds go; null = <see cref="RepoPaths.PilotResults"/> (tests point it elsewhere).</summary>
        public string ResultsPath { get; set; }

        static void Append(PilotChallengeResult result, string path)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path));
                File.AppendAllText(path, JsonUtility.ToJson(result) + "\n");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[BMB] could not write the pilot result: {e.Message}");
            }
        }

        /// <summary>The round as the pilot sees it.</summary>
        public PilotChallengeStatus Status()
        {
            var d = DistanceLeft;
            var s = new PilotChallengeStatus
            {
                running = Running,
                round = Round,
                goal_x = R(Goal.x),
                goal_y = R(Goal.y),
                goal_radius = GoalRadius,
                time_limit_s = TimeLimit,
                elapsed_s = R(Elapsed, 1),
                time_left_s = R(TimeLeft, 1),
                distance = R(float.IsFinite(d) ? d : 0f),
                score = R(Running ? Score(d) : LastResult?.score ?? 0f, 1),
                turn_based = TurnBased,
                last_result = LastResult,
            };
            return s;
        }

        static float R(float v, int digits = 2)
        {
            if (!float.IsFinite(v))
                return 0f;
            var s = Mathf.Pow(10f, digits);
            return Mathf.Round(v * s) / s;
        }
    }

    [Serializable]
    public sealed class PilotChallengeResult
    {
        public int round;
        /// <summary>reached, timeout or cancelled.</summary>
        public string outcome;
        public bool success;
        /// <summary>100 · max(0, 1 − distance left / start distance), 100 when reached.</summary>
        public float score;
        public float time_s;
        public float time_limit_s;
        public float start_distance;
        public float final_distance;
        public float goal_x;
        public float goal_y;
        public int falls;
        public int disturbances;
        public int orders;
        public int vetoes;
        public int halts;
        public int alarms;
        public bool turn_based;
        public bool veto;
        public bool guard;
        public string pilot;
        public string policy;
        public string wall_time;
    }

    [Serializable]
    public sealed class PilotChallengeStatus
    {
        public bool running;
        public int round;
        public float goal_x;
        public float goal_y;
        public float goal_radius;
        public float time_limit_s;
        public float elapsed_s;
        public float time_left_s;
        public float distance;
        public float score;
        public bool turn_based;
        public PilotChallengeResult last_result;
    }
}
