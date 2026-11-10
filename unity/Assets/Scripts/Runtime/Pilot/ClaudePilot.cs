using System;
using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Executes an AI pilot's high-level orders (walk to a point, hold a velocity for a while, stand) on the Go2 by
    /// writing <see cref="Go2Controller.CommandOverride"/> after every policy step; the locomotion policy does the rest.
    /// While engaged the player's keyboard / stick command is ignored; <see cref="Release"/> hands it back.
    /// It also raises "attention" when something happens that is worth a new decision (order finished, fall, reset,
    /// the player used a tool, a surprise alarm), which the pilot's tools wait for.
    /// Orders arrive through <see cref="PilotToolbox"/> (MCP tools for Claude Code, later the in-game API agent).
    /// </summary>
    public sealed class ClaudePilot : MonoBehaviour
    {
        /// <summary>Orders are imagined this far ahead for the veto (s).</summary>
        public const float VetoSeconds = 1f;
        /// <summary>After a surprise alarm the cautious gait holds this long (s); later alarms extend it.</summary>
        public const float GuardHoldSeconds = 2f;
        /// <summary>Time per frame the lookahead may spend dreaming (ms); a 1 s lookahead costs about 50 ms in all.</summary>
        public const double LookaheadBudgetMs = 3;
        /// <summary>
        /// Fall horizon the veto reads: world_model.fall_horizons_steps [10, 25, 50], index 2 = within 1 s. The 0.5 s
        /// horizon reacts too late for a 1 s look ahead (a pit 0.45 m ahead: 0.31 at 0.5 s, 0.66 at 1 s; open ground 0.03).
        /// </summary>
        public const int VetoHorizon = 2;

        /// <summary>The veto's risk: the highest imagined P(fall within 1 s).</summary>
        public static float Risk(float[] maxFallProbs) =>
            maxFallProbs != null && maxFallProbs.Length > VetoHorizon ? maxFallProbs[VetoHorizon] : 0f;

        [SerializeField] Go2Controller controller;
        [SerializeField] DisturbanceManager manager;
        [Tooltip("Reject orders whose imagined first second reaches a high predicted fall risk")]
        [SerializeField] bool veto = true;
        [Tooltip("Veto when the imagined P(fall within 1 s) exceeds this")]
        [Range(0f, 1f)] [SerializeField] float vetoFallProb = 0.5f;
        [Tooltip("On a world-model surprise alarm, cap the command to the cautious gait (policies.v2.cautious_gait)")]
        [SerializeField] bool guard = true;

        bool subscribed;
        int nextOrderId = 1;
        long attention;
        bool lastFallHandled;
        double guardUntil = double.NegativeInfinity;
        Vector2 cautiousGait = new Vector2(0.3f, 0.5f);

        /// <summary>The pilot in the scene (play mode) or the last one configured (headless runs).</summary>
        public static ClaudePilot Active { get; private set; }

        public Go2Controller Controller => controller;
        public DisturbanceManager Manager => manager;
        public PilotToolbox Toolbox { get; private set; }
        public PilotImagination Imagination { get; private set; }

        public bool Veto
        {
            get => veto;
            set => veto = value;
        }

        public float VetoFallProb
        {
            get => vetoFallProb;
            set => vetoFallProb = value;
        }

        public bool Guard
        {
            get => guard;
            set => guard = value;
        }

        /// <summary>The cautious-gait cap is on (guard enabled and a surprise alarm within the hold time).</summary>
        public bool GuardActive => guard && Engaged && Now < guardUntil;
        public float GuardSecondsLeft => GuardActive ? (float)(guardUntil - Now) : 0f;

        /// <summary>The last imagination (imagine tool or veto check), for the view and the HUD.</summary>
        public PilotImagined LastImagined { get; private set; }
        public event Action<ClaudePilot, PilotImagined> Imagined;

        /// <summary>The lookahead's last finished dream: imagined base path (Unity) and max P(fall within 1 s).</summary>
        public Vector3[] LookaheadPath => lookDreamer?.Path ?? Array.Empty<Vector3>();
        public float LookaheadRisk { get; private set; }
        public int LookaheadDreams { get; private set; }

        WorldModelDreamer lookDreamer;
        int lookOrderId;
        public bool Engaged { get; private set; }
        /// <summary>The current order, or the last one (check <see cref="PilotOrder.Running"/>); null before the first.</summary>
        public PilotOrder Order { get; private set; }
        /// <summary>Goal the pilot is trying to reach (Isaac world x, y), set by the challenge; null = none.</summary>
        public Vector2? Goal { get; set; }
        /// <summary>The challenge using this pilot (set by <see cref="PilotChallenge"/>), for the tools and the HUD.</summary>
        public PilotChallenge Challenge { get; internal set; }

        /// <summary>Counts attention events; a waiting tool returns when it changes.</summary>
        public long Attention => attention;
        public string AttentionCode { get; private set; } = "";
        public string AttentionDetail { get; private set; } = "";
        public event Action<ClaudePilot> AttentionRaised;
        /// <summary>Raised when an order is issued or finishes (HUD).</summary>
        public event Action<ClaudePilot, PilotOrder> OrderChanged;

        /// <summary>Counters for scoring (orders issued, vetoed, halted by the lookahead, surprise alarms while engaged).</summary>
        public int OrdersIssued { get; private set; }
        public int Vetoes { get; private set; }
        public int Halts { get; private set; }
        public int Alarms { get; private set; }
        public double LastVetoTime { get; private set; } = double.NegativeInfinity;

        internal void CountVeto()
        {
            Vetoes++;
            LastVetoTime = Now;
        }

        /// <summary>Wiring for code-built setups (scene builder, headless runs, tests).</summary>
        public void Configure(Go2Controller robot, DisturbanceManager disturbances)
        {
            Unsubscribe();
            controller = robot;
            manager = disturbances;
            Toolbox = new PilotToolbox(this);
            Imagination = new PilotImagination(this);
            Subscribe();
            Active = this;
        }

        void OnEnable()
        {
            Toolbox ??= new PilotToolbox(this);
            Imagination ??= new PilotImagination(this);
            Subscribe();
            Active = this;
        }

        void OnDisable()
        {
            if (Engaged)
                Release();
            Unsubscribe();
            if (Active == this)
                Active = null;
        }

        void Subscribe()
        {
            if (subscribed || controller == null)
                return;
            controller.PolicyStepped += OnPolicyStepped;
            controller.Fell += OnFell;
            controller.WasReset += OnReset;
            controller.SurpriseAlarm += OnSurpriseAlarm;
            if (manager != null)
                manager.Log.Added += OnLogAdded;
            Toolbox?.Subscribe();
            subscribed = true;
        }

        void Unsubscribe()
        {
            if (!subscribed)
                return;
            if (controller != null)
            {
                controller.PolicyStepped -= OnPolicyStepped;
                controller.Fell -= OnFell;
                controller.WasReset -= OnReset;
                controller.SurpriseAlarm -= OnSurpriseAlarm;
            }
            if (manager != null)
                manager.Log.Added -= OnLogAdded;
            Toolbox?.Unsubscribe();
            subscribed = false;
        }

        double Now => controller != null && controller.Initialized ? controller.TotalTime : 0d;

        /// <summary>Takes control of the robot (it stands until the first order).</summary>
        public void Engage()
        {
            if (Engaged)
                return;
            Engaged = true;
            // The pilot's tools are called from a terminal: keep simulating while the editor is not focused.
            if (Application.isPlaying)
                Application.runInBackground = true;
            var gait = controller.Interface.Policies.TryGetValue(controller.Variant, out var p) ? p.CautiousGait : null;
            if (gait != null)
                cautiousGait = new Vector2(gait.MaxLinVel, gait.MaxAngVel);
            controller.CommandOverride = Vector3.zero;
            Record("pilot", "AI pilot has control");
        }

        /// <summary>Hands control back to the player's command source.</summary>
        public void Release()
        {
            if (!Engaged)
                return;
            Finish(PilotOrderState.Released, "control handed back");
            Engaged = false;
            controller.CommandOverride = null;
            Record("pilot", "control back to the player");
            Raise("released", "control handed back to the player");
        }

        /// <summary>Starts an order (engaging if needed); a running order is replaced. Returns the order.</summary>
        public PilotOrder Issue(PilotOrder order)
        {
            Engage();
            Finish(PilotOrderState.Replaced, "replaced by a new order");
            order.Id = nextOrderId++;
            order.IssuedAt = Now;
            order.State = order.Kind == PilotOrderKind.Stand ? PilotOrderState.Standing : PilotOrderState.Running;
            if (!order.Running)
                order.FinishedAt = order.IssuedAt;
            Order = order;
            var note = string.IsNullOrEmpty(order.Note) ? "" : $" — {order.Note}";
            Record("pilot", $"#{order.Id} {order.Describe()}{note}");
            controller.CommandOverride = Guarded(CurrentCommand());
            OrdersIssued++;
            OrderChanged?.Invoke(this, order);
            return order;
        }

        /// <summary>Imagines an order without executing it (null without a world model); remembered for the view.</summary>
        public PilotImagined Imagine(PilotOrder order, float seconds)
        {
            var result = Imagination?.Run(order, seconds);
            if (result != null)
            {
                LastImagined = result;
                Imagined?.Invoke(this, result);
            }
            return result;
        }

        /// <summary>The cautious gait: |(vx, vy)| ≤ max_lin_vel, |wz| ≤ max_ang_vel.</summary>
        public Vector3 CautiousCap(Vector3 command) => PilotDrive.Cautious(command, cautiousGait.x, cautiousGait.y);

        Vector3 Guarded(Vector3 command) => GuardActive ? CautiousCap(command) : command;

        /// <summary>The command the pilot wants now (before any safety cap), from the current order.</summary>
        public Vector3 CurrentCommand()
        {
            if (Order == null || !Order.Running)
                return Vector3.zero;
            var ranges = controller.Interface.Commands;
            return Order.Kind switch
            {
                PilotOrderKind.WalkTo => PilotDrive.WalkTo(PilotPose.Of(controller), Order.Target, Order.Speed, ranges),
                PilotOrderKind.Move => Order.Velocity,
                _ => Vector3.zero,
            };
        }

        void OnPolicyStepped(Go2Controller c)
        {
            if (!Engaged)
                return;
            lastFallHandled = false;
            if (Order != null && Order.Running)
            {
                if (Order.Kind == PilotOrderKind.WalkTo)
                {
                    var d = PilotPose.Of(c).DistanceTo(Order.Target);
                    if (d <= PilotDrive.ArriveRadius)
                        Complete(PilotOrderState.Arrived, "arrived", $"reached ({Order.Target.x:F1}, {Order.Target.y:F1}), {d:F2} m off, after {Order.Elapsed(Now):F1} s");
                }
                else if (Order.Kind == PilotOrderKind.Move && Order.Elapsed(Now) >= Order.Seconds)
                    Complete(PilotOrderState.Done, "done", $"moved for {Order.Seconds:F1} s");
            }
            c.CommandOverride = Guarded(CurrentCommand());
        }

        void Update()
        {
            if (!Application.isPlaying)
                return;
            // P: the player takes the robot back, or hands it to the AI pilot (it stands until the next order).
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            if (keyboard != null && keyboard.pKey.wasPressedThisFrame)
            {
                if (Engaged)
                    Release();
                else
                    Engage();
            }
            TickLookahead(LookaheadBudgetMs);
        }

        /// <summary>
        /// While an order runs (and the veto is on), keeps imagining its next <see cref="VetoSeconds"/> from the latest
        /// state, a slice of work per call. When a dream ends with P(fall within 1 s) above the threshold, the robot
        /// stops and the pilot is told: this catches what appears after the order was issued (a pit the player digs
        /// ahead, a step it was not aimed at). Called every frame in play mode; headless runs call it themselves.
        /// </summary>
        public void TickLookahead(double budgetMs)
        {
            var m = controller != null && controller.Initialized ? controller.WorldModel : null;
            if (!veto || !Engaged || Order == null || !Order.Running || m == null || !m.HasPrediction)
            {
                lookDreamer?.Cancel();
                return;
            }
            lookDreamer ??= new WorldModelDreamer(controller);
            if (!lookDreamer.Busy)
            {
                lookDreamer.Horizon = Mathf.Max(1, Mathf.RoundToInt(VetoSeconds / controller.Interface.Timing.PolicyDt));
                lookOrderId = Order.Id;
                var driver = new DreamDriver(this, Order, GuardActive, Order.Elapsed(Now));
                if (!lookDreamer.Begin(driver.Command))
                    return;
            }
            if (!lookDreamer.Advance(budgetMs))
                return;
            LookaheadDreams++;
            LookaheadRisk = Risk(lookDreamer.MaxFallProbs);
            if (Order == null || !Order.Running || Order.Id != lookOrderId || LookaheadRisk <= vetoFallProb)
                return;
            var detail = $"halted by the world-model lookahead: going on with \"{Order.Describe()}\" predicts P(fall within 1 s) " +
                         $"up to {LookaheadRisk:0.00} (> {vetoFallProb:0.00}) within {VetoSeconds:0} s. The robot stopped at " +
                         $"({PilotPose.Of(controller).Position.x:0.0}, {PilotPose.Of(controller).Position.y:0.0}); find another way.";
            Finish(PilotOrderState.Halted, detail);
            controller.CommandOverride = Vector3.zero;
            Record("pilot", $"#{Order.Id} lookahead halt (P fall {LookaheadRisk:0.00})");
            Halts++;
            Raise("halted", detail);
        }

        double lastAlarmRaised = double.NegativeInfinity;

        /// <summary>
        /// The world model was surprised (ice, a push, a trap): with the guard on, hold the cautious gait for
        /// <see cref="GuardHoldSeconds"/> (each alarm extends it). The pilot is told at most once per hold time.
        /// </summary>
        void OnSurpriseAlarm(Go2Controller c, WorldModelMonitor m)
        {
            if (!Engaged)
                return;
            var d = m.Detector;
            var what = d.Spike ? $"spike |z| {d.SpikeStat:0} > {d.Params.SpikeZ:0} in base velocity" : $"CUSUM g {d.GPeak:0} > h {d.Params.H:0}";
            Alarms++;
            var fresh = !GuardActive;
            if (guard)
                guardUntil = Now + GuardHoldSeconds;
            if (guard && fresh)
                Record("pilot", $"guard: surprise alarm, cautious gait (≤ {cautiousGait.x:0.0} m/s) for {GuardHoldSeconds:0} s");
            c.CommandOverride = Guarded(CurrentCommand());
            if (Now - lastAlarmRaised < GuardHoldSeconds)
                return;
            lastAlarmRaised = Now;
            var cap = guard ? $"; the guard caps speed at {cautiousGait.x:0.0} m/s for {GuardHoldSeconds:0} s" : "";
            Raise("alarm", $"world-model surprise alarm ({what}): the robot is not moving as predicted (ice, a push or a trap?){cap}");
        }

        void Complete(PilotOrderState state, string code, string detail)
        {
            Finish(state, detail);
            Record("pilot", $"#{Order.Id} {detail}");
            Raise(code, detail);
        }

        void Finish(PilotOrderState state, string detail)
        {
            if (Order == null || !Order.Running)
                return;
            Order.State = state;
            Order.Detail = detail;
            Order.FinishedAt = Now;
            OrderChanged?.Invoke(this, Order);
        }

        void OnFell(Go2Controller c, string reason)
        {
            if (!Engaged)
                return;
            lastFallHandled = true;
            Finish(PilotOrderState.Fell, $"fell ({reason})");
            Raise("fell", $"the robot fell ({reason}) and was reset to its spawn point");
        }

        void OnReset(Go2Controller c)
        {
            if (!Engaged || lastFallHandled)
                return;
            Finish(PilotOrderState.Reset, "robot reset");
            Raise("reset", "the robot was reset to its spawn point");
        }

        /// <summary>Event types the pilot itself causes or that are reported elsewhere (fall, reset, surprise alarms).</summary>
        static bool Ignored(string type) => type is "pilot" or "goal" or "fall" or "reset" or "mode" or "surprise";

        void OnLogAdded(DisturbanceEvent e)
        {
            if (!Engaged || Ignored(e.Type))
                return;
            Raise("disturbed", $"{e.Type}: {PilotToolbox.PilotDetail(e)}");
        }

        /// <summary>Something worth a new decision happened.</summary>
        public void Raise(string code, string detail)
        {
            attention++;
            AttentionCode = code;
            AttentionDetail = detail;
            AttentionRaised?.Invoke(this);
        }

        void Record(string type, string detail)
        {
            if (manager != null)
                manager.Record(type, detail, false);
        }
    }
}
