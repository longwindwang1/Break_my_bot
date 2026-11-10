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
        [SerializeField] Go2Controller controller;
        [SerializeField] DisturbanceManager manager;

        bool subscribed;
        int nextOrderId = 1;
        long attention;
        bool lastFallHandled;

        /// <summary>The pilot in the scene (play mode) or the last one configured (headless runs).</summary>
        public static ClaudePilot Active { get; private set; }

        public Go2Controller Controller => controller;
        public DisturbanceManager Manager => manager;
        public PilotToolbox Toolbox { get; private set; }
        public bool Engaged { get; private set; }
        /// <summary>The current order, or the last one (check <see cref="PilotOrder.Running"/>); null before the first.</summary>
        public PilotOrder Order { get; private set; }
        /// <summary>Goal the pilot is trying to reach (Isaac world x, y), set by the challenge; null = none.</summary>
        public Vector2? Goal { get; set; }

        /// <summary>Counts attention events; a waiting tool returns when it changes.</summary>
        public long Attention => attention;
        public string AttentionCode { get; private set; } = "";
        public string AttentionDetail { get; private set; } = "";
        public event Action<ClaudePilot> AttentionRaised;
        /// <summary>Raised when an order is issued or finishes (HUD).</summary>
        public event Action<ClaudePilot, PilotOrder> OrderChanged;

        /// <summary>Wiring for code-built setups (scene builder, headless runs, tests).</summary>
        public void Configure(Go2Controller robot, DisturbanceManager disturbances)
        {
            Unsubscribe();
            controller = robot;
            manager = disturbances;
            Toolbox = new PilotToolbox(this);
            Subscribe();
            Active = this;
        }

        void OnEnable()
        {
            Toolbox ??= new PilotToolbox(this);
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
            controller.CommandOverride = CurrentCommand();
            OrderChanged?.Invoke(this, order);
            return order;
        }

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
            c.CommandOverride = CurrentCommand();
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

        /// <summary>Event types the pilot itself causes or that are reported elsewhere (fall, reset).</summary>
        static bool Ignored(string type) => type is "pilot" or "goal" or "fall" or "reset" or "mode";

        void OnLogAdded(DisturbanceEvent e)
        {
            if (!Engaged || Ignored(e.Type))
                return;
            Raise(e.Type == "surprise" ? "surprise" : "disturbed", $"{e.Type}: {PilotToolbox.PilotDetail(e)}");
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
