using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>
    /// The player's (and scripts') way to disturb the robot. Every disturbance goes through an API method here, which
    /// clamps it to the active <see cref="DisturbanceLimits"/> mode and records it in <see cref="Log"/>. Interactive
    /// tools (keys 1–7) only translate mouse / keyboard input into these calls; headless evaluations call them directly.
    /// Keys: 1–7 select a tool, Esc deselects, H toggles training-range / hard mode.
    /// </summary>
    [DefaultExecutionOrder(-40)]
    public sealed class DisturbanceManager : MonoBehaviour
    {
        [SerializeField] Go2Controller controller;
        [SerializeField] DisturbanceLimits limits;
        [SerializeField] Camera viewCamera;
        [Tooltip("Allow values beyond the v2 training ranges (flagged in the HUD and the log)")]
        [SerializeField] bool hardMode;
        [Tooltip("Write the event log to unity/Logs/disturbances_<time>.csv when play mode ends")]
        [SerializeField] bool writeLogOnExit = true;

        readonly List<DisturbanceTool> tools = new List<DisturbanceTool>();
        int terrainMask, robotMask;
        bool subscribed;

        public DisturbanceLog Log { get; } = new DisturbanceLog();
        public Go2Controller Controller => controller;
        public DisturbanceLimits Limits => limits;
        public Camera ViewCamera => viewCamera;
        public IReadOnlyList<DisturbanceTool> Tools => tools;
        public DisturbanceTool ActiveTool { get; private set; }
        /// <summary>Extra HUD lines (payload, foot friction, …) contributed by subsystems.</summary>
        public List<Func<string>> StatusLines { get; } = new List<Func<string>>();

        public bool HardMode
        {
            get => hardMode;
            set
            {
                if (hardMode == value)
                    return;
                hardMode = value;
                Record("mode", value ? "hard mode (beyond training ranges allowed)" : "training ranges", false);
            }
        }

        /// <summary>Wiring for code-built setups (headless runs, tests).</summary>
        public void Configure(Go2Controller robotController, DisturbanceLimits disturbanceLimits, Camera camera = null)
        {
            controller = robotController;
            limits = disturbanceLimits;
            viewCamera = camera;
            Subscribe();
        }

        void Awake()
        {
            terrainMask = LayerMask.GetMask(SimSettings.TerrainLayer);
            robotMask = LayerMask.GetMask(SimSettings.RobotLayer);
            if (limits == null)
                limits = DisturbanceLimits.CreateDefault();
            if (viewCamera == null)
                viewCamera = Camera.main;
            RegisterTools();
        }

        void OnEnable() => Subscribe();

        void OnDisable()
        {
            if (subscribed && controller != null)
            {
                controller.Fell -= OnFell;
                controller.WasReset -= OnReset;
            }
            subscribed = false;
        }

        void Subscribe()
        {
            if (subscribed || controller == null)
                return;
            controller.Fell += OnFell;
            controller.WasReset += OnReset;
            subscribed = true;
        }

        void OnFell(Go2Controller c, string reason) => Record("fall", reason, false);
        void OnReset(Go2Controller c) => Record("reset", "robot back at spawn", false);

        /// <summary>Tool slots for keys 1–7; filled in step by step.</summary>
        void RegisterTools()
        {
            tools.Clear();
            foreach (var name in new[] { "Push", "Drag", "Ball", "Payload", "Friction", "Obstacle", "Deceptive terrain" })
                tools.Add(new PlaceholderTool(name));
        }

        /// <summary>Replace the tool in slot <paramref name="index"/> (0-based; key = index + 1).</summary>
        public void SetTool(int index, DisturbanceTool tool)
        {
            while (tools.Count <= index)
                tools.Add(new PlaceholderTool("–"));
            tools[index] = tool;
        }

        void Update()
        {
            var keyboard = Keyboard.current;
            if (keyboard == null)
                return;
            if (keyboard.hKey.wasPressedThisFrame)
                HardMode = !HardMode;
            if (keyboard.escapeKey.wasPressedThisFrame)
                Select(null);
            var digits = new[] { keyboard.digit1Key, keyboard.digit2Key, keyboard.digit3Key, keyboard.digit4Key, keyboard.digit5Key, keyboard.digit6Key, keyboard.digit7Key };
            for (var i = 0; i < digits.Length && i < tools.Count; i++)
                if (digits[i].wasPressedThisFrame)
                    Select(ActiveTool == tools[i] ? null : tools[i]);
            ActiveTool?.Update(this);
        }

        void FixedUpdate()
        {
            ActiveTool?.FixedUpdate(this);
        }

        public void Select(DisturbanceTool tool)
        {
            if (tool == ActiveTool)
                return;
            ActiveTool?.OnDeselect(this);
            ActiveTool = tool;
            ActiveTool?.OnSelect(this);
        }

        // --- limits and log --------------------------------------------------------------------------------

        /// <summary>Clamps to the active mode's range and reports whether the value is outside the training range.</summary>
        public float Limit(Quantity q, float value, out bool outsideTraining) => limits.Clamp(q, value, hardMode, out outsideTraining);

        public Vector2 Range(Quantity q) => limits.Range(q, hardMode);

        /// <summary>Adds an event stamped with the controller's clock and the robot's base state.</summary>
        public DisturbanceEvent Record(string type, string detail, bool outsideTraining)
        {
            var e = new DisturbanceEvent { Type = type, Detail = detail, OutsideTraining = outsideTraining };
            if (controller != null && controller.Initialized)
            {
                e.Time = controller.TotalTime;
                e.EpisodeTime = controller.SimTime;
                e.BasePosition = IsaacFrame.PolarToIsaac(controller.Body.BasePosition);
                e.BaseVelocity = IsaacFrame.PolarToIsaac(controller.Body.BaseComLinearVelocity);
            }
            Log.Add(e);
            return e;
        }

        void OnApplicationQuit()
        {
            if (!writeLogOnExit || Log.Events.Count == 0)
                return;
            var path = Path.Combine(Application.dataPath, "..", "Logs", $"disturbances_{DateTime.Now:yyyyMMdd_HHmmss}.csv");
            Log.WriteCsv(path);
            Debug.Log($"[BMB] Disturbance log ({Log.Events.Count} events): {Path.GetFullPath(path)}");
        }

        // --- mouse helpers for tools --------------------------------------------------------------------------

        public bool MouseRay(out Ray ray)
        {
            ray = default;
            if (viewCamera == null || Mouse.current == null)
                return false;
            ray = viewCamera.ScreenPointToRay(Mouse.current.position.ReadValue());
            return true;
        }

        /// <summary>Point on the terrain under the mouse.</summary>
        public bool MouseOnGround(out RaycastHit hit)
        {
            hit = default;
            return MouseRay(out var ray) && controller.Physics.Raycast(ray.origin, ray.direction, out hit, 500f, terrainMask, QueryTriggerInteraction.Ignore);
        }

        /// <summary>Robot link under the mouse.</summary>
        public bool MouseOnRobot(out RaycastHit hit)
        {
            hit = default;
            return MouseRay(out var ray) && controller.Physics.Raycast(ray.origin, ray.direction, out hit, 500f, robotMask, QueryTriggerInteraction.Ignore);
        }

        /// <summary>Mouse wheel steps this frame (+1 up, −1 down, 0 none); Ctrl + wheel is the camera's zoom.</summary>
        public static int ScrollSteps()
        {
            if (Keyboard.current != null && Keyboard.current.ctrlKey.isPressed)
                return 0;
            var y = Mouse.current?.scroll.ReadValue().y ?? 0f;
            return y > 0.01f ? 1 : y < -0.01f ? -1 : 0;
        }

        public static bool Shift => Keyboard.current != null && (Keyboard.current.leftShiftKey.isPressed || Keyboard.current.rightShiftKey.isPressed);
    }

    /// <summary>An interactive tool bound to a number key. Input handling only; effects go through the manager's API.</summary>
    public abstract class DisturbanceTool
    {
        public abstract string Name { get; }
        /// <summary>One-line usage, shown in the HUD while selected.</summary>
        public virtual string Help => "";
        /// <summary>Current parameters, shown in the HUD while selected.</summary>
        public virtual string Status(DisturbanceManager m) => "";
        public virtual void OnSelect(DisturbanceManager m) { }
        public virtual void OnDeselect(DisturbanceManager m) { }
        public virtual void Update(DisturbanceManager m) { }
        public virtual void FixedUpdate(DisturbanceManager m) { }
    }

    sealed class PlaceholderTool : DisturbanceTool
    {
        readonly string name;
        public PlaceholderTool(string name) => this.name = name;
        public override string Name => name;
        public override string Help => "not implemented yet";
    }
}
