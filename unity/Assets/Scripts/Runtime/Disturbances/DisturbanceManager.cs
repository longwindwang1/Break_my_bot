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
        [SerializeField] FrictionZones frictionZones;
        [SerializeField] LevelGround level;
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
        public FrictionZones Zones => frictionZones;
        public LevelGround Level => level;
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
        public void Configure(Go2Controller robotController, DisturbanceLimits disturbanceLimits, Camera camera = null,
                              FrictionZones zones = null, LevelGround levelGround = null)
        {
            controller = robotController;
            limits = disturbanceLimits;
            viewCamera = camera;
            frictionZones = zones;
            level = levelGround;
            if (zones != null && controller != null)
                controller.Zones = zones;
            EnsureInitialized();
            Subscribe();
        }

        bool initialized;

        void Awake() => EnsureInitialized();

        /// <summary>Awake's work; also called by Configure because Awake does not run for edit-mode objects.</summary>
        void EnsureInitialized()
        {
            if (initialized)
                return;
            initialized = true;
            terrainMask = LayerMask.GetMask(SimSettings.TerrainLayer);
            robotMask = LayerMask.GetMask(SimSettings.RobotLayer);
            if (limits == null)
                limits = DisturbanceLimits.CreateDefault();
            if (viewCamera == null)
                viewCamera = Camera.main;
            RegisterTools();
            StatusLines.Add(PayloadStatus);
            StatusLines.Add(FrictionStatus);
            if (frictionZones != null && controller != null)
                controller.Zones = frictionZones;
        }

        void OnEnable() => Subscribe();

        void OnDisable()
        {
            if (subscribed && controller != null)
            {
                controller.Fell -= OnFell;
                controller.WasReset -= OnReset;
                controller.SurpriseAlarm -= OnSurprise;
            }
            if (subscribed && level != null)
                level.CoverBroken -= OnCoverBroken;
            subscribed = false;
        }

        void Subscribe()
        {
            if (subscribed || controller == null)
                return;
            controller.Fell += OnFell;
            controller.WasReset += OnReset;
            controller.SurpriseAlarm += OnSurprise;
            if (level != null)
                level.CoverBroken += OnCoverBroken;
            subscribed = true;
        }

        /// <summary>Surprisal alarms are logged at most once per this many simulated seconds (ice re-alarms 1–2×/s).</summary>
        public const float SurpriseLogInterval = 1f;
        double lastSurpriseLogged = double.NegativeInfinity;
        int surprisesSinceLogged;

        void OnSurprise(Go2Controller c, WorldModelMonitor m)
        {
            surprisesSinceLogged++;
            if (c.TotalTime - lastSurpriseLogged < SurpriseLogInterval)
                return;
            var d = m.Detector;
            var p = d.Params;
            var what = d.Spike
                ? $"spike |z| {d.SpikeStat:0} > {p.SpikeZ:0} (base velocity)"
                : $"CUSUM g {d.GPeak:0} > h {p.H:0}";
            var more = surprisesSinceLogged > 1 ? $" (+{surprisesSinceLogged - 1} since last entry)" : "";
            Record("surprise", what + more, false);
            lastSurpriseLogged = c.TotalTime;
            surprisesSinceLogged = 0;
        }

        void OnFell(Go2Controller c, string reason) => Record("fall", reason, false);
        void OnReset(Go2Controller c) => Record("reset", "robot back at spawn", false);

        /// <summary>Tool slots for keys 1–7; filled in step by step.</summary>
        void RegisterTools()
        {
            tools.Clear();
            tools.Add(new PushTool());
            tools.Add(new DragTool());
            tools.Add(new BallTool());
            tools.Add(new PayloadTool());
            tools.Add(new FrictionTool());
            tools.Add(new ObstacleTool());
            tools.Add(new DeceptiveTool());
        }

        /// <summary>Set by the game UI every frame: the mouse is over a UI panel, so tools ignore world clicks.</summary>
        public bool PointerOverUI { get; set; }

        /// <summary>The mouse may act on the world this frame (not over the UI).</summary>
        public bool WorldPointer => !PointerOverUI;

        // --- disturbances: forces and payload ----------------------------------------------------------------

        /// <summary>
        /// Push like Isaac's push_by_setting_velocity: adds Δv to the base's world velocity (Isaac world axes x, y;
        /// each axis clamped to the push range). Returns the Δv applied.
        /// </summary>
        public Vector2 Push(Vector2 dvIsaacWorld)
        {
            var dx = Limit(Quantity.PushAxis, dvIsaacWorld.x, out var ox);
            var dy = Limit(Quantity.PushAxis, dvIsaacWorld.y, out var oy);
            var root = controller.Body.Root;
            root.linearVelocity += IsaacFrame.PolarFromIsaac(new Vector3(dx, dy, 0f));
            Record("push", $"dv=({dx:F2}, {dy:F2}) m/s, |dv| {Mathf.Sqrt(dx * dx + dy * dy):F2}", ox || oy);
            return new Vector2(dx, dy);
        }

        /// <summary>Random push with each axis uniform in ±<paramref name="maxAxis"/> (Isaac's velocity_range).</summary>
        public Vector2 PushRandom(float maxAxis, System.Random rng) =>
            Push(new Vector2((float)(rng.NextDouble() * 2 - 1) * maxAxis, (float)(rng.NextDouble() * 2 - 1) * maxAxis));

        /// <summary>Payload on the base: added mass (kg) and COM offset (Isaac base axes, m), both clamped. (0, 0) removes it.</summary>
        public void SetPayload(float addedMass, Vector3 comOffsetIsaac)
        {
            var m = Limit(Quantity.Payload, addedMass, out var om);
            var cx = Limit(Quantity.ComXY, comOffsetIsaac.x, out var ox);
            var cy = Limit(Quantity.ComXY, comOffsetIsaac.y, out var oy);
            var cz = Limit(Quantity.ComZ, comOffsetIsaac.z, out var oz);
            controller.Body.SetPayload(m, new Vector3(cx, cy, cz));
            Record("payload", $"{m:+0.0;-0.0;0} kg, COM ({cx * 100:F0}, {cy * 100:F0}, {cz * 100:F0}) cm", om || ox || oy || oz);
        }

        /// <summary>
        /// External force on a robot link at a world point (drag tool), its magnitude clamped to the drag-force limit.
        /// Call it every physics step while pulling; not logged (the caller records one event per drag).
        /// Returns the force applied.
        /// </summary>
        public Vector3 ApplyDragForce(ArticulationBody link, Vector3 worldPoint, Vector3 force, out bool outsideTraining)
        {
            var magnitude = Limit(Quantity.DragForce, force.magnitude, out outsideTraining);
            var f = force.sqrMagnitude > 0f ? force.normalized * magnitude : Vector3.zero;
            link.AddForceAtPosition(f, worldPoint);
            return f;
        }

        // --- disturbances: balls ------------------------------------------------------------------------------

        public const float BallLifetime = 10f;
        readonly List<Rigidbody> balls = new List<Rigidbody>();
        Material ballMaterial;
        PhysicsMaterial ballBounce;

        public IReadOnlyList<Rigidbody> Balls => balls;

        /// <summary>Ball radius for a mass (m): solid rubber-ish, 0.5 kg ≈ 8.5 cm.</summary>
        public static float BallRadius(float mass) => 0.05f * Mathf.Pow(Mathf.Max(mass, 0.01f) / 0.1f, 1f / 3f);

        /// <summary>
        /// Launch velocity from <paramref name="origin"/> that lands on <paramref name="target"/> at the given speed
        /// (the low ballistic arc); straight at the target if it is out of reach.
        /// </summary>
        public static Vector3 AimVelocity(Vector3 origin, Vector3 target, float speed)
        {
            var d = target - origin;
            var flat = new Vector3(d.x, 0f, d.z);
            var x = flat.magnitude;
            if (x < 1e-3f)
                return d.normalized * speed;
            var g = -UnityEngine.Physics.gravity.y;
            var v2 = speed * speed;
            var disc = v2 * v2 - g * (g * x * x + 2f * d.y * v2);
            var angle = disc >= 0f ? Mathf.Atan((v2 - Mathf.Sqrt(disc)) / (g * x)) : Mathf.Atan2(d.y, x);
            return flat / x * (Mathf.Cos(angle) * speed) + Vector3.up * (Mathf.Sin(angle) * speed);
        }

        /// <summary>
        /// Throws a ball (Default layer: hits the robot and the ground, invisible to the height scan). Momentum m·|v| is
        /// clamped by lowering the speed. Removed after <see cref="BallLifetime"/> s (play mode). Returns its body.
        /// </summary>
        public Rigidbody ThrowBall(Vector3 origin, Vector3 velocity, float mass)
        {
            mass = Mathf.Max(mass, 0.01f);
            var p = Limit(Quantity.BallMomentum, mass * velocity.magnitude, out var outside);
            var v = velocity.sqrMagnitude > 0f ? velocity.normalized * (p / mass) : Vector3.zero;

            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = "Ball";
            UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(go, gameObject.scene);
            var r = BallRadius(mass);
            go.transform.SetPositionAndRotation(origin, Quaternion.identity);
            go.transform.localScale = Vector3.one * (2f * r);
            ballBounce ??= new PhysicsMaterial("BallBounce") { hideFlags = HideFlags.DontSave, bounciness = 0.4f, staticFriction = 0.6f, dynamicFriction = 0.5f };
            go.GetComponent<Collider>().sharedMaterial = ballBounce;
            var renderer = go.GetComponent<MeshRenderer>();
            if (BallMaterial() is { } mat)
                renderer.sharedMaterial = mat;
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;

            var body = go.AddComponent<Rigidbody>();
            body.mass = mass;
            body.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            body.interpolation = RigidbodyInterpolation.Interpolate;
            body.linearVelocity = v;
            var projectile = go.AddComponent<BallProjectile>();
            projectile.Manager = this;
            projectile.Mass = mass;
            if (Application.isPlaying)
            {
                AddTrail(go, r);
                Destroy(go, BallLifetime);
            }
            balls.RemoveAll(b => b == null);
            balls.Add(body);
            Record("ball", $"thrown: {mass:F1} kg at {v.magnitude:F1} m/s, p {p:F1} N·s", outside);
            return body;
        }

        public void ClearBalls()
        {
            foreach (var b in balls)
                if (b != null)
                {
                    if (Application.isPlaying) Destroy(b.gameObject);
                    else DestroyImmediate(b.gameObject);
                }
            balls.Clear();
        }

        Material BallMaterial()
        {
            if (ballMaterial != null || !Application.isPlaying)
                return ballMaterial;
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                return null;
            ballMaterial = new Material(shader) { name = "Ball" };
            var colour = new Color(1f, 0.45f, 0.08f);
            ballMaterial.SetColor("_BaseColor", colour);
            ballMaterial.EnableKeyword("_EMISSION");
            ballMaterial.SetColor("_EmissionColor", colour * 1.5f);
            return ballMaterial;
        }

        static void AddTrail(GameObject go, float radius)
        {
            var trail = go.AddComponent<TrailRenderer>();
            trail.time = 0.35f;
            trail.widthMultiplier = radius * 1.4f;
            trail.widthCurve = AnimationCurve.Linear(0f, 1f, 1f, 0f);
            trail.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            trail.sharedMaterial = new Material(Shader.Find("Sprites/Default"));
            trail.startColor = new Color(1f, 0.55f, 0.1f, 0.8f);
            trail.endColor = new Color(1f, 0.17f, 0.84f, 0f);
        }

        // --- disturbances: ground friction ---------------------------------------------------------------------

        /// <summary>Paints ground μ (clamped) in a disc; <paramref name="record"/> = false while a stroke is in progress.</summary>
        public float PaintFriction(Vector3 centre, float radius, float mu, bool record = true)
        {
            var v = Limit(Quantity.Friction, mu, out var outside);
            var cells = RequireZones().Paint(centre, radius, v);
            if (record)
                Record("friction", $"zone μ {v:F2}, r {radius:F1} m at {FormatGround(centre)} ({cells} cells)", outside);
            return v;
        }

        public void EraseFriction(Vector3 centre, float radius, bool record = true)
        {
            var cells = RequireZones().Erase(centre, radius);
            if (record)
                Record("friction", $"erased r {radius:F1} m at {FormatGround(centre)} ({cells} cells)", false);
        }

        /// <summary>Ground μ wherever nothing is painted (1.0 = normal ground). Like Isaac's mu_drop when changed mid-run.</summary>
        public float SetGlobalFriction(float mu)
        {
            var v = Limit(Quantity.Friction, mu, out var outside);
            RequireZones().GlobalMu = v;
            Record("friction", $"global ground μ {v:F2}", outside);
            return v;
        }

        public void ClearFriction()
        {
            RequireZones().ClearAll();
            Record("friction", "all zones cleared", false);
        }

        FrictionZones RequireZones()
        {
            if (frictionZones == null)
            {
                frictionZones = new GameObject("FrictionZones").AddComponent<FrictionZones>();
                UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(frictionZones.gameObject, gameObject.scene);
            }
            if (controller != null && controller.Zones != frictionZones)
                controller.Zones = frictionZones;
            return frictionZones;
        }

        // --- disturbances: obstacles ------------------------------------------------------------------------------

        /// <summary>
        /// Adds a ground feature at a world point, its size parameter clamped to the active limits (box / pit / rough:
        /// obstacle height; stairs: step height; ramp: slope). Stairs and ramps get their natural length, Isaac tiles
        /// are 8 m with yaw in multiples of 90°. Returns the feature added to <see cref="Level"/>.
        /// </summary>
        public GroundFeature PlaceObstacle(GroundFeature f, Vector3 worldCentre)
        {
            if (level == null)
                throw new InvalidOperationException("no LevelGround to place obstacles on");
            var outside = false;
            switch (f.type)
            {
                case GroundFeatureType.Box:
                case GroundFeatureType.Pit:
                case GroundFeatureType.Rough:
                case GroundFeatureType.HiddenPit:
                case GroundFeatureType.BreakableIce:
                    f.height = Limit(Quantity.ObstacleHeight, f.height, out outside);
                    break;
                case GroundFeatureType.Stairs:
                    f.height = Limit(Quantity.StairStep, f.height, out outside);
                    f.steps = Mathf.Max(1, f.steps);
                    break;
                case GroundFeatureType.Ramp:
                    f.slopeDeg = Limit(Quantity.Slope, f.slopeDeg * Mathf.Deg2Rad, out outside) * Mathf.Rad2Deg;
                    break;
                case GroundFeatureType.IsaacTile:
                    f.difficulty = Mathf.Clamp01(f.difficulty);
                    f.yawDeg = Mathf.Round(f.yawDeg / 90f) * 90f;
                    f.size = new Vector2(IsaacTiles.Size, IsaacTiles.Size);
                    break;
            }
            var length = GroundFeature.NaturalLength(f.type, f.steps, f.RampLength);
            if (length > 0f)
                f.size = new Vector2(f.type == GroundFeatureType.IsaacTile ? IsaacTiles.Size : f.size.x, length);
            var snapped = level.Snap(worldCentre) - level.transform.position;
            f.centre = new Vector2(snapped.x, snapped.z);
            level.Add(f);
            Record(f.Covered ? "deceptive" : "obstacle", $"{f.Describe()} at {FormatGround(worldCentre)}, yaw {f.yawDeg:F0}°", outside);
            return f;
        }

        /// <summary>Puts every broken ice slab back.</summary>
        public void RestoreIce()
        {
            if (level == null)
                return;
            level.RestoreCovers();
            Record("deceptive", "broken ice restored", false);
        }

        void OnCoverBroken(DeceptiveCover cover) =>
            Record("deceptive", $"ice broke after {cover.ContactTime:F2} s of contact ({cover.Feature.size.x:F1}×{cover.Feature.size.y:F1} m over {cover.Feature.height * 100:F0} cm)", false);

        /// <summary>Removes the topmost feature under a world point; false if there is none.</summary>
        public bool RemoveObstacleAt(Vector3 world)
        {
            var f = level != null ? level.FeatureAt(world) : null;
            if (f == null)
                return false;
            level.Remove(f.id);
            Record("obstacle", $"removed {f.Describe()}", false);
            return true;
        }

        public void ClearObstacles()
        {
            if (level == null)
                return;
            level.Clear();
            Record("obstacle", "all obstacles removed", false);
        }

        static string FormatGround(Vector3 unityPoint)
        {
            var p = IsaacFrame.PolarToIsaac(unityPoint);
            return $"({p.x:F1}, {p.y:F1})";
        }

        // --- HUD lines -------------------------------------------------------------------------------------------

        string PayloadStatus()
        {
            var b = controller.Body;
            if (b.AddedMass == 0f && b.ComOffset == Vector3.zero)
                return "Payload: none";
            limits.Clamp(Quantity.Payload, b.AddedMass, false, out var outside);
            var text = $"Payload: {b.AddedMass:+0.0;-0.0} kg, COM ({b.ComOffset.x * 100:F0}, {b.ComOffset.y * 100:F0}, {b.ComOffset.z * 100:F0}) cm";
            return outside ? $"<color=#ff5a4a>{text}</color>" : text;
        }

        string FrictionStatus()
        {
            var f = controller.FootFriction;
            if (f == null)
                return "";
            var sb = new System.Text.StringBuilder("Foot μ:");
            for (var i = 0; i < 4; i++)
            {
                var mu = f.EffectiveStatic(i);
                var name = Go2Body.FootNames[i].Substring(0, 2);
                sb.Append(mu < 0.5f ? $" {name} <color=#6fb7ff>{mu:F2}</color>" : $" {name} {mu:F2}");
            }
            if (frictionZones != null)
                sb.Append($"   ground: global {frictionZones.GlobalMu:F2}, {frictionZones.PaintedCells} cells painted");
            return sb.ToString();
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
                if (digits[i].wasPressedThisFrame && tools[i].Available)
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

        /// <summary>Camera ray through the mouse; false while the mouse is over the UI (tools then ignore it).</summary>
        public bool MouseRay(out Ray ray)
        {
            ray = default;
            if (viewCamera == null || Mouse.current == null || PointerOverUI)
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

        /// <summary>Wheel steps for tools: none while the mouse is over the UI.</summary>
        public int Scroll() => PointerOverUI ? 0 : ScrollSteps();

        /// <summary>Left button pressed this frame on the world (not on the UI).</summary>
        public bool WorldClick => !PointerOverUI && Mouse.current != null && Mouse.current.leftButton.wasPressedThisFrame;

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

    /// <summary>
    /// An interactive tool bound to a number key. Input handling only; effects go through the manager's API.
    /// Its parameters are described by <see cref="BuildControls"/> for the game UI.
    /// </summary>
    public abstract class DisturbanceTool
    {
        public abstract string Name { get; }
        public virtual NeonIconKind Icon => NeonIconKind.Lock;
        /// <summary>False for tools not implemented yet (shown locked in the UI).</summary>
        public virtual bool Available => true;
        /// <summary>One-line usage, shown in the HUD while selected.</summary>
        public virtual string Help => "";
        /// <summary>Short mouse hint at the bottom of the tool card.</summary>
        public virtual string Hint => "";
        /// <summary>Current parameters, shown in the HUD while selected.</summary>
        public virtual string Status(DisturbanceManager m) => "";
        /// <summary>Parameters and actions for the tool card (built once per manager).</summary>
        public virtual System.Collections.Generic.List<ToolControl> BuildControls(DisturbanceManager m) => new System.Collections.Generic.List<ToolControl>();
        public virtual void OnSelect(DisturbanceManager m) { }
        public virtual void OnDeselect(DisturbanceManager m) { }
        public virtual void Update(DisturbanceManager m) { }
        public virtual void FixedUpdate(DisturbanceManager m) { }
    }

    sealed class PlaceholderTool : DisturbanceTool
    {
        readonly string name;
        readonly NeonIconKind icon;
        public PlaceholderTool(string name, NeonIconKind icon = NeonIconKind.Lock)
        {
            this.name = name;
            this.icon = icon;
        }
        public override string Name => name;
        public override NeonIconKind Icon => icon;
        public override bool Available => false;
        public override string Help => "not implemented yet";
        public override string Hint => "Coming soon";
    }
}
