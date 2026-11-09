using System;
using System.Collections.Generic;
using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Runs a locomotion policy on the Go2 in Unity physics, with the timing of Isaac Lab (docs/unity_handoff.md §5.1):
    /// every `decimation`-th physics step (before the step) build the observation from the state the previous steps
    /// produced, run the policy and set q_target = default + scale * a; every physics step recompute the DC-motor
    /// torque from the current q, qd. Also logs a trial (distance, speed, base height) and resets after a fall.
    ///
    /// In play mode FixedUpdate drives it. A headless runner (Editor/Go2Trial) sets <see cref="ExternallyStepped"/>
    /// and calls <see cref="ControlStep"/> + PhysicsScene.Simulate itself, independent of frame rate and focus.
    /// </summary>
    [DefaultExecutionOrder(-50)]
    public sealed class Go2Controller : MonoBehaviour
    {
        [SerializeField] Go2Body body;
        [Tooltip("policies.<variant> in robot_interface.yaml: flat, rough or v2")]
        [SerializeField] string variant = "flat";
        [SerializeField] CommandSource commandSource;
        [Tooltip("Static foot friction μ (dynamic = foot_dynamic_ratio · μ). ≤ 0 keeps the prefab's v1 material " +
                 "(0.8 / 0.6, what flat and rough were trained with). v2 was trained with 0.05–1.25; its reference used 1.0")]
        [SerializeField] float footFriction;
        [Tooltip("Ground friction map read per foot every physics step (optional; without it the ground is μ 1.0)")]
        [SerializeField] FrictionZones frictionZones;
        [Tooltip("Draw the height-scan hit points as gizmos")]
        [SerializeField] bool drawScan = true;

        [Header("Trial log")]
        [Tooltip("Log distance / speed / base height after this much simulated time (0 = off)")]
        [SerializeField] float trialSeconds = 10f;

        [Header("Fall handling")]
        [Tooltip("A fall is the base touching the ground (Isaac's base_contact termination), or optionally the base " +
                 "tilting more than fallTiltDeg / dropping below fallHeight above the terrain (≤ 0 disables those two)")]
        [SerializeField] float fallTiltDeg = 70f;
        [SerializeField] float fallHeight = 0.08f;
        [SerializeField] bool resetOnFall = true;

        [Header("World model")]
        [Tooltip("Run the world model (go2_wm.onnx) next to the policy: predictions, surprisal and alarms. Rough and v2 only")]
        [SerializeField] bool worldModel;

        RobotInterface itf;
        WorldModelRunner wmRunner;
        bool ownsWm;
        SurprisalParams wmSurprisal;
        WorldModelMonitor monitor;
        PolicyRunner policy;
        bool ownsPolicy;
        ObservationBuilder builder;
        float[] obs, actions, lastAction, qTarget, q, qd, tau, scan;
        float[] kpScale, kdScale;
        HeightScanner scanner;
        FootFrictionDriver footDriver;
        PhysicsScene physics;
        int terrainMask;
        int physicsStep;
        long totalPhysicsSteps;
        Vector3 spawnPosition;
        Quaternion spawnRotation;

        // trial
        Vector3 trialStart;
        Quaternion trialHeading;
        double heightSum;
        int heightSamples;
        bool trialLogged;

        public string Variant => variant;
        public RobotInterface Interface => itf;
        public Go2Body Body => body;
        public bool Initialized => builder != null;
        public int PhysicsStep => physicsStep;
        /// <summary>Time since the last reset (episode time).</summary>
        public float SimTime => physicsStep * (itf?.Timing.PhysicsDt ?? 0f);
        /// <summary>Simulated time since Initialize, across resets.</summary>
        public double TotalTime => totalPhysicsSteps * (double)(itf?.Timing.PhysicsDt ?? 0f);
        /// <summary>Physics scene the robot lives in (play scene or a headless preview scene).</summary>
        public PhysicsScene Physics => physics;
        public string LastFallReason { get; private set; }
        public Vector3 Command { get; private set; }
        /// <summary>Base height above the terrain directly below it.</summary>
        public float BaseHeight => body.BasePosition.y - GroundHeightAt(body.BasePosition);
        public int Falls { get; private set; }
        /// <summary>Null for policies without a height scan.</summary>
        public HeightScanner Scanner => scanner;
        /// <summary>Per-foot friction (ground μ under each foot, effective μ).</summary>
        public FootFrictionDriver FootFriction => footDriver;
        /// <summary>The world-model monitor (null unless enabled for a rough / v2 policy).</summary>
        public WorldModelMonitor WorldModel => monitor;
        internal ObservationBuilder Builder => builder;
        internal PolicyRunner Policy => policy;
        internal WorldModelRunner WorldModelRunner => wmRunner;

        /// <summary>Ground friction map read per foot (null = uniform μ 1.0 ground).</summary>
        public FrictionZones Zones
        {
            get => frictionZones;
            set
            {
                frictionZones = value;
                if (footDriver != null)
                    footDriver.Zones = value;
            }
        }

        /// <summary>Joint angles / velocities read at the last control step (policy order).</summary>
        public IReadOnlyList<float> JointPos => q;
        public IReadOnlyList<float> JointVel => qd;
        /// <summary>Raw policy output of the last policy step.</summary>
        public IReadOnlyList<float> Actions => actions;
        /// <summary>DC-motor torques applied at the last control step (N·m).</summary>
        public IReadOnlyList<float> Torques => tau;

        /// <summary>Set by headless runners: FixedUpdate does nothing, the runner calls ControlStep.</summary>
        public bool ExternallyStepped { get; set; }
        /// <summary>If set, used instead of the CommandSource (headless trials).</summary>
        public Vector3? CommandOverride { get; set; }
        /// <summary>Raised after every policy step (after the new targets are set).</summary>
        public event Action<Go2Controller> PolicyStepped;
        /// <summary>Raised when a fall is detected (before an automatic reset), with the reason.</summary>
        public event Action<Go2Controller, string> Fell;
        /// <summary>Raised after every reset to the spawn pose.</summary>
        public event Action<Go2Controller> WasReset;
        /// <summary>Raised on a world-model surprisal alarm (spike or CUSUM), inside the policy step.</summary>
        public event Action<Go2Controller, WorldModelMonitor> SurpriseAlarm;

        /// <summary>Wiring for code-built setups (the scene builder uses serialized fields instead).</summary>
        /// <param name="sharedPolicy">Optional inference runner shared by several robots (it is stateless); not disposed here.</param>
        public void Configure(Go2Body robot, string policyVariant, CommandSource commands = null, bool resetAfterFall = true,
                              float footFrictionMu = 0f, PolicyRunner sharedPolicy = null)
        {
            body = robot;
            variant = policyVariant;
            commandSource = commands;
            resetOnFall = resetAfterFall;
            footFriction = footFrictionMu;
            policy = sharedPolicy;
        }

        /// <summary>Fall criteria; Isaac's evaluation uses base contact only (tilt and height ≤ 0).</summary>
        public void SetFallCriteria(float tiltDeg, float minBaseHeight)
        {
            fallTiltDeg = tiltDeg;
            fallHeight = minBaseHeight;
        }

        /// <summary>
        /// Per-joint actuator gain scales (policy order), like v2's randomize_actuator_gains; null = nominal gains.
        /// </summary>
        public void SetGainScales(float[] stiffnessScale, float[] dampingScale)
        {
            kpScale = stiffnessScale;
            kdScale = dampingScale;
        }

        /// <summary>
        /// Runs the world model next to the policy (rough / v2). Before Initialize it only sets what Initialize will
        /// create; after it, the monitor starts at the next episode step.
        /// </summary>
        /// <param name="sharedRunner">Optional runner shared by several robots (not disposed here).</param>
        /// <param name="surprisal">Thresholds; null = Unity calibration if present, else the yaml's.</param>
        public void EnableWorldModel(WorldModelRunner sharedRunner = null, SurprisalParams surprisal = null)
        {
            worldModel = true;
            if (sharedRunner != null)
                wmRunner = sharedRunner;
            wmSurprisal = surprisal;
            if (builder != null && monitor == null)
                CreateMonitor();
        }

        void CreateMonitor()
        {
            if (WorldModelSpec.PolicyName(variant) == null)
            {
                Debug.LogWarning($"[BMB] {name}: no world model for the {variant} policy");
                return;
            }
            ownsWm = wmRunner == null;
            if (ownsWm)
                wmRunner = WorldModelRunner.Load(itf);
            monitor = new WorldModelMonitor(itf, variant, wmRunner, wmSurprisal);
            monitor.Alarmed += m => SurpriseAlarm?.Invoke(this, m);
        }

        /// <summary>Where the robot returns on reset (Unity frame; y is recomputed from the terrain).</summary>
        public void SetSpawn(Vector3 position, float yawDeg)
        {
            spawnPosition = new Vector3(position.x, GroundHeightAt(position) + itf.InitBaseHeight, position.z);
            spawnRotation = Quaternion.Euler(0f, yawDeg, 0f);
        }

        void Start() => Initialize();

        /// <summary>Loads the interface and the policy and resets to the spawn pose (the body's current x/z and yaw).</summary>
        public void Initialize()
        {
            if (builder != null)
                return;
            itf = RobotInterface.Load();
            body.Validate(itf);
            ownsPolicy = policy == null;
            if (ownsPolicy)
                policy = PolicyRunner.Load(variant, itf);
            builder = new ObservationBuilder(itf, variant);
            if (policy.ObsDim != builder.Dim)
                throw new InvalidOperationException($"{name}: shared policy {policy.Name} takes {policy.ObsDim} inputs, {variant} builds {builder.Dim}");
            physics = gameObject.scene.GetPhysicsScene();
            terrainMask = LayerMask.GetMask(SimSettings.TerrainLayer);
            if (builder.NeedsHeightScan)
            {
                scanner = new HeightScanner(itf.HeightScan, physics, SimSettings.ScanMask);  // hidden-pit covers included
                scan = new float[scanner.Count];
            }
            footDriver = new FootFrictionDriver(body, itf, footFriction) { Zones = frictionZones };
            if (worldModel)
                CreateMonitor();

            var n = itf.NumJoints;
            obs = new float[builder.Dim];
            actions = new float[n];
            lastAction = new float[n];
            qTarget = new float[n];
            q = new float[n];
            qd = new float[n];
            tau = new float[n];

            // Like Isaac: init_base_height above the terrain at the spawn point.
            spawnPosition = new Vector3(body.BasePosition.x, 0f, body.BasePosition.z);
            spawnPosition.y = GroundHeightAt(spawnPosition) + itf.InitBaseHeight;
            spawnRotation = Quaternion.Euler(0f, body.BaseRotation.eulerAngles.y, 0f);
            ResetRobot();
        }

        void OnDestroy() => Shutdown();

        /// <summary>Releases the inference worker (OnDestroy does not run for edit-mode objects; call it there).</summary>
        public void Shutdown()
        {
            if (ownsPolicy)
                policy?.Dispose();
            if (ownsWm)
                wmRunner?.Dispose();
            policy = null;
            wmRunner = null;
            monitor = null;
            builder = null;
        }

        /// <summary>Back to the spawn pose: default joint angles, zero velocities, last action 0, history cleared.</summary>
        public void ResetRobot()
        {
            body.Teleport(spawnPosition, spawnRotation, itf.DefaultJointPos);
            Array.Clear(lastAction, 0, lastAction.Length);
            Array.Clear(actions, 0, actions.Length);
            Array.Copy(itf.DefaultJointPos, qTarget, qTarget.Length);
            builder.Reset();
            monitor?.Reset();
            physicsStep = 0;
            trialStart = spawnPosition;
            trialHeading = spawnRotation;
            heightSum = 0;
            heightSamples = 0;
            trialLogged = false;
            WasReset?.Invoke(this);
        }

        void FixedUpdate()
        {
            if (!ExternallyStepped && Initialized)
                ControlStep();
        }

        /// <summary>Everything that happens before one physics step. The caller then advances physics by physics_dt.</summary>
        public void ControlStep()
        {
            body.ReadJoints(q, qd);
            if (physicsStep % itf.Timing.Decimation == 0)
                PolicyStep();
            DCMotor.Torques(qTarget, q, qd, itf.Actuator, tau, kpScale, kdScale);
            body.SetJointTorques(tau);
            footDriver.Update();
            physicsStep++;
            totalPhysicsSteps++;
        }

        void PolicyStep()
        {
            Command = CommandOverride ?? (commandSource != null ? commandSource.Command : Vector3.zero);
            var state = new RobotState
            {
                BaseRotation = body.BaseRotation,
                BaseComLinearVelocity = body.BaseComLinearVelocity,
                BaseAngularVelocity = body.BaseAngularVelocity,
                JointPos = q,
                JointVel = qd,
            };
            scanner?.Scan(body.BasePosition, body.BaseRotation, scan);
            builder.Build(state, Command, lastAction, scan, obs);
            policy.Run(obs, actions);
            for (var i = 0; i < actions.Length; i++)
                qTarget[i] = itf.DefaultJointPos[i] + itf.Action.Scale * actions[i];
            Array.Copy(actions, lastAction, actions.Length);
            monitor?.Step(obs, actions, physicsStep / itf.Timing.Decimation);

            heightSum += BaseHeight;
            heightSamples++;
            PolicyStepped?.Invoke(this);
            LogTrial();
            CheckFall();
        }

        /// <summary>Displacement since the last reset in the spawn heading frame, Isaac axes (x forward, y left).</summary>
        public Vector3 Displacement => IsaacFrame.PolarToIsaac(Quaternion.Inverse(trialHeading) * (body.BasePosition - trialStart));

        public float MeanBaseHeight => heightSamples > 0 ? (float)(heightSum / heightSamples) : BaseHeight;

        void LogTrial()
        {
            if (trialLogged || trialSeconds <= 0f || SimTime < trialSeconds || ExternallyStepped)
                return;
            trialLogged = true;
            var d = Displacement;
            Debug.Log($"[BMB] Trial {variant}: {SimTime:F2} s, command {Command:F2} (Isaac vx, vy, wz), travelled x {d.x:F2} m, y {d.y:F2} m " +
                      $"({d.x / SimTime:F3} m/s forward), mean base height {MeanBaseHeight:F3} m, falls {Falls}");
        }

        /// <summary>Height of the Terrain layer below a point (0 if there is none).</summary>
        public float GroundHeightAt(Vector3 point)
        {
            if (!physics.IsValid())
                return 0f;
            var origin = new Vector3(point.x, point.y + 50f, point.z);
            return physics.Raycast(origin, Vector3.down, out var hit, 1000f, terrainMask, QueryTriggerInteraction.Ignore) ? hit.point.y : 0f;
        }

        void OnDrawGizmos()
        {
            if (!drawScan || scanner == null)
                return;
            Gizmos.color = new Color(0.1f, 0.9f, 0.4f);
            foreach (var h in scanner.Hits)
                if (!float.IsNaN(h.x))
                    Gizmos.DrawSphere(h, 0.012f);
        }

        void CheckFall()
        {
            var tilt = Vector3.Angle(body.BaseRotation * Vector3.up, Vector3.up);
            string reason = null;
            if (body.BaseTouches(physics, terrainMask))
                reason = "base contact";
            else if (fallTiltDeg > 0f && tilt >= fallTiltDeg)
                reason = $"tilt {tilt:F0} deg";
            else if (fallHeight > 0f && BaseHeight <= fallHeight)
                reason = $"base height {BaseHeight:F2} m";
            if (reason == null)
                return;
            Falls++;
            LastFallReason = reason;
            if (!ExternallyStepped)
                Debug.LogWarning($"[BMB] Fall #{Falls} at {SimTime:F2} s ({reason})");
            Fell?.Invoke(this, reason);
            if (resetOnFall)
                ResetRobot();
        }
    }
}
