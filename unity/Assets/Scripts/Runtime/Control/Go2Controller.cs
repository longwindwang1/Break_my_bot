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
        [Tooltip("Draw the height-scan hit points as gizmos")]
        [SerializeField] bool drawScan = true;

        [Header("Trial log")]
        [Tooltip("Log distance / speed / base height after this much simulated time (0 = off)")]
        [SerializeField] float trialSeconds = 10f;

        [Header("Fall handling")]
        [Tooltip("Reset when the base tilts more than this (deg) or drops below fallHeight")]
        [SerializeField] float fallTiltDeg = 70f;
        [SerializeField] float fallHeight = 0.08f;
        [SerializeField] bool resetOnFall = true;

        RobotInterface itf;
        PolicyRunner policy;
        ObservationBuilder builder;
        float[] obs, actions, lastAction, qTarget, q, qd, tau, scan;
        HeightScanner scanner;
        PhysicsScene physics;
        int terrainMask;
        int physicsStep;
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
        public bool Initialized => policy != null;
        public int PhysicsStep => physicsStep;
        public float SimTime => physicsStep * (itf?.Timing.PhysicsDt ?? 0f);
        public Vector3 Command { get; private set; }
        /// <summary>Base height above the terrain directly below it.</summary>
        public float BaseHeight => body.BasePosition.y - GroundHeightAt(body.BasePosition);
        public int Falls { get; private set; }
        /// <summary>Null for policies without a height scan.</summary>
        public HeightScanner Scanner => scanner;

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

        /// <summary>Wiring for code-built setups (the scene builder uses serialized fields instead).</summary>
        public void Configure(Go2Body robot, string policyVariant, CommandSource commands = null, bool resetAfterFall = true, float footFrictionMu = 0f)
        {
            body = robot;
            variant = policyVariant;
            commandSource = commands;
            resetOnFall = resetAfterFall;
            footFriction = footFrictionMu;
        }

        void Start() => Initialize();

        /// <summary>Loads the interface and the policy and resets to the spawn pose (the body's current x/z and yaw).</summary>
        public void Initialize()
        {
            if (Initialized)
                return;
            itf = RobotInterface.Load();
            body.Validate(itf);
            policy = PolicyRunner.Load(variant, itf);
            builder = new ObservationBuilder(itf, variant);
            physics = gameObject.scene.GetPhysicsScene();
            terrainMask = LayerMask.GetMask(SimSettings.TerrainLayer);
            if (builder.NeedsHeightScan)
            {
                scanner = new HeightScanner(itf.HeightScan, physics, terrainMask);
                scan = new float[scanner.Count];
            }
            if (footFriction > 0f)
                body.SetFootFriction(footFriction, itf.Friction.FootDynamicRatio, itf.Friction.UnityCombineMode());

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
            policy?.Dispose();
            policy = null;
        }

        /// <summary>Back to the spawn pose: default joint angles, zero velocities, last action 0, history cleared.</summary>
        public void ResetRobot()
        {
            body.Teleport(spawnPosition, spawnRotation, itf.DefaultJointPos);
            Array.Clear(lastAction, 0, lastAction.Length);
            Array.Clear(actions, 0, actions.Length);
            Array.Copy(itf.DefaultJointPos, qTarget, qTarget.Length);
            builder.Reset();
            physicsStep = 0;
            trialStart = spawnPosition;
            trialHeading = spawnRotation;
            heightSum = 0;
            heightSamples = 0;
            trialLogged = false;
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
            DCMotor.Torques(qTarget, q, qd, itf.Actuator, tau);
            body.SetJointTorques(tau);
            physicsStep++;
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
            if (trialLogged || trialSeconds <= 0f || SimTime < trialSeconds)
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
            if (tilt < fallTiltDeg && BaseHeight > fallHeight)
                return;
            Falls++;
            Debug.LogWarning($"[BMB] Fall #{Falls} at {SimTime:F2} s (tilt {tilt:F0} deg, base height {BaseHeight:F2} m)");
            if (resetOnFall)
                ResetRobot();
        }
    }
}
