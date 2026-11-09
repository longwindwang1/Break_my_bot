using System;
using System.Diagnostics;
using UnityEngine;

namespace BMB
{
    /// <summary>
    /// An imagined rollout ("dream") of the world model with the policy in the loop, for the ghost preview.
    /// It starts from a robot's latest policy step: the monitor already predicted step t+1. Each further step then
    /// 1. integrates the predicted base displacement (yaw frame) and takes the predicted state s;
    /// 2. ray-casts the height scan at that imagined pose;
    /// 3. builds the observation from s with a copy of the policy's history (v2) and runs the policy;
    /// 4. runs the world model on its own copied buffer.
    /// The command is held at the player's command at the start. The world model's dynamics only know normal physics,
    /// so on ice or after a push the dream walks on while the real robot does not: that gap is the surprise.
    /// Work is split into <see cref="Advance"/> calls under a time budget; nothing here touches the real robot.
    /// </summary>
    public sealed class WorldModelDreamer
    {
        readonly Go2Controller controller;
        readonly RobotInterface itf;
        readonly WorldModelLayout layout;
        readonly FrameHistory history;
        readonly ObservationBuilder builder;
        readonly HeightScanner scanner;
        readonly WorldModelPrediction prediction;
        readonly float[] obs, action, lastAction, scan, frame, jointPosRel, jointVel, wmIn;
        Vector3 command;
        Vector3 position;
        float yaw;
        int step;  // imagined steps done (the state now is t0 + step)

        /// <summary>Policy steps to imagine ahead (state t0 + Horizon is the ghost).</summary>
        public int Horizon { get; set; } = 25;
        public bool Busy { get; private set; }
        /// <summary>Base positions t0 .. t0 + Horizon of the last finished dream (Unity frame).</summary>
        public Vector3[] Path { get; private set; } = Array.Empty<Vector3>();
        /// <summary>The ghost: base pose and absolute joint angles at t0 + Horizon of the last finished dream.</summary>
        public Vector3 GhostPosition { get; private set; }
        public Quaternion GhostRotation { get; private set; } = Quaternion.identity;
        public float[] GhostJointPos { get; }
        /// <summary>Highest fall probability (shortest horizon) predicted along the last finished dream.</summary>
        public float MaxFallProb { get; private set; }
        /// <summary>Finished dreams since creation.</summary>
        public int Completed { get; private set; }
        /// <summary>Simulated time (controller TotalTime) at the start of the last finished dream.</summary>
        public double StartTime { get; private set; }

        Vector3[] path = Array.Empty<Vector3>();
        float maxFall;
        double startTime;

        public WorldModelDreamer(Go2Controller controller)
        {
            if (controller.WorldModel == null)
                throw new InvalidOperationException($"{controller.name}: enable the world model first");
            this.controller = controller;
            itf = controller.Interface;
            layout = controller.WorldModel.Layout;
            // ImagineStep reads the state by these offsets
            if (!System.Linq.Enumerable.SequenceEqual(layout.FrameTerms, new[] { "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions" }))
                throw new InvalidOperationException($"unexpected world-model frame terms [{string.Join(", ", layout.FrameTerms)}]");
            history = new FrameHistory(layout);
            builder = new ObservationBuilder(itf, controller.Variant);
            scanner = new HeightScanner(itf.HeightScan, controller.Physics, SimSettings.ScanMask);
            prediction = new WorldModelPrediction(itf.WorldModel);
            var n = itf.NumJoints;
            obs = new float[builder.Dim];
            action = new float[n];
            lastAction = new float[n];
            scan = new float[scanner.Count];
            frame = new float[layout.FrameDim];
            jointPosRel = new float[n];
            jointVel = new float[n];
            wmIn = new float[layout.InputDim];
            GhostJointPos = new float[n];
        }

        /// <summary>
        /// Starts a dream from the controller's latest policy step (call right after it, e.g. from PolicyStepped).
        /// Returns false if the monitor has no prediction yet.
        /// </summary>
        public bool Begin()
        {
            var m = controller.WorldModel;
            if (m == null || !m.HasPrediction || Horizon < 1)
                return false;
            history.CopyFrom(m.History);
            builder.CopyFrom(controller.Builder);
            prediction.CopyFrom(m.Prediction);
            for (var i = 0; i < lastAction.Length; i++)
                lastAction[i] = controller.Actions[i];  // a_t: the "actions" term of frame t+1
            command = controller.Command;
            position = controller.Body.BasePosition;
            yaw = WmPose.Yaw(controller.Body.BaseRotation);
            scanner.Deceive = controller.Scanner != null && controller.Scanner.Deceive;
            if (path.Length != Horizon + 1)
                path = new Vector3[Horizon + 1];
            path[0] = position;
            maxFall = prediction.FallProb[0];
            startTime = controller.TotalTime;
            step = 0;
            Busy = true;
            Consume();  // step t+1 is already predicted
            return true;
        }

        /// <summary>Stops the current dream (e.g. after a reset); the last finished one stays.</summary>
        public void Cancel() => Busy = false;

        /// <summary>
        /// Runs imagined steps until the dream finishes, <paramref name="maxSteps"/> steps were done, or the time
        /// budget is spent. Returns true when a dream finished in this call.
        /// </summary>
        public bool Advance(double budgetMs = double.PositiveInfinity, int maxSteps = int.MaxValue)
        {
            if (!Busy)
                return false;
            var sw = Stopwatch.StartNew();
            for (var k = 0; k < maxSteps && step < Horizon; k++)
            {
                ImagineStep();
                Consume();
                if (sw.Elapsed.TotalMilliseconds >= budgetMs)
                    break;
            }
            if (step < Horizon)
                return false;
            Publish();
            return true;
        }

        /// <summary>Takes the prediction for the next step: pose, state, fall risk.</summary>
        void Consume()
        {
            WmPose.Integrate(ref position, ref yaw, prediction.DispMean);
            step++;
            path[step] = position;
            maxFall = Mathf.Max(maxFall, prediction.FallProb[0]);
        }

        /// <summary>One policy step in imagination, from the predicted state of the current step.</summary>
        void ImagineStep()
        {
            var s = prediction.StateMean;
            // state_terms: base_lin_vel 0..2, base_ang_vel 3..5, projected_gravity 6..8, joint_pos 9..20, joint_vel 21..32
            Array.Copy(s, 9, jointPosRel, 0, jointPosRel.Length);
            Array.Copy(s, 9 + jointPosRel.Length, jointVel, 0, jointVel.Length);
            var gravity = new Vector3(s[6], s[7], s[8]);
            var rotation = WmPose.Rotation(yaw, gravity);
            scanner.Scan(position, rotation, scan);
            builder.BuildFromTerms(new Vector3(s[0], s[1], s[2]), new Vector3(s[3], s[4], s[5]), gravity, command,
                                   jointPosRel, jointVel, lastAction, scan, obs);
            controller.Policy.Run(obs, action);
            layout.FrameFromObs(obs, frame);
            history.Push(frame);
            history.BuildInput(action, obs, wmIn);
            controller.WorldModelRunner.Run(wmIn, prediction);
            Array.Copy(action, lastAction, action.Length);
        }

        void Publish()
        {
            Busy = false;
            var s = prediction.StateMean;  // state at t0 + Horizon
            GhostPosition = position;
            GhostRotation = WmPose.Rotation(yaw, new Vector3(s[6], s[7], s[8]));
            for (var i = 0; i < GhostJointPos.Length; i++)
                GhostJointPos[i] = itf.DefaultJointPos[i] + s[9 + i];
            if (Path.Length != path.Length)
                Path = new Vector3[path.Length];
            Array.Copy(path, Path, path.Length);
            MaxFallProb = maxFall;
            StartTime = startTime;
            Completed++;
        }
    }
}
