using System;

namespace BMB
{
    /// <summary>
    /// The world model as a side monitor of one robot (docs/unity_handoff.md §11.2). Once per policy step, after the
    /// policy produced a_t:
    /// <code>
    /// if episode_step > 0: score(s_t) against the prediction made at t−1
    /// wm_buffer.push(frame_t)             // first frame of an episode fills every slot
    /// prediction = world_model(wm_in(wm_buffer, a_t, scan_t, onehot))
    /// </code>
    /// It never produces actions. Several monitors may share one <see cref="WorldModelRunner"/> (same thread).
    /// </summary>
    public sealed class WorldModelMonitor
    {
        readonly WorldModelRunner runner;
        readonly float[] frame, state;
        bool hasPrediction;

        public WorldModelLayout Layout { get; }
        public FrameHistory History { get; }
        public SurprisalDetector Detector { get; }
        /// <summary>Prediction made at the last step, for the next one.</summary>
        public WorldModelPrediction Prediction { get; }
        /// <summary>wm_in of the last step.</summary>
        public float[] Input { get; }
        /// <summary>The last observed state s_t (state_terms order, Isaac base frame).</summary>
        public float[] State => state;
        public int EpisodeStep { get; private set; } = -1;
        public bool HasPrediction => hasPrediction;
        public string Policy => Layout.Policy;

        /// <summary>Raised on a surprisal alarm (spike or CUSUM), after the step's prediction is made.</summary>
        public event Action<WorldModelMonitor> Alarmed;

        public WorldModelMonitor(RobotInterface itf, string variant, WorldModelRunner runner, SurprisalParams surprisal = null)
        {
            this.runner = runner;
            Layout = new WorldModelLayout(itf, variant);
            History = new FrameHistory(Layout);
            Detector = new SurprisalDetector(surprisal ?? SurprisalParams.Load(itf, Layout.Policy));
            Prediction = new WorldModelPrediction(itf.WorldModel);
            Input = new float[Layout.InputDim];
            frame = new float[Layout.FrameDim];
            state = new float[Layout.StateDim];
        }

        /// <summary>Episode reset: empty buffer, no prediction, CUSUM back to 0.</summary>
        public void Reset()
        {
            History.Clear();
            hasPrediction = false;
            EpisodeStep = -1;
        }

        /// <param name="obs">The policy observation of this step (variant layout; frame and clipped scan are cut from it).</param>
        /// <param name="action">a_t, the raw policy output of this step.</param>
        /// <param name="episodeStep">Policy steps since the episode started; 0 also resets.</param>
        public void Step(float[] obs, float[] action, int episodeStep)
        {
            if (episodeStep == 0)
                Reset();
            EpisodeStep = episodeStep;
            Layout.FrameFromObs(obs, frame);
            Layout.StateFromFrame(frame, state);
            Detector.Step(state, hasPrediction ? Prediction.StateMean : null, hasPrediction ? Prediction.StateVar : null, episodeStep);
            History.Push(frame);
            History.BuildInput(action, obs, Input);
            runner.Run(Input, Prediction);
            hasPrediction = true;
            if (Detector.Alarm)
                Alarmed?.Invoke(this);
        }
    }
}
