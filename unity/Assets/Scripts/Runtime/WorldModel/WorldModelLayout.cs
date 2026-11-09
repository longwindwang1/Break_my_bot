using System;
using System.Collections.Generic;
using System.Linq;

namespace BMB
{
    /// <summary>
    /// Index bookkeeping for wm_in, from robot_interface.yaml only (port of models/world_model/layout.py), plus where
    /// a policy variant's observation vector holds the frame terms and the clipped height scan.
    /// A frame is one step of the world model's history: the history block's terms (45-d: base_lin_vel,
    /// base_ang_vel, projected_gravity, joint_pos, joint_vel, actions = the previous action). The state s (33-d) is
    /// the state_terms part of a frame.
    /// </summary>
    public sealed class WorldModelLayout
    {
        public string Variant { get; }
        /// <summary>World-model policy name ("v1" / "v2") and its index in the one-hot.</summary>
        public string Policy { get; }
        public int PolicyIndex { get; }
        public int HistoryLength { get; }
        public string[] FrameTerms { get; }
        public int[] FrameTermDims { get; }
        /// <summary>Start of each frame term inside a frame.</summary>
        public int[] FrameTermOffsets { get; }
        public int FrameDim { get; }
        public int StateDim { get; }
        public int ActionDim { get; }
        public int ScanDim { get; }
        public int InputDim { get; }
        /// <summary>Positions in the policy observation vector of the frame values, frame order.</summary>
        public int[] FrameIndexInObs { get; }
        /// <summary>Start of the (clipped) height_scan term in the policy observation vector.</summary>
        public int ScanOffsetInObs { get; }
        /// <summary>Positions of the state terms inside a frame.</summary>
        public int[] StateInFrame { get; }
        /// <summary>Positions in wm_in of the newest frame's state terms (what the ONNX adds the predicted delta to).</summary>
        public int[] CurrentStateInInput { get; }
        public int ActionOffset { get; }
        public int ScanOffset { get; }
        public int OneHotOffset { get; }
        public int NumPolicies { get; }

        public WorldModelLayout(RobotInterface itf, string variant)
        {
            var wm = itf.WorldModel ?? throw new InvalidOperationException("robot_interface.yaml has no world_model section");
            Variant = variant;
            Policy = WorldModelSpec.PolicyName(variant) ?? throw new ArgumentException($"policy variant {variant} has no world model (needs v1 rough or v2)");
            PolicyIndex = wm.Policies.IndexOf(Policy);
            if (PolicyIndex < 0)
                throw new ArgumentException($"world_model.policies has no {Policy}");
            NumPolicies = wm.Policies.Count;

            var names = wm.Input.Select(b => b.Name).ToArray();
            if (!names.SequenceEqual(new[] { "history", "action", "height_scan", "policy_onehot" }))
                throw new InvalidOperationException($"unexpected world_model.input blocks [{string.Join(", ", names)}]");
            var hist = wm.Block("history");
            HistoryLength = hist.HistoryLength;
            FrameTerms = hist.Terms.ToArray();
            FrameTermDims = FrameTerms.Select(t => itf.Observation.Term(t).Dim).ToArray();
            FrameTermOffsets = new int[FrameTerms.Length];
            for (var k = 1; k < FrameTerms.Length; k++)
                FrameTermOffsets[k] = FrameTermOffsets[k - 1] + FrameTermDims[k - 1];
            FrameDim = FrameTermDims.Sum();
            // the training rollout and the exporter assume frame = [state terms, actions]
            if (!FrameTerms.SequenceEqual(wm.StateTerms.Append("actions")))
                throw new InvalidOperationException($"history terms [{string.Join(", ", FrameTerms)}] != state_terms + actions");
            StateInFrame = wm.StateTerms.SelectMany(t =>
            {
                var k = Array.IndexOf(FrameTerms, t);
                return Enumerable.Range(FrameTermOffsets[k], FrameTermDims[k]);
            }).ToArray();
            StateDim = StateInFrame.Length;

            ActionDim = wm.Block("action").Dim;
            ScanDim = wm.Block("height_scan").Dim;
            ActionOffset = HistoryLength * FrameDim;
            ScanOffset = ActionOffset + ActionDim;
            OneHotOffset = ScanOffset + ScanDim;
            InputDim = OneHotOffset + wm.Block("policy_onehot").Dim;
            if (InputDim != wm.InputDim || wm.Block("policy_onehot").Dim != NumPolicies)
                throw new InvalidOperationException($"world_model.input adds up to {InputDim}, input_dim is {wm.InputDim}");
            if (StateDim != wm.OutputDim("state_mean"))
                throw new InvalidOperationException($"state_terms give {StateDim} dims, outputs.state_mean has {wm.OutputDim("state_mean")}");

            // newest frame = last slot of each term block
            CurrentStateInInput = wm.StateTerms.SelectMany(t =>
            {
                var k = Array.IndexOf(FrameTerms, t);
                var termBlock = HistoryLength * FrameTermOffsets[k];
                return Enumerable.Range(termBlock + (HistoryLength - 1) * FrameTermDims[k], FrameTermDims[k]);
            }).ToArray();

            // where the policy observation holds the current value of each term (groups without history)
            var termInObs = new Dictionary<string, int>();
            var offset = 0;
            foreach (var (_, history, terms) in itf.ObsGroups(variant))
            {
                var slots = Math.Max(1, history);
                foreach (var term in terms)
                {
                    if (slots == 1 && !termInObs.ContainsKey(term.Name))
                        termInObs[term.Name] = offset;
                    offset += slots * term.Dim;
                }
            }
            int InObs(string term) => termInObs.TryGetValue(term, out var o)
                ? o
                : throw new InvalidOperationException($"{variant} observation has no current-frame {term} term");
            FrameIndexInObs = FrameTerms.SelectMany((t, k) => Enumerable.Range(InObs(t), FrameTermDims[k])).ToArray();
            ScanOffsetInObs = InObs("height_scan");
            if (itf.Observation.Term("height_scan").Dim != ScanDim)
                throw new InvalidOperationException("world_model height_scan dim differs from the observation term");
        }

        /// <summary>The current frame, cut from a policy observation vector.</summary>
        public void FrameFromObs(float[] obs, float[] frame)
        {
            for (var i = 0; i < FrameDim; i++)
                frame[i] = obs[FrameIndexInObs[i]];
        }

        /// <summary>The state part (state_terms) of a frame.</summary>
        public void StateFromFrame(float[] frame, float[] state)
        {
            for (var i = 0; i < StateDim; i++)
                state[i] = frame[StateInFrame[i]];
        }
    }

    /// <summary>
    /// The world model's own frame buffer (history_length frames). After <see cref="Clear"/> the next push fills
    /// every slot with that frame. It is NOT cleared on a surprisal alarm (handoff §11.1), only on an episode reset.
    /// </summary>
    public sealed class FrameHistory
    {
        readonly WorldModelLayout layout;
        readonly float[][] frames;
        int newest = -1;

        public FrameHistory(WorldModelLayout layout)
        {
            this.layout = layout;
            frames = new float[layout.HistoryLength][];
            for (var i = 0; i < frames.Length; i++)
                frames[i] = new float[layout.FrameDim];
        }

        public bool IsEmpty => newest < 0;

        public void Clear() => newest = -1;

        public void Push(float[] frame)
        {
            if (newest < 0)
            {
                newest = 0;
                foreach (var f in frames)
                    Array.Copy(frame, f, layout.FrameDim);
                return;
            }
            newest = (newest + 1) % frames.Length;
            Array.Copy(frame, frames[newest], layout.FrameDim);
        }

        /// <summary>The newest frame (read-only by convention).</summary>
        public float[] Newest => frames[Math.Max(0, newest)];

        public void CopyFrom(FrameHistory other)
        {
            newest = other.newest;
            for (var i = 0; i < frames.Length; i++)
                Array.Copy(other.frames[i], frames[i], layout.FrameDim);
        }

        /// <summary>
        /// wm_in = [history term by term, each oldest → newest] + action + height scan + policy one-hot.
        /// </summary>
        public void BuildInput(float[] action, float[] obs, float[] wmIn) =>
            BuildInput(action, obs, layout.ScanOffsetInObs, wmIn);

        /// <param name="scanSource">Array holding the clipped height scan at <paramref name="scanOffset"/>.</param>
        public void BuildInput(float[] action, float[] scanSource, int scanOffset, float[] wmIn)
        {
            if (newest < 0)
                throw new InvalidOperationException("world-model history is empty; push a frame first");
            var H = frames.Length;
            var o = 0;
            for (var k = 0; k < layout.FrameTerms.Length; k++)
            {
                var start = layout.FrameTermOffsets[k];
                var dim = layout.FrameTermDims[k];
                for (var age = H - 1; age >= 0; age--)
                {
                    Array.Copy(frames[(newest - age + H) % H], start, wmIn, o, dim);
                    o += dim;
                }
            }
            Array.Copy(action, 0, wmIn, layout.ActionOffset, layout.ActionDim);
            Array.Copy(scanSource, scanOffset, wmIn, layout.ScanOffset, layout.ScanDim);
            for (var p = 0; p < layout.NumPolicies; p++)
                wmIn[layout.OneHotOffset + p] = p == layout.PolicyIndex ? 1f : 0f;
        }
    }
}
