using System;
using System.Collections.Generic;
using UnityEngine;

namespace BMB
{
    /// <summary>Robot state in Unity coordinates, as read from the articulation once per policy step.</summary>
    public struct RobotState
    {
        public Quaternion BaseRotation;
        /// <summary>World linear velocity of the base centre of mass.</summary>
        public Vector3 BaseComLinearVelocity;
        public Vector3 BaseAngularVelocity;
        /// <summary>Joint angles / velocities in policy order (rad, rad/s).</summary>
        public float[] JointPos;
        public float[] JointVel;
    }

    /// <summary>
    /// Builds the policy input exactly like Isaac Lab (reference: export/test_parity.py::build_obs).
    /// Terms come from robot_interface.yaml in order; no scaling, only per-term clip (the height scan).
    /// History groups (v2 proprio_history): per term, the last H frames oldest → newest, current frame last;
    /// after <see cref="Reset"/> the next frame fills every slot (Isaac Lab CircularBuffer semantics) — also what
    /// "clear the adaptation history" on a surprisal alarm means.
    /// </summary>
    public sealed class ObservationBuilder
    {
        sealed class Group
        {
            public int History;  // 0 = current frame only
            public List<ObsTermSpec> Terms;
            public float[][][] Buffer;  // [term][slot][dim], ring of max(1, History) frames
            public int Newest = -1;     // slot of the newest frame; -1 = empty
        }

        readonly RobotInterface itf;
        readonly List<Group> groups = new List<Group>();
        readonly Dictionary<string, float[]> terms = new Dictionary<string, float[]>();
        readonly float[] defaultJointPos;

        public string Variant { get; }
        public int Dim { get; }
        public bool NeedsHeightScan { get; }

        public ObservationBuilder(RobotInterface itf, string variant)
        {
            this.itf = itf;
            Variant = variant;
            defaultJointPos = itf.DefaultJointPos;
            foreach (var (_, history, groupTerms) in itf.ObsGroups(variant))
            {
                var slots = Math.Max(1, history);
                var group = new Group { History = history, Terms = groupTerms, Buffer = new float[groupTerms.Count][][] };
                for (var k = 0; k < groupTerms.Count; k++)
                {
                    group.Buffer[k] = new float[slots][];
                    for (var s = 0; s < slots; s++)
                        group.Buffer[k][s] = new float[groupTerms[k].Dim];
                    terms[groupTerms[k].Name] = new float[groupTerms[k].Dim];
                    Dim += slots * groupTerms[k].Dim;
                    NeedsHeightScan |= groupTerms[k].Name == "height_scan";
                }
                groups.Add(group);
            }
            if (Dim != itf.ObsDim(variant))
                throw new InvalidOperationException($"{variant}: groups add up to {Dim}, policies.{variant}.obs_dim is {itf.ObsDim(variant)}");
        }

        /// <summary>Forget the history; the next Build fills every history slot with its frame.</summary>
        public void Reset()
        {
            foreach (var g in groups)
                g.Newest = -1;
        }

        /// <summary>Copies another builder's history (same variant), e.g. to continue it in an imagined rollout.</summary>
        public void CopyFrom(ObservationBuilder other)
        {
            if (other.Variant != Variant)
                throw new ArgumentException($"cannot copy a {other.Variant} builder into a {Variant} one");
            for (var gi = 0; gi < groups.Count; gi++)
            {
                var (g, o) = (groups[gi], other.groups[gi]);
                g.Newest = o.Newest;
                for (var k = 0; k < g.Buffer.Length; k++)
                    for (var s = 0; s < g.Buffer[k].Length; s++)
                        Array.Copy(o.Buffer[k][s], g.Buffer[k][s], g.Buffer[k][s].Length);
            }
        }

        float[] jointPosRel;

        /// <param name="state">Robot state in Unity coordinates.</param>
        /// <param name="command">(vx, vy, wz) velocity command in the Isaac base frame.</param>
        /// <param name="lastAction">Raw policy output of the previous step (zeros after a reset).</param>
        /// <param name="heightScanRaw">base_z − hit_z − offset per ray (clipped here); null if the policy has no scan.</param>
        /// <param name="obs">Output, length <see cref="Dim"/>.</param>
        public void Build(in RobotState state, Vector3 command, float[] lastAction, float[] heightScanRaw, float[] obs)
        {
            jointPosRel ??= new float[defaultJointPos.Length];
            for (var i = 0; i < jointPosRel.Length; i++)
                jointPosRel[i] = state.JointPos[i] - defaultJointPos[i];
            BuildFromTerms(IsaacFrame.PolarInBase(state.BaseRotation, state.BaseComLinearVelocity),
                           IsaacFrame.AxialInBase(state.BaseRotation, state.BaseAngularVelocity),
                           IsaacFrame.PolarInBase(state.BaseRotation, IsaacFrame.GravityDirection),
                           command, jointPosRel, state.JointVel, lastAction, heightScanRaw, obs);
        }

        /// <summary>
        /// Same as <see cref="Build"/> from the observation terms themselves (Isaac base frame; joint_pos relative to
        /// the default pose), e.g. a state predicted by the world model.
        /// </summary>
        public void BuildFromTerms(Vector3 baseLinVel, Vector3 baseAngVel, Vector3 projectedGravity, Vector3 command,
                                   float[] jointPosRelative, float[] jointVel, float[] lastAction, float[] heightScanRaw, float[] obs)
        {
            if (obs.Length != Dim)
                throw new ArgumentException($"obs has {obs.Length} slots, expected {Dim}");

            Set3("base_lin_vel", baseLinVel);
            Set3("base_ang_vel", baseAngVel);
            Set3("projected_gravity", projectedGravity);
            Set3("velocity_commands", command);
            Array.Copy(jointPosRelative, terms["joint_pos"], terms["joint_pos"].Length);
            Array.Copy(jointVel, terms["joint_vel"], terms["joint_vel"].Length);
            Array.Copy(lastAction, terms["actions"], terms["actions"].Length);
            if (NeedsHeightScan)
            {
                if (heightScanRaw == null)
                    throw new ArgumentNullException(nameof(heightScanRaw), $"{Variant} needs the height scan");
                Array.Copy(heightScanRaw, terms["height_scan"], terms["height_scan"].Length);
            }

            var offset = 0;
            foreach (var g in groups)
            {
                var slots = g.Buffer[0].Length;
                var first = g.Newest < 0;
                g.Newest = first ? 0 : (g.Newest + 1) % slots;
                for (var k = 0; k < g.Terms.Count; k++)
                {
                    var term = g.Terms[k];
                    var value = terms[term.Name];
                    if (first)
                        for (var s = 0; s < slots; s++)
                            Store(value, term, g.Buffer[k][s]);
                    else
                        Store(value, term, g.Buffer[k][g.Newest]);
                    for (var age = slots - 1; age >= 0; age--)  // oldest first, current frame last
                    {
                        var frame = g.Buffer[k][(g.Newest - age + slots) % slots];
                        Array.Copy(frame, 0, obs, offset, frame.Length);
                        offset += frame.Length;
                    }
                }
            }
        }

        static void Store(float[] value, ObsTermSpec term, float[] slot)
        {
            if (term.Clip == null)
                Array.Copy(value, slot, slot.Length);
            else
                for (var i = 0; i < slot.Length; i++)
                    slot[i] = Mathf.Clamp(value[i], term.Clip[0], term.Clip[1]);
        }

        void Set3(string name, Vector3 v)
        {
            if (!terms.TryGetValue(name, out var t))
                return;  // term not used by this variant
            t[0] = v.x;
            t[1] = v.y;
            t[2] = v.z;
        }
    }
}
