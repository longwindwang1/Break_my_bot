using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace BMB
{
    /// <summary>
    /// Typed view of configs/robot_interface.yaml, the single source of truth shared with Isaac Lab
    /// (Python reader: isaac/interface.py). Property names map to the yaml keys in snake_case.
    /// Unity only reads this file; changes go through the Isaac side and its parity tests.
    /// </summary>
    public sealed class RobotInterface
    {
        public int Version { get; set; }
        public string Robot { get; set; }
        public TimingSpec Timing { get; set; }
        public List<string> JointNames { get; set; }
        public float[] DefaultJointPos { get; set; }
        public float InitBaseHeight { get; set; }
        public ActionSpec Action { get; set; }
        public ActuatorSpec Actuator { get; set; }
        public ObservationSpec Observation { get; set; }
        public HeightScanSpec HeightScan { get; set; }
        public FrictionSpec Friction { get; set; }
        public CommandsSpec Commands { get; set; }
        public Dictionary<string, PolicySpec> Policies { get; set; }

        public int NumJoints => JointNames.Count;

        /// <summary>Editor: the repo's configs/robot_interface.yaml. Player builds: a copy in StreamingAssets.</summary>
        public static string DefaultPath =>
            Application.isEditor ? RepoPaths.InterfaceYaml : Path.Combine(Application.streamingAssetsPath, "robot_interface.yaml");

        public static RobotInterface Load(string path = null)
        {
            path ??= DefaultPath;
            var itf = Parse(File.ReadAllText(path));
            itf.Validate(path);
            return itf;
        }

        public static RobotInterface Parse(string yaml) =>
            new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .IgnoreUnmatchedProperties()  // new keys added on the Isaac side must not break Unity
                .Build()
                .Deserialize<RobotInterface>(yaml);

        /// <summary>Policy input size for a variant (flat / rough / v2), as listed under policies.</summary>
        public int ObsDim(string variant) => Policy(variant).ObsDim;

        public PolicySpec Policy(string variant) =>
            Policies.TryGetValue(variant, out var p) ? p : throw new ArgumentException($"no policies.{variant} in robot_interface.yaml");

        /// <summary>
        /// Observation groups the deployed policy consumes, in concatenation order (port of
        /// isaac/interface.py::expected_obs_groups). v1 policies have a single "policy" group without history;
        /// flat drops the rough-only terms.
        /// </summary>
        public List<(string name, int historyLength, List<ObsTermSpec> terms)> ObsGroups(string variant)
        {
            var spec = Policy(variant);
            if (spec.Groups == null)
                return new List<(string, int, List<ObsTermSpec>)>
                {
                    ("policy", 0, Observation.Terms.Where(t => variant == "rough" || !t.RoughOnly).ToList()),
                };
            return spec.Groups.Select(g => (g.Name, g.HistoryLength, g.Terms.Select(Observation.Term).ToList())).ToList();
        }

        void Validate(string path)
        {
            var errors = new List<string>();
            if (Timing == null || JointNames == null || DefaultJointPos == null || Action == null || Actuator == null || Observation == null || Policies == null)
                errors.Add("missing top-level section");
            else
            {
                if (DefaultJointPos.Length != NumJoints) errors.Add($"default_joint_pos has {DefaultJointPos.Length} values for {NumJoints} joints");
                if (Action.Dim != NumJoints) errors.Add($"action.dim {Action.Dim} != {NumJoints} joints");
                if (Math.Abs(Timing.PhysicsDt * Timing.Decimation - Timing.PolicyDt) > 1e-6f) errors.Add("timing.policy_dt != physics_dt * decimation");
            }
            if (errors.Count > 0)
                throw new InvalidDataException($"{path}: " + string.Join("; ", errors));
        }
    }

    public sealed class TimingSpec
    {
        public float PhysicsDt { get; set; }
        public int Decimation { get; set; }
        public float PolicyDt { get; set; }
        public float EpisodeLengthS { get; set; }
    }

    public sealed class ActionSpec
    {
        public string Type { get; set; }
        public int Dim { get; set; }
        public float Scale { get; set; }
        public string Offset { get; set; }
        public float[] Clip { get; set; }
    }

    /// <summary>Explicit DC motor (isaaclab.actuators.DCMotor); see <see cref="DCMotor"/>.</summary>
    public sealed class ActuatorSpec
    {
        public string Model { get; set; }
        public float Stiffness { get; set; }
        public float Damping { get; set; }
        public float EffortLimit { get; set; }
        public float SaturationEffort { get; set; }
        public float VelocityLimit { get; set; }
        public float Friction { get; set; }
    }

    public sealed class ObservationSpec
    {
        public List<ObsTermSpec> Terms { get; set; }
        public Dictionary<string, int> Dim { get; set; }

        public ObsTermSpec Term(string name) =>
            Terms.FirstOrDefault(t => t.Name == name) ?? throw new ArgumentException($"no observation term {name}");
    }

    public sealed class ObsTermSpec
    {
        public string Name { get; set; }
        public string Func { get; set; }
        public int Dim { get; set; }
        public string Frame { get; set; }
        public string Def { get; set; }
        public bool RoughOnly { get; set; }
        public float[] Clip { get; set; }
    }

    public sealed class HeightScanSpec
    {
        public float Offset { get; set; }
        public float[] SensorOffset { get; set; }
        public float[] GridSize { get; set; }
        public float Resolution { get; set; }
        public int[] GridShape { get; set; }
        public string Ordering { get; set; }
        public float UpdatePeriod { get; set; }
    }

    public sealed class FrictionSpec
    {
        public float TerrainStaticFriction { get; set; }
        public float TerrainDynamicFriction { get; set; }
        public string CombineMode { get; set; }
        public float[] V1Foot { get; set; }
        public float FootDynamicRatio { get; set; }

        /// <summary>combine_mode as Unity's enum (PhysX combine modes have the same names).</summary>
        public PhysicsMaterialCombine UnityCombineMode() =>
            Enum.TryParse<PhysicsMaterialCombine>(CombineMode, ignoreCase: true, out var mode)
                ? mode
                : throw new InvalidDataException($"friction.combine_mode '{CombineMode}' has no Unity equivalent");
    }

    public sealed class CommandsSpec
    {
        public float[] LinVelX { get; set; }
        public float[] LinVelY { get; set; }
        public float[] AngVelZ { get; set; }
        public bool HeadingCommand { get; set; }
        public float HeadingControlStiffness { get; set; }
        public float ResamplingTimeS { get; set; }
    }

    public sealed class PolicySpec
    {
        public string Task { get; set; }
        public string Deploy { get; set; }
        public int ObsDim { get; set; }
        public int[] ActorHiddenDims { get; set; }
        public string Activation { get; set; }
        public bool ObsNormalization { get; set; }
        /// <summary>Only for policies with history (v2); flat/rough use the single "policy" group.</summary>
        public List<ObsGroupSpec> Groups { get; set; }
        public CautiousGaitSpec CautiousGait { get; set; }
    }

    public sealed class ObsGroupSpec
    {
        public string Name { get; set; }
        public int HistoryLength { get; set; }
        public List<string> Terms { get; set; }
        public Dictionary<string, string> TrainFunc { get; set; }
    }

    public sealed class CautiousGaitSpec
    {
        public float MaxLinVel { get; set; }
        public float MaxAngVel { get; set; }
    }
}
