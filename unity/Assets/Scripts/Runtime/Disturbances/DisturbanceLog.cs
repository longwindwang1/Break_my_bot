using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace BMB
{
    /// <summary>One disturbance or robot event. Base state in Isaac axes (x forward, y left, z up; world frame).</summary>
    public sealed class DisturbanceEvent
    {
        /// <summary>Simulated time since the controller started (s), across resets.</summary>
        public double Time;
        /// <summary>Time since the last robot reset (s).</summary>
        public float EpisodeTime;
        /// <summary>push, payload, friction, obstacle, …, fall, reset, mode.</summary>
        public string Type;
        /// <summary>Human-readable parameters, e.g. "dv=(0.40, -0.10) m/s".</summary>
        public string Detail;
        /// <summary>The applied value lies outside the v2 training range.</summary>
        public bool OutsideTraining;
        public Vector3 BasePosition;
        public Vector3 BaseVelocity;
    }

    /// <summary>
    /// Time-stamped record of everything done to the robot and of its falls. Later the ground truth for
    /// "when did the physics change" (surprisal detection delay, fall prediction lead time).
    /// </summary>
    public sealed class DisturbanceLog
    {
        readonly List<DisturbanceEvent> events = new List<DisturbanceEvent>();

        public IReadOnlyList<DisturbanceEvent> Events => events;
        public event Action<DisturbanceEvent> Added;

        public void Add(DisturbanceEvent e)
        {
            events.Add(e);
            Added?.Invoke(e);
        }

        public void Clear() => events.Clear();

        public string ToCsv()
        {
            var ci = CultureInfo.InvariantCulture;
            var sb = new StringBuilder("time_s,episode_time_s,type,outside_training,base_x,base_y,base_z,base_vx,base_vy,base_vz,detail\n");
            foreach (var e in events)
                sb.AppendLine(string.Format(ci, "{0:F3},{1:F3},{2},{3},{4:F3},{5:F3},{6:F3},{7:F3},{8:F3},{9:F3},\"{10}\"",
                    e.Time, e.EpisodeTime, e.Type, e.OutsideTraining ? 1 : 0, e.BasePosition.x, e.BasePosition.y, e.BasePosition.z,
                    e.BaseVelocity.x, e.BaseVelocity.y, e.BaseVelocity.z, e.Detail?.Replace("\"", "'")));
            return sb.ToString();
        }

        public void WriteCsv(string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, ToCsv());
        }
    }
}
