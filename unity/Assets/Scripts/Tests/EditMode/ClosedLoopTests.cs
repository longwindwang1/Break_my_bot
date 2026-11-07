using BMB.Editor;
using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>
    /// Verification level 4 (docs/unity_handoff.md §8), headless: each policy walks in Unity physics for the length of
    /// the Isaac reference rollouts without falling and tracks the forward command. Trajectories drift from Isaac's
    /// (different engine; that gap is experiment 3), so only coarse bounds are asserted; the detailed comparison is
    /// BMB/Headless trials/Run all and write report.
    /// </summary>
    public class ClosedLoopTests
    {
        static readonly object[] Cases =
        {
            new object[] { 0 },  // flat policy, flat ground
            new object[] { 1 },  // rough policy, rough patch
            new object[] { 2 },  // v2, rough patch, μ 1.0
            new object[] { 3 },  // v2, flat ground, μ 1.0
        };

        [TestCaseSource(nameof(Cases))]
        public void Policy_walks_and_tracks_the_command(int index)
        {
            var spec = Go2Trial.Standard[index];
            var m = Go2Trial.Metrics.Of(Go2Trial.Run(spec));
            Debug.Log($"[BMB] Closed loop '{spec.Name}': falls {m.Falls}, forward speed {m.Speed:F3} m/s for {spec.Command.x:F2}, base height {m.Height:F3} m");

            Assert.AreEqual(0, m.Falls, "fell");
            Assert.That(m.Speed, Is.InRange(0.7f * spec.Command.x, 1.3f * spec.Command.x), "forward speed");
            Assert.That(m.Height, Is.GreaterThan(0.15f), "base height");
        }
    }
}
