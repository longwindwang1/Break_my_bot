using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace BMB.Tests
{
    /// <summary>The player's speed limit (top-left HUD slider): caps manual and cruise commands, not the AI pilot.</summary>
    public class SpeedLimitTests
    {
        static CommandSource NewSource(out GameObject go)
        {
            go = new GameObject("Commands");
            return go.AddComponent<CommandSource>();
        }

        [Test]
        public void Limit_scales_the_walking_speed_and_keeps_the_turn()
        {
            var source = NewSource(out var go);
            try
            {
                // Cruise (fixed command, 0.5 m/s forward by default).
                Assert.AreEqual(0.5f, source.Compute(null, null).x, 1e-6f, "no limit by default");
                source.MaxSpeed = 0.3f;
                Assert.AreEqual(new Vector3(0.3f, 0f, 0f), source.Compute(null, null));

                // Manual: on-screen stick fully forward-right plus a left turn.
                source.SetFixed(false);
                source.UiStick = new Vector2(1f, 1f);
                source.UiTurn = 1f;
                var c = source.Compute(null, null);
                Assert.AreEqual(0.3f, new Vector2(c.x, c.y).magnitude, 1e-5f);
                Assert.AreEqual(1f, c.x / -c.y, 1e-5f, "direction kept (forward and to the right)");
                Assert.AreEqual(1f, c.z, 1e-6f, "the turn rate is not limited");

                source.MaxSpeed = float.PositiveInfinity;
                c = source.Compute(null, null);
                Assert.AreEqual(1f, c.x, 1e-6f, "unlimited again: each axis up to its range");
                Assert.AreEqual(-1f, c.y, 1e-6f);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void Top_left_slider_sets_the_limit()
        {
            using var world = HeadlessWorld.Create(flatGround: false);
            var level = world.AddLevelGround(new Vector2(10f, 10f));
            var robot = world.AddRobot("v2", Vector3.zero);
            var manager = world.AddDisturbances(robot, level: level);
            var root = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(PlaygroundScene.HudPath).CloneTree();
            var commands = world.Add(new GameObject("Commands")).AddComponent<CommandSource>();
            var ui = world.Add(new GameObject("UI")).AddComponent<GameUI>();
            ui.Bind(root, manager, commands);

            var slot = root.Q("speed-limit");
            Assert.IsNotNull(slot, "in the top bar");
            Assert.AreSame(root.Q("topbar"), slot.parent);
            var slider = slot.Q<NeonSlider>();
            Assert.IsNotNull(slider);
            ui.Refresh();
            Assert.AreEqual("1.0 m/s", slider.Q<Label>("value").text, "unlimited shows the training maximum");

            slider.Control.SetClamped(0.42f);
            Assert.AreEqual(0.4f, commands.MaxSpeed, 1e-6f, "snapped to 0.1 m/s steps");
            ui.Refresh();
            Assert.AreEqual("0.4 m/s", slider.Q<Label>("value").text);
            slider.Control.SetClamped(0f);
            Assert.AreEqual(0.1f, commands.MaxSpeed, 1e-6f, "never below 0.1 m/s");
        }
    }
}
