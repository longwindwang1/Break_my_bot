using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace BMB.Tests
{
    /// <summary>Game HUD: tool control descriptors, building the UI tree for every tool, on-screen driving.</summary>
    public class GameUiTests
    {
        static (HeadlessWorld world, DisturbanceManager m) Setup()
        {
            var world = HeadlessWorld.Create(flatGround: false);
            var level = world.AddLevelGround(new Vector2(10f, 10f));
            var robot = world.AddRobot("v2", Vector3.zero, footFriction: 1f);
            return (world, world.AddDisturbances(robot, level: level));
        }

        static T Control<T>(DisturbanceManager m, string tool, string label) where T : ToolControl =>
            m.Tools.Single(t => t.Name == tool).BuildControls(m).OfType<T>().Single(c => c.Label == label);

        [Test]
        public void Push_slider_follows_the_limits_mode()
        {
            var (world, m) = Setup();
            using (world)
            {
                var push = (PushTool)m.Tools.Single(t => t is PushTool);
                var slider = Control<SliderControl>(m, "Push", "Impulse Δv");
                slider.SetClamped(3f);
                Assert.AreEqual(0.5f, push.Speed, 1e-5f, "training range");
                Assert.IsFalse(slider.OutsideTraining(push.Speed));
                m.HardMode = true;
                slider.SetClamped(1.5f);
                Assert.AreEqual(1.5f, push.Speed, 1e-5f, "hard mode allows more");
                Assert.IsTrue(slider.OutsideTraining(push.Speed), "and flags it");
                Assert.AreEqual((0.5f - 0.1f) / (2f - 0.1f), NeonSlider.Fraction(slider.Training().y, slider.Range()), 1e-6f,
                    "training limit marker sits at 0.5 m/s on the hard-mode track");
            }
        }

        [Test]
        public void Obstacle_controls_switch_with_the_kind_and_use_display_units()
        {
            var (world, m) = Setup();
            using (world)
            {
                var tool = (ObstacleTool)m.Tools.Single(t => t is ObstacleTool);
                var controls = tool.BuildControls(m);
                var height = controls.OfType<SliderControl>().Single(c => c.Label == "Height");
                var steps = controls.OfType<SliderControl>().Single(c => c.Label == "Steps");
                tool.Kind = 0;  // block
                Assert.IsTrue(height.Visible());
                Assert.IsFalse(steps.Visible());
                height.SetClamped(40f);  // cm
                Assert.AreEqual(0.16f, tool.BoxHeight, 1e-5f, "training range: 16 cm");
                tool.Kind = 3;  // stairs
                Assert.IsFalse(height.Visible());
                Assert.IsTrue(steps.Visible());

                var kind = controls.OfType<ChoiceControl>().Single(c => c.Label == "Kind");
                kind.Set(5);
                Assert.IsTrue(controls.OfType<ChoiceControl>().Single(c => c.Label == "Isaac tile").Visible());
            }
        }

        [Test]
        public void Payload_buttons_load_and_unload_the_robot()
        {
            var (world, m) = Setup();
            using (world)
            {
                var tool = (PayloadTool)m.Tools.Single(t => t is PayloadTool);
                var controls = tool.BuildControls(m);
                controls.OfType<SliderControl>().Single(c => c.Label == "Mass").SetClamped(2f);
                controls.OfType<ButtonControl>().Single(c => c.Label == "Apply").Click();
                Assert.AreEqual(2f, m.Controller.Body.AddedMass, 1e-5f);
                controls.OfType<ButtonControl>().Single(c => c.Label == "Remove").Click();
                Assert.AreEqual(0f, m.Controller.Body.AddedMass, 1e-5f);
            }
        }

        [Test]
        public void Hud_binds_and_builds_a_card_for_every_tool()
        {
            var (world, m) = Setup();
            using (world)
            {
                var tree = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(PlaygroundScene.HudPath);
                Assert.IsNotNull(tree, PlaygroundScene.HudPath);
                var root = tree.CloneTree();
                var commands = world.Add(new GameObject("Commands")).AddComponent<CommandSource>();
                var ui = world.Add(new GameObject("UI")).AddComponent<GameUI>();
                ui.Bind(root, m, commands);

                Assert.AreEqual(m.Tools.Count, root.Query(className: "rail-btn").ToList().Count, "one rail button per tool");
                Assert.AreEqual(m.Tools.Count(t => !t.Available), root.Query(className: "locked").ToList().Count);
                foreach (var tool in m.Tools.Where(t => t.Available))
                {
                    m.Select(tool);
                    ui.Refresh();
                    var controls = root.Q("tool-controls");
                    Assert.AreEqual(1, controls.childCount, $"{tool.Name}: card shown");
                    Assert.Greater(controls[0].childCount, 1, $"{tool.Name}: card has controls");
                    Assert.AreEqual(tool.Name.ToUpperInvariant(), root.Q<Label>("tool-title").text);
                }
                m.Select(null);
                ui.Refresh();
                Assert.AreEqual(DisplayStyle.None, root.Q("tool-card").style.display.value, "no tool: no card");

                m.HardMode = true;
                ui.Refresh();
                Assert.AreEqual("HARD MODE", root.Q<Button>("mode").text);
                Assert.IsTrue(root.Q<Button>("mode").ClassListContains("hard"));
                Assert.Greater(root.Q("events").childCount, 0, "the mode change shows in the event feed");
            }
        }

        [Test]
        public void Stick_and_turn_buttons_drive_the_manual_command()
        {
            var commands = new GameObject("Commands").AddComponent<CommandSource>();
            try
            {
                var r = commands.Ranges;
                commands.SetFixed(false);
                commands.UiStick = new Vector2(0f, 1f);
                Assert.AreEqual(r.LinVelX[1], commands.Compute(null, null).x, 1e-6f, "full forward");
                commands.UiStick = new Vector2(1f, 0f);
                Assert.AreEqual(-r.LinVelY[1], commands.Compute(null, null).y, 1e-6f, "stick right = move right (Isaac −y)");
                commands.UiStick = Vector2.zero;
                commands.UiTurn = 1f;
                Assert.AreEqual(r.AngVelZ[1], commands.Compute(null, null).z, 1e-6f, "turn left = +wz");
                commands.SetFixed(true);
                commands.FixedSpeed = 0.3f;
                Assert.AreEqual(new Vector3(0.3f, 0f, 0f), commands.Compute(null, null), "cruise ignores the stick");
            }
            finally
            {
                Object.DestroyImmediate(commands.gameObject);
            }
        }
    }
}
