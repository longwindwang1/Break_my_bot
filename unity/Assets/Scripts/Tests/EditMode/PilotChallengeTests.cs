using System.IO;
using System.Linq;
using BMB.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace BMB.Tests
{
    /// <summary>Claude vs the player: challenge rounds, scoring, results file, turn-based time, the AI PILOT card.</summary>
    public class PilotChallengeTests
    {
        HeadlessWorld world;
        string resultsPath;

        [SetUp]
        public void SetUp() => resultsPath = Path.Combine(Path.GetTempPath(), $"bmb_pilot_{System.Guid.NewGuid():N}.jsonl");

        [TearDown]
        public void TearDown()
        {
            world?.Dispose();
            Time.timeScale = 1f;
            if (File.Exists(resultsPath))
                File.Delete(resultsPath);
        }

        (Go2Controller robot, DisturbanceManager manager, ClaudePilot pilot, PilotChallenge challenge) Setup(bool level = false)
        {
            world = HeadlessWorld.Create(flatGround: !level);
            var ground = level ? world.AddLevelGround(new Vector2(30f, 30f)) : null;
            var robot = world.AddRobot("v2", Vector3.zero, worldModel: true);
            var manager = world.AddDisturbances(robot, level: ground);
            var go = world.Add(new GameObject("Pilot"));
            var pilot = go.AddComponent<ClaudePilot>();
            pilot.Configure(robot, manager);
            var challenge = go.AddComponent<PilotChallenge>();
            challenge.Configure(pilot);
            challenge.ResultsPath = resultsPath;
            world.Run(0.5f);
            return (robot, manager, pilot, challenge);
        }

        PilotActionResult Wait(PilotToolbox tools, PilotActionResult issued, float maxWait)
        {
            var w = tools.BeginWait(maxWait);
            for (var i = 0; i < 20000 && !tools.Settled(w); i++)
                world.Step();
            return tools.Conclude(w, issued);
        }

        [Test]
        public void Reaching_the_goal_wins_the_round_and_writes_a_result()
        {
            var (robot, _, pilot, challenge) = Setup();
            var status = pilot.Toolbox.StartChallenge(3f, 0f, null, 30f, false, out var error);
            Assert.IsNull(error);
            Assert.IsTrue(status.running);
            Assert.AreEqual(3f, status.distance, 0.1f);
            Assert.IsTrue(pilot.Engaged, "the round hands the robot to the pilot");
            var obs = pilot.Toolbox.Observe();
            Assert.IsTrue(obs.challenge_running);
            Assert.AreEqual(30f, obs.time_left_s, 0.1f);

            var issued = pilot.Toolbox.WalkTo(3f, 0f, 0.6f, "straight for it");
            var result = Wait(pilot.Toolbox, issued, 30f);
            Assert.AreEqual("goal_reached", result.outcome, result.detail);
            var r = challenge.LastResult;
            Assert.IsTrue(r.success);
            Assert.AreEqual("reached", r.outcome);
            Assert.AreEqual(100f, r.score);
            Assert.Less(r.final_distance, PilotChallenge.GoalRadius + 0.02f);
            Assert.AreEqual(1, r.orders);
            Assert.AreEqual(0, r.falls);
            Assert.Less(r.time_s, 10f);
            Assert.IsFalse(challenge.Running);

            var lines = File.ReadAllLines(resultsPath);
            Assert.AreEqual(1, lines.Length);
            var saved = JsonUtility.FromJson<PilotChallengeResult>(lines[0]);
            Assert.AreEqual(r.time_s, saved.time_s);
            Assert.AreEqual("v2", saved.policy);
            Debug.Log($"[BMB] Challenge result line: {lines[0]}");
        }

        [Test]
        public void Timeout_scores_progress_and_counts_disturbances()
        {
            var (robot, manager, pilot, challenge) = Setup();
            challenge.Begin(new Vector2(10f, 0f), timeLimit: 10f);
            pilot.Toolbox.WalkTo(10f, 0f, 0.6f);
            world.Run(3f);
            manager.Push(new Vector2(0f, 0.3f));
            manager.SetPayload(2f, Vector3.zero);
            var w = pilot.Toolbox.BeginWait(30f);
            var steps = 0;
            while (challenge.Running && steps++ < 4000)
                world.Step();
            var r = challenge.LastResult;
            Debug.Log($"[BMB] Timeout round: {JsonUtility.ToJson(r)}");
            Assert.AreEqual("timeout", r.outcome);
            Assert.IsFalse(r.success);
            Assert.AreEqual(10f, r.time_s, 0.05f);
            Assert.AreEqual(2, r.disturbances);
            Assert.AreEqual(challenge.Score(r.final_distance), r.score, 0.1f);
            Assert.That(r.score, Is.InRange(30f, 90f), "about 0.5 m/s for 10 s of 10 m");
            Assert.AreEqual("timeout", pilot.Toolbox.Conclude(w, null).outcome, "the waiting pilot hears about it");
        }

        [Test]
        public void Random_goals_are_seeded_and_on_free_ground()
        {
            var (robot, manager, pilot, challenge) = Setup(level: true);
            manager.PlaceObstacle(new GroundFeature { type = GroundFeatureType.Box, size = new Vector2(30f, 3f), height = 0.1f }, new Vector3(0f, 0f, 8f));
            var a = challenge.Begin(seed: 7);
            challenge.Cancel();
            var b = challenge.Begin(seed: 7);
            Assert.AreEqual(a.goal_x, b.goal_x);
            Assert.AreEqual(a.goal_y, b.goal_y);
            for (var seed = 0; seed < 20; seed++)
            {
                var s = challenge.Begin(seed: seed);
                Assert.That(s.distance, Is.InRange(PilotChallenge.MinGoalDistance - 0.2f, PilotChallenge.MaxGoalDistance + 0.2f));
                var world3 = IsaacFrame.PolarFromIsaac(new Vector3(s.goal_x, s.goal_y, 0f));
                Assert.IsNull(manager.Level.FeatureAt(world3), $"seed {seed}: goal not on an obstacle");
                Assert.Less(Mathf.Abs(world3.x), 13f);
                Assert.Less(Mathf.Abs(world3.z), 13f);
            }
            Assert.AreEqual("cancelled", challenge.LastResult.outcome);
            Assert.IsNull(pilot.Toolbox.StartChallenge(3f, null, null, 30f, false, out var error));
            StringAssert.Contains("both", error);
            Assert.IsNull(pilot.Toolbox.StartChallenge(50f, 0f, null, 30f, false, out error));
            StringAssert.Contains("outside", error);
        }

        [Test]
        public void Turn_based_rounds_pause_while_the_pilot_decides()
        {
            var (_, _, pilot, challenge) = Setup();
            challenge.Begin(new Vector2(4f, 0f), turnBased: true);
            challenge.UpdateTimeScale();
            Assert.AreEqual(0f, Time.timeScale, "no order yet: paused");
            pilot.Toolbox.WalkTo(4f, 0f);
            challenge.UpdateTimeScale();
            Assert.AreEqual(1f, Time.timeScale, "an order runs");
            pilot.Toolbox.Stop();
            challenge.UpdateTimeScale();
            Assert.AreEqual(0f, Time.timeScale);
            challenge.Cancel();
            Assert.AreEqual(1f, Time.timeScale, "back to normal after the round");
        }

        [Test]
        public void Pilot_card_shows_order_note_goal_and_flags()
        {
            var (robot, manager, pilot, challenge) = Setup();
            var root = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(PlaygroundScene.HudPath).CloneTree();
            var commands = world.Add(new GameObject("Commands")).AddComponent<CommandSource>();
            var ui = world.Add(new GameObject("UI")).AddComponent<GameUI>();
            ui.Bind(root, manager, commands, null, pilot);
            ui.Refresh();
            Assert.AreEqual(DisplayStyle.None, root.Q("pilot-card").style.display.value, "hidden until the pilot is used");

            challenge.Begin(new Vector2(5f, 0f), timeLimit: 60f);
            pilot.Toolbox.WalkTo(5f, 0f, 0.6f, "straight ahead, nothing in the way");
            world.Run(1f);
            ui.Refresh();
            Assert.AreEqual(DisplayStyle.Flex, root.Q("pilot-card").style.display.value);
            Assert.AreEqual("CLAUDE DRIVING", root.Q<Label>("pilot-mode").text);
            StringAssert.Contains("walk to (5.0, 0.0)", root.Q<Label>("pilot-order").text);
            StringAssert.Contains("nothing in the way", root.Q<Label>("pilot-note").text);
            StringAssert.Contains(" m · ", root.Q<Label>("pilot-goal").text);
            StringAssert.EndsWith(" s", root.Q<Label>("pilot-time").text);
            var buttons = root.Q("pilot-buttons").Query<Button>().ToList();
            Assert.AreEqual(3, buttons.Count);
            Assert.IsTrue(buttons[1].ClassListContains("on"), "veto on by default");

            pilot.Guard = false;
            pilot.Release();
            ui.Refresh();
            Assert.AreEqual("PLAYER DRIVING", root.Q<Label>("pilot-mode").text);
            Assert.IsFalse(buttons[2].ClassListContains("on"));
            Assert.IsTrue(root.Q("help-columns").Query<Label>().ToList().Any(l => l.text == "P"), "help lists the P key");
        }
    }
}
