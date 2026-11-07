using NUnit.Framework;
using UnityEngine;

namespace BMB.Tests
{
    /// <summary>
    /// Verification level 2 (docs/unity_handoff.md §8): the ONNX policy in Unity's Inference Engine reproduces the
    /// actions Isaac Lab produced on the GPU for every recorded observation. Tolerance 1e-4, the same bound CPU
    /// onnxruntime meets in export/test_parity.py::test_reference_replay.
    /// </summary>
    public class PolicyParityTests
    {
        const float Tolerance = 1e-4f;

        [OneTimeSetUp]
        public void SyncModelsFromRepo() => Editor.SyncModels.Sync();  // never test a stale copy

        [TestCase("flat")]
        [TestCase("rough")]
        [TestCase("v2")]
        public void Onnx_reproduces_isaac_actions(string variant)
        {
            var itf = RobotInterface.Load();
            var reference = ReferenceRollout.Load(variant);
            using var policy = PolicyRunner.Load(variant, itf);
            Assert.AreEqual(policy.ObsDim, reference.obs.RowSize, "obs size");
            Assert.AreEqual(policy.ActionDim, reference.actions.RowSize, "action size");

            var actions = new float[policy.ActionDim];
            float maxErr = 0f;
            int worstStep = -1, worstJoint = -1;
            for (var t = 0; t < reference.Steps; t++)
            {
                policy.Run(reference.obs.Row(t), actions);
                for (var j = 0; j < actions.Length; j++)
                {
                    var err = Mathf.Abs(actions[j] - reference.actions[t, j]);
                    if (err > maxErr) (maxErr, worstStep, worstJoint) = (err, t, j);
                }
            }

            Debug.Log($"[BMB] ONNX parity {variant}: max |Unity - Isaac| = {maxErr:E2} over {reference.Steps} steps x {policy.ActionDim} joints " +
                      $"(worst: step {worstStep}, {itf.JointNames[System.Math.Max(worstJoint, 0)]})");
            Assert.That(maxErr, Is.LessThanOrEqualTo(Tolerance));
        }
    }
}
