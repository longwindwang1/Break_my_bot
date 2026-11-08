using System;
using UnityEngine;

namespace BMB
{
    /// <summary>Quantities a disturbance can set; each has a training range and a hard-mode range.</summary>
    public enum Quantity
    {
        /// <summary>Push Δv per horizontal axis (m/s), like Isaac's push velocity_range per axis.</summary>
        PushAxis,
        /// <summary>Effective foot–ground friction μ.</summary>
        Friction,
        /// <summary>Mass added to the base (kg).</summary>
        Payload,
        /// <summary>Base centre-of-mass offset, forward / lateral (m).</summary>
        ComXY,
        /// <summary>Base centre-of-mass offset, vertical (m).</summary>
        ComZ,
        /// <summary>Height of a box or depth of a pit (m).</summary>
        ObstacleHeight,
        /// <summary>Stair step height (m).</summary>
        StairStep,
        /// <summary>Slope angle (rad).</summary>
        Slope,
        /// <summary>Drag (spring) force on a link (N).</summary>
        DragForce,
        /// <summary>Momentum of a thrown ball (N·s).</summary>
        BallMomentum,
    }

    /// <summary>
    /// How hard the player may disturb the robot. "Training" = what the v2 policy saw in training
    /// (isaac/bmb_tasks/go2_v2_env_cfg.py EventCfg and terrain), so it should cope; "hard" allows more and every value
    /// outside the training range is flagged in the HUD and the event log. Sustained forces and thrown objects were
    /// not part of training at all; their training limits are kept small (≈ the push range for a 15 kg robot).
    /// </summary>
    [CreateAssetMenu(menuName = "BMB/Disturbance Limits", fileName = "DisturbanceLimits")]
    public sealed class DisturbanceLimits : ScriptableObject
    {
        [Serializable]
        public struct LimitSet
        {
            [Tooltip("push Δv per axis (m/s)")] public float pushAxis;
            [Tooltip("foot–ground μ (min, max)")] public Vector2 friction;
            [Tooltip("payload (min, max) kg")] public Vector2 payload;
            [Tooltip("|COM offset| forward/lateral (m)")] public float comXY;
            [Tooltip("|COM offset| vertical (m)")] public float comZ;
            [Tooltip("box height / pit depth (m)")] public float obstacleHeight;
            [Tooltip("stair step height (m)")] public float stairStep;
            [Tooltip("slope (rad)")] public float slope;
            [Tooltip("drag force (N)")] public float dragForce;
            [Tooltip("ball momentum (N·s)")] public float ballMomentum;
        }

        [Tooltip("v2 training ranges (go2_v2_env_cfg.py): push ±0.5 m/s per axis, foot μ 0.05–1.25, payload −1…+3 kg, " +
                 "COM ±5 / ±5 / ±2 cm, pits up to 16 cm, stairs up to 23 cm, slopes up to 0.4 rad")]
        public LimitSet training = new LimitSet
        {
            pushAxis = 0.5f,
            friction = new Vector2(0.05f, 1.25f),
            payload = new Vector2(-1f, 3f),
            comXY = 0.05f,
            comZ = 0.02f,
            obstacleHeight = 0.16f,
            stairStep = 0.23f,
            slope = 0.4f,
            dragForce = 20f,
            ballMomentum = 7.5f,
        };

        public LimitSet hard = new LimitSet
        {
            pushAxis = 2f,
            friction = new Vector2(0f, 2f),
            payload = new Vector2(-2f, 8f),
            comXY = 0.1f,
            comZ = 0.05f,
            obstacleHeight = 0.4f,
            stairStep = 0.35f,
            slope = 0.61f,  // 35°
            dragForce = 300f,
            ballMomentum = 40f,
        };

        public static DisturbanceLimits CreateDefault() => CreateInstance<DisturbanceLimits>();

        /// <summary>(min, max) of a quantity in training or hard mode.</summary>
        public Vector2 Range(Quantity q, bool hardMode)
        {
            var s = hardMode ? hard : training;
            return q switch
            {
                Quantity.PushAxis => new Vector2(-s.pushAxis, s.pushAxis),
                Quantity.Friction => s.friction,
                Quantity.Payload => s.payload,
                Quantity.ComXY => new Vector2(-s.comXY, s.comXY),
                Quantity.ComZ => new Vector2(-s.comZ, s.comZ),
                Quantity.ObstacleHeight => new Vector2(0f, s.obstacleHeight),
                Quantity.StairStep => new Vector2(0f, s.stairStep),
                Quantity.Slope => new Vector2(0f, s.slope),
                Quantity.DragForce => new Vector2(0f, s.dragForce),
                Quantity.BallMomentum => new Vector2(0f, s.ballMomentum),
                _ => throw new ArgumentOutOfRangeException(nameof(q)),
            };
        }

        /// <summary>Clamps to the active mode's range; <paramref name="outsideTraining"/> tells whether the result
        /// lies outside the training range (only possible in hard mode).</summary>
        public float Clamp(Quantity q, float value, bool hardMode, out bool outsideTraining)
        {
            var r = Range(q, hardMode);
            var v = Mathf.Clamp(value, r.x, r.y);
            var t = Range(q, false);
            outsideTraining = v < t.x - 1e-6f || v > t.y + 1e-6f;
            return v;
        }
    }
}
