using System.Collections.Generic;
using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Key 3: throw a ball at the point under the cursor (a ballistic arc from the camera). Its momentum m·v is capped
    /// by the active limit by lowering the speed. Balls are ordinary rigid bodies on the Default layer: they hit the
    /// robot, the height scan never sees them, and they disappear after <see cref="DisturbanceManager.BallLifetime"/>.
    /// </summary>
    public sealed class BallTool : DisturbanceTool
    {
        public float Speed { get; set; } = 10f;
        public float Mass { get; set; } = 0.5f;

        BrushCursor cursor;

        public override string Name => "Ball";
        public override NeonIconKind Icon => NeonIconKind.Ball;
        public override string Help => "click: throw at the cursor · wheel: speed · Shift+wheel: mass";
        public override string Hint => "Click a target: the ball flies there from the camera";

        /// <summary>Momentum the next ball would have after the cap.</summary>
        public float Momentum(DisturbanceManager m) => m.Limit(Quantity.BallMomentum, Mass * Speed, out _);

        public override string Status(DisturbanceManager m) => $"{Mass:F1} kg at {Speed:F0} m/s, p {Momentum(m):F1} N·s";

        public override List<ToolControl> BuildControls(DisturbanceManager m) => new List<ToolControl>
        {
            new SliderControl
            {
                Label = "Speed", Unit = "m/s", Format = "0", Step = 1f,
                Get = () => Speed, Set = v => Speed = v,
                Range = () => new Vector2(2f, Mathf.Min(40f, m.Range(Quantity.BallMomentum).y / Mass)),
                Training = () => new Vector2(2f, m.Limits.Range(Quantity.BallMomentum, false).y / Mass),
                Tooltip = "Capped so that momentum m·v stays within the limit",
            },
            new SliderControl
            {
                Label = "Mass", Unit = "kg", Format = "0.0", Step = 0.1f,
                Get = () => Mass, Set = v => Mass = v, Range = () => new Vector2(0.1f, 5f),
                Tooltip = "Heavier balls are also bigger",
            },
            new InfoControl
            {
                Text = () =>
                {
                    var p = Momentum(m);
                    m.Limits.Clamp(Quantity.BallMomentum, p, false, out var beyond);
                    return $"Momentum {p:F1} N·s" + (beyond ? " · beyond training" : "");
                },
            },
            new ButtonControl { Label = "Clear balls", Click = m.ClearBalls },
        };

        public override void OnSelect(DisturbanceManager m) { if (Application.isPlaying) cursor ??= new BrushCursor(Color.white); }
        public override void OnDeselect(DisturbanceManager m) => cursor?.Hide();

        public override void Update(DisturbanceManager m)
        {
            var steps = m.Scroll();
            if (steps != 0)
            {
                if (DisturbanceManager.Shift)
                    Mass = Mathf.Clamp(Mathf.Round((Mass + 0.1f * steps) * 10f) / 10f, 0.1f, 5f);
                else
                    Speed = Mathf.Clamp(Speed + steps, 2f, Mathf.Min(40f, m.Range(Quantity.BallMomentum).y / Mass));
            }
            if (!m.MouseRay(out var ray) || m.ViewCamera == null)
            {
                cursor?.Hide();
                return;
            }
            if (!m.Controller.Physics.Raycast(ray.origin, ray.direction, out var hit, 500f, ~0, QueryTriggerInteraction.Ignore))
            {
                cursor?.Hide();
                return;
            }
            cursor?.Show(hit.point, DisturbanceManager.BallRadius(Mass), new Color(1f, 0.6f, 0.1f));
            if (m.WorldClick)
            {
                var origin = m.ViewCamera.transform.position + ray.direction * 0.5f;
                m.ThrowBall(origin, DisturbanceManager.AimVelocity(origin, hit.point, Speed), Mass);
            }
        }
    }

    /// <summary>A thrown ball: reports its first hit on the robot to the manager's log.</summary>
    public sealed class BallProjectile : MonoBehaviour
    {
        public DisturbanceManager Manager { get; set; }
        public float Mass { get; set; }
        public bool HitRobot { get; private set; }

        void OnCollisionEnter(Collision c)
        {
            if (HitRobot || Manager == null || c.collider.gameObject.layer != LayerMask.NameToLayer(SimSettings.RobotLayer))
                return;
            HitRobot = true;
            var link = c.collider.GetComponentInParent<ArticulationBody>();
            var speed = c.relativeVelocity.magnitude;
            Manager.Record("ball", $"hit {(link != null ? link.name : c.collider.name)} at {speed:F1} m/s ({Mass * speed:F1} N·s)", false);
        }
    }
}
