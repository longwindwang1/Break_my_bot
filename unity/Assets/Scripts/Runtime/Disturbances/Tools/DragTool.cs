using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>
    /// Key 2: grab a part of the robot with the mouse and pull it. A spring-damper between the grabbed point and the
    /// point under the mouse (kept at the grab distance from the camera); the force is capped by the active limit
    /// (v2 never trained with external forces; the training mark is a gentle 20 N). One event per drag.
    /// </summary>
    public sealed class DragTool : DisturbanceTool
    {
        /// <summary>Spring stiffness (N/m); the damping is critical for a 5 kg point mass.</summary>
        public float Stiffness { get; set; } = 150f;
        /// <summary>Force cap (N), itself clamped to the active limit.</summary>
        public float MaxForce { get; set; } = 20f;

        ArticulationBody link;
        Vector3 localGrab, target;
        float grabDistance, startTime, peak, impulse;
        bool outside;
        LineRenderer line;

        public bool Holding => link != null;
        public string HeldLink => link != null ? link.name : null;
        /// <summary>Force applied in the last physics step (world, N).</summary>
        public Vector3 LastForce { get; private set; }

        public override string Name => "Drag";
        public override NeonIconKind Icon => NeonIconKind.Drag;
        public override string Help => "hold left mouse on the robot and drag · wheel: max force";
        public override string Hint => Holding ? $"Pulling {HeldLink}: {LastForce.magnitude:F0} N" : "Grab any part of the robot and pull";

        public override string Status(DisturbanceManager m) => $"max {m.Limit(Quantity.DragForce, MaxForce, out _):F0} N, k {Stiffness:F0} N/m";

        public override List<ToolControl> BuildControls(DisturbanceManager m) => new List<ToolControl>
        {
            new SliderControl
            {
                Label = "Max force", Unit = "N", Format = "0", Step = 5f,
                Get = () => MaxForce, Set = v => MaxForce = v,
                Range = () => new Vector2(5f, m.Range(Quantity.DragForce).y),
                Training = () => new Vector2(5f, m.Limits.Range(Quantity.DragForce, false).y),
                Tooltip = "v2 never trained with external forces; up to 20 N counts as gentle",
            },
            new SliderControl
            {
                Label = "Stiffness", Unit = "N/m", Format = "0", Step = 10f,
                Get = () => Stiffness, Set = v => Stiffness = v, Range = () => new Vector2(20f, 600f),
                Tooltip = "How hard the spring pulls per metre of stretch",
            },
            new InfoControl { Text = () => Holding ? $"Holding {HeldLink} · {LastForce.magnitude:F0} N" : "Not holding anything" },
        };

        public override void OnDeselect(DisturbanceManager m) => Release(m);

        public override void Update(DisturbanceManager m)
        {
            var steps = m.Scroll();
            if (steps != 0)
            {
                var r = m.Range(Quantity.DragForce);
                MaxForce = Mathf.Clamp(MaxForce + 5f * steps, 5f, r.y);
            }
            var mouse = Mouse.current;
            if (mouse == null)
                return;
            if (!Holding && m.WorldClick && m.MouseOnRobot(out var hit))
                Grab(m, hit.collider.GetComponentInParent<ArticulationBody>(), hit.point, hit.distance);
            if (!Holding)
                return;
            if (!mouse.leftButton.isPressed)
            {
                Release(m);
                return;
            }
            if (m.ViewCamera != null)
            {
                var ray = m.ViewCamera.ScreenPointToRay(mouse.position.ReadValue());
                target = ray.origin + ray.direction * grabDistance;
            }
            DrawLine(m);
        }

        public override void FixedUpdate(DisturbanceManager m)
        {
            if (!Holding)
                return;
            if (link == null || !link.gameObject.activeInHierarchy)
            {
                Release(m);
                return;
            }
            var f = Pull(m, Time.fixedDeltaTime);
            peak = Mathf.Max(peak, f.magnitude);
        }

        /// <summary>Starts pulling <paramref name="body"/> at a world point toward <see cref="Target"/> (scripts, tests).</summary>
        public void Grab(DisturbanceManager m, ArticulationBody body, Vector3 worldPoint, float distanceFromCamera = 0f)
        {
            if (body == null)
                return;
            link = body;
            localGrab = body.transform.InverseTransformPoint(worldPoint);
            target = worldPoint;
            grabDistance = distanceFromCamera;
            startTime = (float)(m.Controller != null && m.Controller.Initialized ? m.Controller.TotalTime : 0f);
            peak = impulse = 0f;
            outside = false;
        }

        /// <summary>Where the grabbed point is pulled to (world).</summary>
        public Vector3 Target { get => target; set => target = value; }

        /// <summary>One physics step of pulling: spring-damper force, capped, applied through the manager.</summary>
        public Vector3 Pull(DisturbanceManager m, float dt)
        {
            var p = link.transform.TransformPoint(localGrab);
            var v = link.GetPointVelocity(p);
            var damping = 2f * Mathf.Sqrt(Stiffness * 5f);
            var wanted = Stiffness * (target - p) - damping * v;
            var cap = Mathf.Min(MaxForce, m.Range(Quantity.DragForce).y);
            if (wanted.magnitude > cap)
                wanted = wanted.normalized * cap;
            LastForce = m.ApplyDragForce(link, p, wanted, out var beyond);
            outside |= beyond;
            impulse += LastForce.magnitude * dt;
            return LastForce;
        }

        public void Release(DisturbanceManager m)
        {
            if (link != null && m.Controller != null && m.Controller.Initialized)
            {
                var duration = (float)m.Controller.TotalTime - startTime;
                m.Record("drag", $"pulled {link.name} for {duration:F1} s, peak {peak:F0} N, impulse {impulse:F1} N·s", outside);
            }
            link = null;
            LastForce = Vector3.zero;
            if (line != null)
                line.gameObject.SetActive(false);
        }

        void DrawLine(DisturbanceManager m)
        {
            if (!Application.isPlaying)
                return;
            if (line == null)
            {
                line = new GameObject("DragLine").AddComponent<LineRenderer>();
                line.positionCount = 2;
                line.widthMultiplier = 0.015f;
                line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                line.sharedMaterial = new Material(Shader.Find("Sprites/Default"));
            }
            line.gameObject.SetActive(true);
            m.Limits.Clamp(Quantity.DragForce, LastForce.magnitude, false, out var beyond);
            line.startColor = line.endColor = beyond ? new Color(1f, 0.23f, 0.36f) : new Color(0f, 0.9f, 1f);
            line.SetPosition(0, link.transform.TransformPoint(localGrab));
            line.SetPosition(1, target);
        }
    }
}
