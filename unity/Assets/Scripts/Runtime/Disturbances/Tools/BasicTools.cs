using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>Key 1: push the robot (Δv on the base, like Isaac's push events).</summary>
    public sealed class PushTool : DisturbanceTool
    {
        public float Speed { get; set; } = 0.5f;
        /// <summary>false: along the camera's view; true: from the robot toward the clicked point.</summary>
        public bool TowardCursor { get; set; }

        public override string Name => "Push";
        public override NeonIconKind Icon => NeonIconKind.Push;
        public override string Help => "left click: push along the view · Shift+click: toward the cursor · wheel: Δv";
        public override string Hint => TowardCursor ? "Click the ground: push toward that point" : "Click anywhere: push along the view";

        public override string Status(DisturbanceManager m)
        {
            var beyond = Speed > m.Limits.Range(Quantity.PushAxis, false).y + 1e-4f;
            return $"Δv {Speed:F1} m/s" + (beyond ? " <color=#ff5a4a>(beyond training)</color>" : "");
        }

        public override List<ToolControl> BuildControls(DisturbanceManager m) => new List<ToolControl>
        {
            new SliderControl
            {
                Label = "Impulse Δv", Unit = "m/s", Format = "0.0", Step = 0.1f,
                Get = () => Speed, Set = v => Speed = v,
                Range = () => new Vector2(0.1f, m.Range(Quantity.PushAxis).y),
                Training = () => new Vector2(0.1f, m.Limits.Range(Quantity.PushAxis, false).y),
                Tooltip = "Velocity added to the base (v2 trained with ±0.5 m/s per axis)",
            },
            new ChoiceControl { Label = "Direction", Options = new[] { "View", "Cursor" }, Get = () => TowardCursor ? 1 : 0, Set = i => TowardCursor = i == 1 },
            new ButtonControl { Label = "Push now", Click = () => PushAlongView(m), Tooltip = "Push along the camera's view" },
        };

        public override void Update(DisturbanceManager m)
        {
            Speed = Mathf.Clamp(Speed + 0.1f * m.Scroll(), 0.1f, m.Range(Quantity.PushAxis).y);
            if (!m.WorldClick)
                return;
            if ((TowardCursor || DisturbanceManager.Shift) && m.MouseOnGround(out var hit))
                PushToward(m, hit.point - m.Controller.Body.BasePosition);
            else
                PushAlongView(m);
        }

        public void PushAlongView(DisturbanceManager m) => PushToward(m, m.ViewCamera != null ? m.ViewCamera.transform.forward : Vector3.forward);

        void PushToward(DisturbanceManager m, Vector3 dir)
        {
            dir.y = 0f;
            if (dir.sqrMagnitude < 1e-6f)
                return;
            var isaac = IsaacFrame.PolarToIsaac(dir.normalized) * Speed;
            m.Push(new Vector2(isaac.x, isaac.y));
        }
    }

    /// <summary>Key 4: payload on the base (mass and centre of mass, like v2's payload randomization).</summary>
    public sealed class PayloadTool : DisturbanceTool
    {
        public float Mass { get; set; } = 1f;
        /// <summary>COM offset in Isaac base axes (m): x forward, y left.</summary>
        public Vector2 Com { get; set; }

        public override string Name => "Payload";
        public override NeonIconKind Icon => NeonIconKind.Payload;
        public override string Help => "wheel: kg · Shift+wheel: COM forward · Alt+wheel: COM left · click/Enter: apply · Backspace: remove";
        public override string Hint => "Apply loads it onto the robot's back";

        public override string Status(DisturbanceManager m) =>
            $"{Mass:+0.0;-0.0;0} kg, COM ({Com.x * 100:F0}, {Com.y * 100:F0}) cm  (on robot: {m.Controller.Body.AddedMass:+0.0;-0.0;0} kg)";

        public override List<ToolControl> BuildControls(DisturbanceManager m) => new List<ToolControl>
        {
            new SliderControl
            {
                Label = "Mass", Unit = "kg", Format = "+0.0;-0.0;0", Step = 0.5f,
                Get = () => Mass, Set = v => Mass = v,
                Range = () => m.Range(Quantity.Payload), Training = () => m.Limits.Range(Quantity.Payload, false),
                Tooltip = "Added to the base (v2 trained with −1…+3 kg)",
            },
            new SliderControl
            {
                Label = "COM forward", Unit = "cm", Format = "+0;-0;0", Step = 1f,
                Get = () => Com.x * 100f, Set = v => Com = new Vector2(v / 100f, Com.y),
                Range = () => m.Range(Quantity.ComXY) * 100f, Training = () => m.Limits.Range(Quantity.ComXY, false) * 100f,
            },
            new SliderControl
            {
                Label = "COM left", Unit = "cm", Format = "+0;-0;0", Step = 1f,
                Get = () => Com.y * 100f, Set = v => Com = new Vector2(Com.x, v / 100f),
                Range = () => m.Range(Quantity.ComXY) * 100f, Training = () => m.Limits.Range(Quantity.ComXY, false) * 100f,
            },
            new ButtonControl { Label = "Apply", Row = "payload", Click = () => Apply(m) },
            new ButtonControl { Label = "Remove", Row = "payload", Danger = true, Click = () => m.SetPayload(0f, Vector3.zero) },
            new InfoControl { Text = () => m.Controller.Body.AddedMass == 0f ? "Robot carries nothing" : $"On robot: {m.Controller.Body.AddedMass:+0.0;-0.0} kg" },
        };

        void Apply(DisturbanceManager m) => m.SetPayload(Mass, new Vector3(Com.x, Com.y, 0f));

        public override void Update(DisturbanceManager m)
        {
            var steps = m.Scroll();
            var keyboard = Keyboard.current;
            var alt = keyboard != null && keyboard.altKey.isPressed;
            if (steps != 0)
            {
                if (DisturbanceManager.Shift)
                    Com = new Vector2(Clamp(m, Quantity.ComXY, Com.x + 0.01f * steps), Com.y);
                else if (alt)
                    Com = new Vector2(Com.x, Clamp(m, Quantity.ComXY, Com.y + 0.01f * steps));
                else
                    Mass = Clamp(m, Quantity.Payload, Mass + 0.5f * steps);
            }
            if (m.WorldClick || (keyboard != null && keyboard.enterKey.wasPressedThisFrame))
                Apply(m);
            if (keyboard != null && keyboard.backspaceKey.wasPressedThisFrame)
                m.SetPayload(0f, Vector3.zero);
        }

        static float Clamp(DisturbanceManager m, Quantity q, float v)
        {
            var r = m.Range(q);
            return Mathf.Clamp(v, r.x, r.y);
        }
    }

    /// <summary>Key 5: paint ground friction (ice / grip) or change it everywhere.</summary>
    public sealed class FrictionTool : DisturbanceTool
    {
        public float Mu { get; set; } = 0.1f;
        public float Radius { get; set; } = 1f;
        public bool Erase { get; set; }

        BrushCursor cursor;
        bool stroking, erasing;
        Vector3 strokeStart;

        public override string Name => "Friction";
        public override NeonIconKind Icon => NeonIconKind.Friction;
        public override string Help => "drag: paint μ · Shift+drag: erase · wheel: μ · [ ]: brush size · G: whole ground to this μ / back to 1 · Backspace: clear zones";
        public override string Hint => Erase ? "Drag on the ground to erase zones" : "Drag on the ground to paint";

        public static string Describe(float mu) => mu < 0.15f ? "ICE" : mu < 0.45f ? "SLIPPERY" : mu <= 1.05f ? "NORMAL" : "GRIP";

        float ClampedMu(DisturbanceManager m)
        {
            var r = m.Range(Quantity.Friction);
            return Mathf.Clamp(Mu, r.x, r.y);
        }

        public override string Status(DisturbanceManager m)
        {
            var mu = ClampedMu(m);
            m.Limits.Clamp(Quantity.Friction, mu, false, out var outside);
            var text = $"μ {mu:F2}, brush {Radius:F1} m";
            return outside ? $"<color=#ff5a4a>{text} (beyond training)</color>" : text;
        }

        public override List<ToolControl> BuildControls(DisturbanceManager m) => new List<ToolControl>
        {
            new SliderControl
            {
                Label = "Friction μ", Format = "0.00", Step = 0.01f,
                Get = () => Mu, Set = v => Mu = v,
                Range = () => m.Range(Quantity.Friction), Training = () => m.Limits.Range(Quantity.Friction, false),
                Tooltip = "Ground friction (v2 trained with feet at 0.05–1.25)",
            },
            new InfoControl { Text = () => $"Surface: {Describe(ClampedMu(m))}" },
            new SliderControl { Label = "Brush", Unit = "m", Format = "0.0", Step = 0.1f, Get = () => Radius, Set = v => Radius = v, Range = () => new Vector2(0.2f, 5f) },
            new ToggleControl { Label = "Eraser", Get = () => Erase, Set = v => Erase = v },
            new ButtonControl
            {
                Label = "Whole ground", Row = "global", Tooltip = "Set every unpainted spot to this μ",
                Click = () => m.SetGlobalFriction(ClampedMu(m)),
                Active = () => m.Zones != null && Mathf.Abs(m.Zones.GlobalMu - 1f) > 1e-4f,
            },
            new ButtonControl { Label = "Restore", Row = "global", Click = () => m.SetGlobalFriction(1f), Tooltip = "Whole ground back to μ 1.0" },
            new ButtonControl { Label = "Clear painted zones", Danger = true, Click = m.ClearFriction },
        };

        public override void OnSelect(DisturbanceManager m) { if (Application.isPlaying) cursor ??= new BrushCursor(Color.white); }

        public override void OnDeselect(DisturbanceManager m)
        {
            cursor?.Hide();
            stroking = false;
        }

        public override void Update(DisturbanceManager m)
        {
            var steps = m.Scroll();
            if (steps != 0)
            {
                var r = m.Range(Quantity.Friction);
                Mu = Mathf.Clamp(Mathf.Round((steps > 0 ? Mathf.Max(Mu, 0.01f) * 1.25f : Mu / 1.25f) * 100f) / 100f, r.x, r.y);
            }
            var keyboard = Keyboard.current;
            if (keyboard != null)
            {
                if (keyboard.leftBracketKey.wasPressedThisFrame)
                    Radius = Mathf.Max(0.2f, Radius / 1.25f);
                if (keyboard.rightBracketKey.wasPressedThisFrame)
                    Radius = Mathf.Min(5f, Radius * 1.25f);
                if (keyboard.gKey.wasPressedThisFrame)
                {
                    var global = m.Zones != null ? m.Zones.GlobalMu : 1f;
                    m.SetGlobalFriction(Mathf.Abs(global - ClampedMu(m)) < 1e-4f ? 1f : ClampedMu(m));
                }
                if (keyboard.backspaceKey.wasPressedThisFrame)
                    m.ClearFriction();
            }

            var mouse = Mouse.current;
            var onGround = m.MouseOnGround(out var hit);
            var eraseNow = Erase || DisturbanceManager.Shift;
            if (onGround)
                cursor?.Show(hit.point, Radius, eraseNow ? Color.white : FrictionZones.MuColour(ClampedMu(m)) * 1.5f);
            else
                cursor?.Hide();
            if (mouse == null)
                return;

            if (m.WorldClick && onGround)
            {
                stroking = true;
                erasing = eraseNow;
                strokeStart = hit.point;
            }
            if (stroking && mouse.leftButton.isPressed && onGround)
            {
                if (erasing)
                    m.EraseFriction(hit.point, Radius, record: false);
                else
                    m.PaintFriction(hit.point, Radius, ClampedMu(m), record: false);
            }
            if (stroking && mouse.leftButton.wasReleasedThisFrame)
            {
                stroking = false;
                var at = IsaacFrame.PolarToIsaac(strokeStart);
                var cells = m.Zones != null ? m.Zones.PaintedCells : 0;
                if (erasing)
                    m.Record("friction", $"erased zones from ({at.x:F1}, {at.y:F1}); {cells} cells painted", false);
                else
                {
                    m.Limits.Clamp(Quantity.Friction, ClampedMu(m), false, out var outside);
                    m.Record("friction", $"painted μ {ClampedMu(m):F2}, brush {Radius:F1} m from ({at.x:F1}, {at.y:F1}); {cells} cells painted", outside);
                }
            }
        }
    }
}
