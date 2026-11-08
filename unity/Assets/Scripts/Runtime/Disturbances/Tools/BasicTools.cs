using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>Key 1: push the robot (Δv on the base, like Isaac's push events).</summary>
    public sealed class PushTool : DisturbanceTool
    {
        float speed = 0.5f;

        public override string Name => "Push";
        public override string Help => "left click: push along the view · Shift+click: toward the cursor · wheel: Δv";

        public override string Status(DisturbanceManager m)
        {
            var beyond = speed > m.Limits.Range(Quantity.PushAxis, false).y + 1e-4f;
            return $"Δv {speed:F1} m/s" + (beyond ? " <color=#ff5a4a>(beyond training)</color>" : "");
        }

        public override void Update(DisturbanceManager m)
        {
            speed = Mathf.Clamp(speed + 0.1f * DisturbanceManager.ScrollSteps(), 0.1f, m.Range(Quantity.PushAxis).y);
            var mouse = Mouse.current;
            if (mouse == null || !mouse.leftButton.wasPressedThisFrame)
                return;
            Vector3 dir;
            if (DisturbanceManager.Shift && m.MouseOnGround(out var hit))
                dir = hit.point - m.Controller.Body.BasePosition;
            else
                dir = m.ViewCamera.transform.forward;
            dir.y = 0f;
            if (dir.sqrMagnitude < 1e-6f)
                return;
            var isaac = IsaacFrame.PolarToIsaac(dir.normalized) * speed;
            m.Push(new Vector2(isaac.x, isaac.y));
        }
    }

    /// <summary>Key 4: payload on the base (mass and centre of mass, like v2's payload randomization).</summary>
    public sealed class PayloadTool : DisturbanceTool
    {
        float mass = 1f;
        Vector3 com;

        public override string Name => "Payload";
        public override string Help => "wheel: kg · Shift+wheel: COM forward · Alt+wheel: COM left · click/Enter: apply · Backspace: remove";

        public override string Status(DisturbanceManager m) =>
            $"{mass:+0.0;-0.0;0} kg, COM ({com.x * 100:F0}, {com.y * 100:F0}) cm  (on robot: {m.Controller.Body.AddedMass:+0.0;-0.0;0} kg)";

        public override void Update(DisturbanceManager m)
        {
            var steps = DisturbanceManager.ScrollSteps();
            var keyboard = Keyboard.current;
            var alt = keyboard != null && keyboard.altKey.isPressed;
            if (steps != 0)
            {
                if (DisturbanceManager.Shift)
                    com.x = Clamp(m, Quantity.ComXY, com.x + 0.01f * steps);
                else if (alt)
                    com.y = Clamp(m, Quantity.ComXY, com.y + 0.01f * steps);
                else
                    mass = Clamp(m, Quantity.Payload, mass + 0.5f * steps);
            }
            var apply = (Mouse.current != null && Mouse.current.leftButton.wasPressedThisFrame) || (keyboard != null && keyboard.enterKey.wasPressedThisFrame);
            if (apply)
                m.SetPayload(mass, com);
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
        static readonly float[] Levels = { 0f, 0.02f, 0.05f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1f, 1.25f, 1.5f, 2f };

        int level = 3;  // 0.1
        float radius = 1f;
        BrushCursor cursor;
        bool stroking, erasing;
        Vector3 strokeStart;

        public override string Name => "Friction";
        public override string Help => "drag: paint μ · Shift+drag: erase · wheel: μ · [ ]: brush size · G: whole ground to this μ / back to 1 · Backspace: clear zones";

        float Mu(DisturbanceManager m)
        {
            var r = m.Range(Quantity.Friction);
            return Mathf.Clamp(Levels[level], r.x, r.y);
        }

        public override string Status(DisturbanceManager m)
        {
            var mu = Mu(m);
            m.Limits.Clamp(Quantity.Friction, mu, false, out var outside);
            var text = $"μ {mu:F2}, brush {radius:F1} m";
            return outside ? $"<color=#ff5a4a>{text} (beyond training)</color>" : text;
        }

        public override void OnSelect(DisturbanceManager m) => cursor ??= new BrushCursor(Color.white);

        public override void OnDeselect(DisturbanceManager m)
        {
            cursor?.Hide();
            stroking = false;
        }

        public override void Update(DisturbanceManager m)
        {
            var steps = DisturbanceManager.ScrollSteps();
            if (steps != 0)
            {
                var r = m.Range(Quantity.Friction);
                var next = Mathf.Clamp(level + steps, 0, Levels.Length - 1);
                if (Levels[next] >= r.x - 1e-6f && Levels[next] <= r.y + 1e-6f)
                    level = next;
            }
            var keyboard = Keyboard.current;
            if (keyboard != null)
            {
                if (keyboard.leftBracketKey.wasPressedThisFrame)
                    radius = Mathf.Max(0.2f, radius / 1.25f);
                if (keyboard.rightBracketKey.wasPressedThisFrame)
                    radius = Mathf.Min(5f, radius * 1.25f);
                if (keyboard.gKey.wasPressedThisFrame)
                {
                    var global = m.Zones != null ? m.Zones.GlobalMu : 1f;
                    m.SetGlobalFriction(Mathf.Abs(global - Mu(m)) < 1e-4f ? 1f : Mu(m));
                }
                if (keyboard.backspaceKey.wasPressedThisFrame)
                    m.ClearFriction();
            }

            var mouse = Mouse.current;
            var onGround = m.MouseOnGround(out var hit);
            if (onGround)
                cursor?.Show(hit.point, radius, DisturbanceManager.Shift ? Color.white : FrictionZones.MuColour(Mu(m)) * 1.5f);
            else
                cursor?.Hide();
            if (mouse == null)
                return;

            if (mouse.leftButton.wasPressedThisFrame && onGround)
            {
                stroking = true;
                erasing = DisturbanceManager.Shift;
                strokeStart = hit.point;
            }
            if (stroking && mouse.leftButton.isPressed && onGround)
            {
                if (erasing)
                    m.EraseFriction(hit.point, radius, record: false);
                else
                    m.PaintFriction(hit.point, radius, Mu(m), record: false);
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
                    m.Limits.Clamp(Quantity.Friction, Mu(m), false, out var outside);
                    m.Record("friction", $"painted μ {Mu(m):F2}, brush {radius:F1} m from ({at.x:F1}, {at.y:F1}); {cells} cells painted", outside);
                }
            }
        }
    }
}
