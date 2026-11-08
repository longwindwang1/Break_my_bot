using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>
    /// Key 7: deceptive terrain. Hidden pit (the height scan sees flat ground, the feet fall in, like v2's deceptive
    /// scan in training) or breakable ice (solid and visible until the robot has stood on it long enough).
    /// Z: kind · wheel: depth · [ ]: size · click: place · X: remove the one under the cursor.
    /// </summary>
    public sealed class DeceptiveTool : DisturbanceTool
    {
        public static readonly string[] KindNames = { "Hidden pit", "Breakable ice" };

        public int Kind { get; set; }
        public float Depth { get; set; } = 0.15f;
        public float Scale { get; set; } = 1f;
        public float BreakAfter { get; set; } = 0.4f;

        int seed = 1;
        BrushCursor cursor;

        GroundFeatureType Type => Kind == 0 ? GroundFeatureType.HiddenPit : GroundFeatureType.BreakableIce;

        public override string Name => "Deceptive terrain";
        public override NeonIconKind Icon => NeonIconKind.Deceptive;
        public override string Help => "Z: hidden pit / breakable ice · wheel: depth · [ ]: size · click: place · X: remove";
        public override string Hint => Kind == 0
            ? "Click the ground: the robot's scan will see flat ground here"
            : "Click the ground: ice holds for a moment, then breaks";

        public override string Status(DisturbanceManager m) => Preview(m).Describe();

        public override List<ToolControl> BuildControls(DisturbanceManager m) => new List<ToolControl>
        {
            new ChoiceControl { Label = "Kind", Options = KindNames, Get = () => Kind, Set = i => Kind = i },
            new SliderControl
            {
                Label = "Depth", Unit = "cm", Format = "0", Step = 1f,
                Get = () => Depth * 100f, Set = v => Depth = v / 100f,
                Range = () => new Vector2(2f, m.Range(Quantity.ObstacleHeight).y * 100f),
                Training = () => new Vector2(2f, m.Limits.Range(Quantity.ObstacleHeight, false).y * 100f),
                Tooltip = "v2 trained on pits up to 16 cm, with the scan sometimes reporting flat ground",
            },
            new SliderControl { Label = "Size", Unit = "×", Format = "0.0", Step = 0.1f, Get = () => Scale, Set = v => Scale = v, Range = () => new Vector2(0.4f, 4f) },
            new SliderControl
            {
                Label = "Breaks after", Unit = "s", Format = "0.0", Step = 0.1f, Visible = () => Kind == 1,
                Get = () => BreakAfter, Set = v => BreakAfter = v, Range = () => new Vector2(0f, 3f),
                Tooltip = "Seconds the robot must stand on the ice before it gives way",
            },
            new ToggleControl
            {
                Label = "Show hidden pits to me", Get = () => m.Level != null && m.Level.RevealHidden,
                Set = v => { if (m.Level != null) m.Level.RevealHidden = v; },
                Tooltip = "Outline hidden pits for the player; the robot's scan is fooled either way",
            },
            new ButtonControl { Label = "Restore broken ice", Click = m.RestoreIce },
            new InfoControl { Text = () => Kind == 0 ? "Scan: flat ground · physics: a hole" : "Scan: solid ice · until it breaks" },
        };

        public override void OnSelect(DisturbanceManager m) { if (Application.isPlaying) cursor ??= new BrushCursor(Color.white); }
        public override void OnDeselect(DisturbanceManager m) => cursor?.Hide();

        public override void Update(DisturbanceManager m)
        {
            var keyboard = Keyboard.current;
            if (keyboard != null)
            {
                if (keyboard.zKey.wasPressedThisFrame)
                    Kind = 1 - Kind;
                if (keyboard.leftBracketKey.wasPressedThisFrame)
                    Scale = Mathf.Max(0.4f, Scale / 1.25f);
                if (keyboard.rightBracketKey.wasPressedThisFrame)
                    Scale = Mathf.Min(4f, Scale * 1.25f);
                if (keyboard.xKey.wasPressedThisFrame && m.MouseOnGround(out var under))
                    m.RemoveObstacleAt(under.point);
            }
            var s = m.Scroll();
            if (s != 0)
                Depth = Mathf.Clamp(Depth + 0.01f * s, 0.02f, m.Range(Quantity.ObstacleHeight).y);

            if (m.Level == null || !m.MouseOnGround(out var hit))
            {
                cursor?.Hide();
                return;
            }
            var f = Preview(m);
            cursor?.ShowRect(m.Level.Snap(hit.point), f.size, f.yawDeg, Kind == 0 ? new Color(1f, 0.17f, 0.84f) : new Color(0.45f, 0.8f, 1f));
            if (m.WorldClick)
            {
                f.seed = seed++;
                m.PlaceObstacle(f, hit.point);
            }
        }

        public GroundFeature Preview(DisturbanceManager m)
        {
            var camYaw = m.ViewCamera != null ? m.ViewCamera.transform.eulerAngles.y : 0f;
            var baseSize = Kind == 0 ? new Vector2(0.8f, 0.8f) : new Vector2(1.2f, 1.2f);
            return new GroundFeature
            {
                type = Type,
                height = Depth,
                size = baseSize * Scale,
                yawDeg = Mathf.Round(camYaw / 15f) * 15f,
                breakAfter = BreakAfter,
            };
        }
    }
}
