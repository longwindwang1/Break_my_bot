using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>
    /// Key 6: place obstacles on the LevelGround. Z / Shift+Z: kind (block, pit, ramp, stairs, rough, Isaac tile) ·
    /// wheel: size parameter (Shift+wheel: stairs count) · [ ]: footprint · , .: rotate · click: place ·
    /// X: remove the obstacle under the cursor · Shift+X: remove all. Placement faces away from the camera.
    /// </summary>
    public sealed class ObstacleTool : DisturbanceTool
    {
        public static readonly string[] KindNames = { "Block", "Pit", "Ramp", "Stairs", "Rough", "Isaac" };
        static readonly GroundFeatureType[] KindTypes =
        {
            GroundFeatureType.Box, GroundFeatureType.Pit, GroundFeatureType.Ramp, GroundFeatureType.Stairs, GroundFeatureType.Rough, GroundFeatureType.IsaacTile,
        };
        public static readonly string[] TileNames = { "Stairs up", "Stairs down", "Boxes", "Rough", "Slope up", "Slope down", "Pits" };

        public int Kind { get; set; }
        public IsaacTileType Tile { get; set; } = IsaacTileType.PyramidStairs;
        public float BoxHeight { get; set; } = 0.1f;
        public float PitDepth { get; set; } = 0.1f;
        public float StepHeight { get; set; } = 0.1f;
        public int Steps { get; set; } = 3;
        public float RoughHeight { get; set; } = 0.06f;
        public float SlopeDeg { get; set; } = 15f;
        public float RampRise { get; set; } = 0.3f;
        public float Difficulty { get; set; } = 0.5f;
        public float Scale { get; set; } = 1f;
        public float YawOffset { get; set; }

        int seed = 1;
        BrushCursor cursor;

        GroundFeatureType Type => KindTypes[Kind];

        public override string Name => "Obstacle";
        public override NeonIconKind Icon => NeonIconKind.Obstacle;
        public override string Help => "Z/Shift+Z: kind · wheel: size (Shift: stairs count) · [ ]: footprint · , .: rotate · click: place · X: remove · Shift+X: remove all";
        public override string Hint => "Click the ground to place · X removes the one under the cursor";

        public override string Status(DisturbanceManager m)
        {
            var f = Preview(m);
            var outside = OutsideTraining(m, f);
            var text = $"{KindNames[Kind]}: {f.Describe()}, footprint {f.size.x:F1}×{f.size.y:F1} m";
            return outside ? $"<color=#ff5a4a>{text} (beyond training)</color>" : text;
        }

        public override List<ToolControl> BuildControls(DisturbanceManager m)
        {
            Vector2 Cm(Quantity q, bool hard) => m.Limits.Range(q, hard) * 100f;
            return new List<ToolControl>
            {
                new ChoiceControl { Label = "Kind", Options = KindNames, Get = () => Kind, Set = i => Kind = i },
                new ChoiceControl
                {
                    Label = "Isaac tile", Options = TileNames, Get = () => (int)Tile, Set = i => Tile = (IsaacTileType)i,
                    Visible = () => Type == GroundFeatureType.IsaacTile,
                    Tooltip = "8 m tiles from the Go2 training terrain mix",
                },
                new SliderControl
                {
                    Label = "Height", Unit = "cm", Format = "0", Step = 1f, Visible = () => Type == GroundFeatureType.Box,
                    Get = () => BoxHeight * 100f, Set = v => BoxHeight = v / 100f,
                    Range = () => new Vector2(1f, Cm(Quantity.ObstacleHeight, m.HardMode).y), Training = () => new Vector2(1f, Cm(Quantity.ObstacleHeight, false).y),
                },
                new SliderControl
                {
                    Label = "Depth", Unit = "cm", Format = "0", Step = 1f, Visible = () => Type == GroundFeatureType.Pit,
                    Get = () => PitDepth * 100f, Set = v => PitDepth = v / 100f,
                    Range = () => new Vector2(1f, Cm(Quantity.ObstacleHeight, m.HardMode).y), Training = () => new Vector2(1f, Cm(Quantity.ObstacleHeight, false).y),
                },
                new SliderControl
                {
                    Label = "Slope", Unit = "°", Format = "0.0", Step = 2.5f, Visible = () => Type == GroundFeatureType.Ramp,
                    Get = () => SlopeDeg, Set = v => SlopeDeg = v,
                    Range = () => new Vector2(2.5f, m.Range(Quantity.Slope).y * Mathf.Rad2Deg), Training = () => new Vector2(0f, m.Limits.Range(Quantity.Slope, false).y * Mathf.Rad2Deg),
                },
                new SliderControl
                {
                    Label = "Step height", Unit = "cm", Format = "0", Step = 1f, Visible = () => Type == GroundFeatureType.Stairs,
                    Get = () => StepHeight * 100f, Set = v => StepHeight = v / 100f,
                    Range = () => new Vector2(2f, Cm(Quantity.StairStep, m.HardMode).y), Training = () => new Vector2(2f, Cm(Quantity.StairStep, false).y),
                },
                new SliderControl
                {
                    Label = "Steps", Format = "0", Step = 1f, Visible = () => Type == GroundFeatureType.Stairs,
                    Get = () => Steps, Set = v => Steps = Mathf.RoundToInt(v), Range = () => new Vector2(1f, 8f),
                },
                new SliderControl
                {
                    Label = "Bumps", Unit = "cm", Format = "0", Step = 1f, Visible = () => Type == GroundFeatureType.Rough,
                    Get = () => RoughHeight * 100f, Set = v => RoughHeight = v / 100f,
                    Range = () => new Vector2(2f, Cm(Quantity.ObstacleHeight, m.HardMode).y), Training = () => new Vector2(2f, 6f),
                    Tooltip = "Isaac's random_rough goes up to 6 cm",
                },
                new SliderControl
                {
                    Label = "Difficulty", Format = "0.00", Step = 0.05f, Visible = () => Type == GroundFeatureType.IsaacTile,
                    Get = () => Difficulty, Set = v => Difficulty = v, Range = () => new Vector2(0f, 1f),
                },
                new SliderControl
                {
                    Label = "Size", Unit = "×", Format = "0.0", Step = 0.1f, Visible = () => Type != GroundFeatureType.IsaacTile,
                    Get = () => Scale, Set = v => Scale = v, Range = () => new Vector2(0.3f, 6f),
                },
                new ButtonControl { Label = "Rotate", Icon = NeonIconKind.TurnLeft, Row = "rotate", Click = () => YawOffset -= Type == GroundFeatureType.IsaacTile ? 90f : 15f },
                new ButtonControl { Label = "Rotate", Icon = NeonIconKind.TurnRight, Row = "rotate", Click = () => YawOffset += Type == GroundFeatureType.IsaacTile ? 90f : 15f },
                new InfoControl { Text = () => { var f = Preview(m); return $"{f.size.x:F1} × {f.size.y:F1} m, {Mathf.Repeat(YawOffset, 360f):F0}° from view"; } },
                new ButtonControl { Label = "Remove all obstacles", Danger = true, Click = m.ClearObstacles },
            };
        }

        public override void OnSelect(DisturbanceManager m) => cursor ??= new BrushCursor(Color.white);
        public override void OnDeselect(DisturbanceManager m) => cursor?.Hide();

        public override void Update(DisturbanceManager m)
        {
            var keyboard = Keyboard.current;
            if (keyboard != null)
            {
                if (keyboard.zKey.wasPressedThisFrame)
                    Kind = (Kind + (DisturbanceManager.Shift ? KindNames.Length - 1 : 1)) % KindNames.Length;
                if (keyboard.leftBracketKey.wasPressedThisFrame)
                    Scale = Mathf.Max(0.3f, Scale / 1.25f);
                if (keyboard.rightBracketKey.wasPressedThisFrame)
                    Scale = Mathf.Min(6f, Scale * 1.25f);
                if (keyboard.commaKey.wasPressedThisFrame)
                    YawOffset -= Type == GroundFeatureType.IsaacTile ? 90f : 15f;
                if (keyboard.periodKey.wasPressedThisFrame)
                    YawOffset += Type == GroundFeatureType.IsaacTile ? 90f : 15f;
                if (keyboard.xKey.wasPressedThisFrame)
                {
                    if (DisturbanceManager.Shift)
                        m.ClearObstacles();
                    else if (m.MouseOnGround(out var under))
                        m.RemoveObstacleAt(under.point);
                }
            }
            AdjustParameter(m, m.Scroll());

            if (m.Level == null || !m.MouseOnGround(out var hit))
            {
                cursor?.Hide();
                return;
            }
            var f = Preview(m);
            var colour = OutsideTraining(m, f) ? new Color(1f, 0.23f, 0.36f) : new Color(0.22f, 1f, 0.53f);
            cursor?.ShowRect(m.Level.Snap(hit.point), f.size, f.yawDeg, colour);
            if (m.WorldClick)
            {
                f.seed = seed++;
                m.PlaceObstacle(f, hit.point);
            }
        }

        void AdjustParameter(DisturbanceManager m, int s)
        {
            if (s == 0)
                return;
            switch (Type)
            {
                case GroundFeatureType.Box: BoxHeight = Clamp(m, Quantity.ObstacleHeight, BoxHeight + 0.01f * s, 0.01f); break;
                case GroundFeatureType.Pit: PitDepth = Clamp(m, Quantity.ObstacleHeight, PitDepth + 0.01f * s, 0.01f); break;
                case GroundFeatureType.Rough: RoughHeight = Clamp(m, Quantity.ObstacleHeight, RoughHeight + 0.01f * s, 0.02f); break;
                case GroundFeatureType.Ramp:
                    SlopeDeg = Mathf.Clamp(SlopeDeg + 2.5f * s, 2.5f, m.Range(Quantity.Slope).y * Mathf.Rad2Deg);
                    break;
                case GroundFeatureType.Stairs:
                    if (DisturbanceManager.Shift) Steps = Mathf.Clamp(Steps + s, 1, 8);
                    else StepHeight = Clamp(m, Quantity.StairStep, StepHeight + 0.01f * s, 0.02f);
                    break;
                case GroundFeatureType.IsaacTile: Difficulty = Mathf.Clamp01(Difficulty + 0.05f * s); break;
            }
        }

        static float Clamp(DisturbanceManager m, Quantity q, float v, float min) => Mathf.Clamp(v, min, m.Range(q).y);

        /// <summary>The feature a click would place now (footprint and yaw resolved).</summary>
        public GroundFeature Preview(DisturbanceManager m)
        {
            var camYaw = m.ViewCamera != null ? m.ViewCamera.transform.eulerAngles.y : 0f;
            var snap = Type == GroundFeatureType.IsaacTile ? 90f : 15f;
            var f = new GroundFeature
            {
                type = Type,
                tile = Tile,
                yawDeg = Mathf.Round((camYaw + YawOffset) / snap) * snap,
                steps = Steps,
                slopeDeg = SlopeDeg,
                difficulty = Difficulty,
            };
            switch (f.type)
            {
                case GroundFeatureType.Box: f.height = BoxHeight; f.size = new Vector2(1f, 1f) * Scale; break;
                case GroundFeatureType.Pit: f.height = PitDepth; f.size = new Vector2(0.6f, 0.6f) * Scale; break;
                case GroundFeatureType.Rough: f.height = RoughHeight; f.size = new Vector2(3f, 3f) * Scale; break;
                case GroundFeatureType.Ramp: f.height = RampRise; f.size = new Vector2(1.5f * Scale, 0f); break;
                case GroundFeatureType.Stairs: f.height = StepHeight; f.size = new Vector2(1.2f * Scale, 0f); break;
                case GroundFeatureType.IsaacTile: f.size = new Vector2(IsaacTiles.Size, IsaacTiles.Size); break;
            }
            var length = GroundFeature.NaturalLength(f.type, f.steps, f.RampLength);
            if (length > 0f && f.type != GroundFeatureType.IsaacTile)
                f.size = new Vector2(f.size.x, length);
            return f;
        }

        static bool OutsideTraining(DisturbanceManager m, GroundFeature f)
        {
            bool outside;
            switch (f.type)
            {
                case GroundFeatureType.Box:
                case GroundFeatureType.Pit:
                case GroundFeatureType.Rough:
                    m.Limits.Clamp(Quantity.ObstacleHeight, f.height, false, out outside);
                    return outside;
                case GroundFeatureType.Stairs:
                    m.Limits.Clamp(Quantity.StairStep, f.height, false, out outside);
                    return outside;
                case GroundFeatureType.Ramp:
                    m.Limits.Clamp(Quantity.Slope, f.slopeDeg * Mathf.Deg2Rad, false, out outside);
                    return outside;
                default:
                    return false;
            }
        }
    }
}
