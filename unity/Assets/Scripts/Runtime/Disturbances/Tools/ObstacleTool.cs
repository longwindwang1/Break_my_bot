using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>
    /// Key 6: place obstacles on the LevelGround. Z / Shift+Z: kind (block, pit, ramp, stairs, rough, Isaac tiles) ·
    /// wheel: size parameter (Shift+wheel: stairs count) · [ ]: footprint · , .: rotate · click: place ·
    /// X: remove the obstacle under the cursor · Shift+X: remove all. Placement faces away from the camera.
    /// </summary>
    public sealed class ObstacleTool : DisturbanceTool
    {
        sealed class Kind
        {
            public string Name;
            public GroundFeatureType Type;
            public IsaacTileType Tile;
        }

        static readonly List<Kind> Kinds = BuildKinds();

        static List<Kind> BuildKinds()
        {
            var kinds = new List<Kind>
            {
                new Kind { Name = "block", Type = GroundFeatureType.Box },
                new Kind { Name = "pit", Type = GroundFeatureType.Pit },
                new Kind { Name = "ramp", Type = GroundFeatureType.Ramp },
                new Kind { Name = "stairs", Type = GroundFeatureType.Stairs },
                new Kind { Name = "rough patch", Type = GroundFeatureType.Rough },
            };
            foreach (IsaacTileType t in System.Enum.GetValues(typeof(IsaacTileType)))
                kinds.Add(new Kind { Name = $"Isaac tile: {t}", Type = GroundFeatureType.IsaacTile, Tile = t });
            return kinds;
        }

        int kind;
        float boxHeight = 0.1f, pitDepth = 0.1f, stepHeight = 0.1f, roughHeight = 0.06f, slopeDeg = 15f, rampRise = 0.3f, difficulty = 0.5f;
        int steps = 3;
        float scale = 1f;
        float yawOffset;
        int seed = 1;
        BrushCursor cursor;

        public override string Name => "Obstacle";
        public override string Help => "Z/Shift+Z: kind · wheel: size (Shift: stairs count) · [ ]: footprint · , .: rotate · click: place · X: remove · Shift+X: remove all";

        Kind Current => Kinds[kind];

        public override string Status(DisturbanceManager m)
        {
            var f = Preview(m);
            var outside = OutsideTraining(m, f);
            var text = $"{Current.Name}: {f.Describe()}, footprint {f.size.x:F1}×{f.size.y:F1} m";
            return outside ? $"<color=#ff5a4a>{text} (beyond training)</color>" : text;
        }

        public override void OnSelect(DisturbanceManager m) => cursor ??= new BrushCursor(Color.white);
        public override void OnDeselect(DisturbanceManager m) => cursor?.Hide();

        public override void Update(DisturbanceManager m)
        {
            var keyboard = Keyboard.current;
            if (keyboard != null)
            {
                if (keyboard.zKey.wasPressedThisFrame)
                    kind = (kind + (DisturbanceManager.Shift ? Kinds.Count - 1 : 1)) % Kinds.Count;
                if (keyboard.leftBracketKey.wasPressedThisFrame)
                    scale = Mathf.Max(0.3f, scale / 1.25f);
                if (keyboard.rightBracketKey.wasPressedThisFrame)
                    scale = Mathf.Min(6f, scale * 1.25f);
                if (keyboard.commaKey.wasPressedThisFrame)
                    yawOffset -= Current.Type == GroundFeatureType.IsaacTile ? 90f : 15f;
                if (keyboard.periodKey.wasPressedThisFrame)
                    yawOffset += Current.Type == GroundFeatureType.IsaacTile ? 90f : 15f;
                if (keyboard.xKey.wasPressedThisFrame)
                {
                    if (DisturbanceManager.Shift)
                        m.ClearObstacles();
                    else if (m.MouseOnGround(out var under))
                        m.RemoveObstacleAt(under.point);
                }
            }
            AdjustParameter(m, DisturbanceManager.ScrollSteps());

            if (m.Level == null || !m.MouseOnGround(out var hit))
            {
                cursor?.Hide();
                return;
            }
            var f = Preview(m);
            var colour = OutsideTraining(m, f) ? new Color(1f, 0.35f, 0.3f) : new Color(0.4f, 1f, 0.5f);
            cursor?.ShowRect(m.Level.Snap(hit.point), f.size, f.yawDeg, colour);
            if (Mouse.current != null && Mouse.current.leftButton.wasPressedThisFrame)
            {
                f.seed = seed++;
                m.PlaceObstacle(f, hit.point);
            }
        }

        void AdjustParameter(DisturbanceManager m, int s)
        {
            if (s == 0)
                return;
            var shift = DisturbanceManager.Shift;
            switch (Current.Type)
            {
                case GroundFeatureType.Box: boxHeight = Clamp(m, Quantity.ObstacleHeight, boxHeight + 0.01f * s, 0.01f); break;
                case GroundFeatureType.Pit: pitDepth = Clamp(m, Quantity.ObstacleHeight, pitDepth + 0.01f * s, 0.01f); break;
                case GroundFeatureType.Rough: roughHeight = Clamp(m, Quantity.ObstacleHeight, roughHeight + 0.01f * s, 0.02f); break;
                case GroundFeatureType.Ramp:
                    var max = m.Range(Quantity.Slope).y * Mathf.Rad2Deg;
                    slopeDeg = Mathf.Clamp(slopeDeg + 2.5f * s, 2.5f, max);
                    break;
                case GroundFeatureType.Stairs:
                    if (shift) steps = Mathf.Clamp(steps + s, 1, 8);
                    else stepHeight = Clamp(m, Quantity.StairStep, stepHeight + 0.01f * s, 0.02f);
                    break;
                case GroundFeatureType.IsaacTile: difficulty = Mathf.Clamp01(difficulty + 0.05f * s); break;
            }
        }

        static float Clamp(DisturbanceManager m, Quantity q, float v, float min) => Mathf.Clamp(v, min, m.Range(q).y);

        /// <summary>The feature a click would place now (footprint and yaw resolved).</summary>
        GroundFeature Preview(DisturbanceManager m)
        {
            var camYaw = m.ViewCamera != null ? m.ViewCamera.transform.eulerAngles.y : 0f;
            var snap = Current.Type == GroundFeatureType.IsaacTile ? 90f : 15f;
            var f = new GroundFeature
            {
                type = Current.Type,
                tile = Current.Tile,
                yawDeg = Mathf.Round((camYaw + yawOffset) / snap) * snap,
                steps = steps,
                slopeDeg = slopeDeg,
                difficulty = difficulty,
            };
            switch (f.type)
            {
                case GroundFeatureType.Box: f.height = boxHeight; f.size = new Vector2(1f, 1f) * scale; break;
                case GroundFeatureType.Pit: f.height = pitDepth; f.size = new Vector2(0.6f, 0.6f) * scale; break;
                case GroundFeatureType.Rough: f.height = roughHeight; f.size = new Vector2(3f, 3f) * scale; break;
                case GroundFeatureType.Ramp: f.height = rampRise; f.size = new Vector2(1.5f * scale, 0f); break;
                case GroundFeatureType.Stairs: f.height = stepHeight; f.size = new Vector2(1.2f * scale, 0f); break;
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
