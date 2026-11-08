using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Isaac Lab ray-caster height scan (robot_interface.yaml height_scan, docs/unity_handoff.md §5.6).
    /// Grid of rays in the base's yaw frame (roll / pitch ignored), x fastest, starting sensor_offset above the base
    /// and cast straight down onto the Terrain layer only. Produces the raw term base_z − hit_z − offset; the
    /// observation builder applies the clip. A missed ray gives −∞, which Isaac's clip turns into −1.
    /// </summary>
    public sealed class HeightScanner
    {
        const float MaxDistance = 1000f;

        readonly HeightScanSpec spec;
        readonly Vector2[] offsets;  // Isaac (x forward, y left) offsets in the yaw frame
        readonly int terrainMask;
        readonly Vector3[] origins;
        PhysicsScene physics;

        public int Count => offsets.Length;
        /// <summary>Unity world hit points of the last scan (NaN where a ray missed).</summary>
        public Vector3[] Hits { get; }

        /// <summary>
        /// Deceptive scan like v2 training's height_scan_deceptive: every ray reports the median reading, i.e. flat
        /// ground at the robot's current level (used to reproduce Isaac's pits_deceived evaluation).
        /// </summary>
        public bool Deceive { get; set; }
        float[] sorted;

        public HeightScanner(HeightScanSpec spec, PhysicsScene physics, int terrainMask)
        {
            this.spec = spec;
            this.physics = physics;
            this.terrainMask = terrainMask;
            int nx = spec.GridShape[0], ny = spec.GridShape[1];
            offsets = new Vector2[nx * ny];
            for (var k = 0; k < offsets.Length; k++)  // ordering "xy": x varies fastest
                offsets[k] = new Vector2(-spec.GridSize[0] / 2f + spec.Resolution * (k % nx), -spec.GridSize[1] / 2f + spec.Resolution * (k / nx));
            origins = new Vector3[offsets.Length];
            Hits = new Vector3[offsets.Length];
        }

        /// <summary>Ray origins in Unity world space for a base pose (Unity frame).</summary>
        public void RayOrigins(Vector3 basePosition, Quaternion baseRotation, Vector3[] originsOut)
        {
            // Yaw of the base in Isaac axes: heading of its x (forward) axis projected on the ground.
            var forward = IsaacFrame.PolarToIsaac(baseRotation * Vector3.forward);
            var yaw = Mathf.Atan2(forward.y, forward.x);
            float c = Mathf.Cos(yaw), s = Mathf.Sin(yaw);
            var sensor = new Vector2(spec.SensorOffset[0], spec.SensorOffset[1]);
            for (var k = 0; k < offsets.Length; k++)
            {
                var o = offsets[k] + sensor;
                var isaacOffset = new Vector3(c * o.x - s * o.y, s * o.x + c * o.y, spec.SensorOffset[2]);
                originsOut[k] = basePosition + IsaacFrame.PolarFromIsaac(isaacOffset);
            }
        }

        /// <summary>rawOut[k] = base_z − hit_z − offset (unclipped), base pose in Unity frame.</summary>
        public void Scan(Vector3 basePosition, Quaternion baseRotation, float[] rawOut)
        {
            RayOrigins(basePosition, baseRotation, origins);
            for (var k = 0; k < origins.Length; k++)
            {
                if (physics.Raycast(origins[k], Vector3.down, out var hit, MaxDistance, terrainMask, QueryTriggerInteraction.Ignore))
                {
                    rawOut[k] = basePosition.y - hit.point.y - spec.Offset;
                    Hits[k] = hit.point;
                }
                else
                {
                    rawOut[k] = float.NegativeInfinity;
                    Hits[k] = new Vector3(float.NaN, float.NaN, float.NaN);
                }
            }
            if (Deceive)
            {
                // torch.median: the lower of the two middle values for an even count.
                sorted ??= new float[rawOut.Length];
                System.Array.Copy(rawOut, sorted, rawOut.Length);
                System.Array.Sort(sorted);
                var median = sorted[(sorted.Length - 1) / 2];
                for (var k = 0; k < rawOut.Length; k++)
                    rawOut[k] = median;
            }
        }
    }
}
