using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Base pose bookkeeping for imagined (world-model) rollouts. disp_mean is (dx, dy, dz, dyaw) of the root link from
    /// t to t+1 in the yaw-only frame at t, Isaac axes (x forward, y left, z up). Yaw is the Isaac heading
    /// (<see cref="CommandSampler.Heading"/>); positions are Unity world coordinates.
    /// </summary>
    public static class WmPose
    {
        public static float Yaw(Quaternion unityRotation) => CommandSampler.Heading(unityRotation);

        /// <summary>Moves a base pose by one predicted displacement.</summary>
        public static void Integrate(ref Vector3 positionUnity, ref float yawIsaac, float[] disp)
        {
            float c = Mathf.Cos(yawIsaac), s = Mathf.Sin(yawIsaac);
            var dIsaac = new Vector3(c * disp[0] - s * disp[1], s * disp[0] + c * disp[1], disp[2]);
            positionUnity += IsaacFrame.PolarFromIsaac(dIsaac);
            yawIsaac = Mathf.Repeat(yawIsaac + disp[3] + Mathf.PI, 2f * Mathf.PI) - Mathf.PI;
        }

        /// <summary>
        /// Unity base rotation from the Isaac heading and projected_gravity (gravity direction in the base frame,
        /// Isaac axes): R = Rz(yaw) · T with T the smallest rotation taking g_base to (0, 0, −1).
        /// </summary>
        public static Quaternion Rotation(float yawIsaac, Vector3 projectedGravityIsaac)
        {
            // Quaternion algebra is handedness-free: compose in Isaac axes, then convert.
            var g = projectedGravityIsaac.sqrMagnitude > 1e-8f ? projectedGravityIsaac.normalized : new Vector3(0f, 0f, -1f);
            var tilt = Quaternion.FromToRotation(g, new Vector3(0f, 0f, -1f));
            var q = Quaternion.AngleAxis(yawIsaac * Mathf.Rad2Deg, new Vector3(0f, 0f, 1f)) * tilt;
            return IsaacFrame.QuatFromIsaac(q.w, q.x, q.y, q.z);
        }
    }
}
