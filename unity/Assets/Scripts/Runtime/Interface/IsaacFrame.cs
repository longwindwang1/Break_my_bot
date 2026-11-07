using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Conversions between Unity coordinates (left-handed, x right, y up, z forward) and the Isaac Lab / URDF
    /// convention (right-handed, x forward, y left, z up). Positions follow the URDF importer:
    /// unity = (-y, z, x)_isaac. That map is a reflection, so axial vectors (angular velocity, quaternion
    /// vector parts) pick up an extra minus sign. See docs/unity_handoff.md §6.
    /// </summary>
    public static class IsaacFrame
    {
        /// <summary>Polar vector (position, linear velocity, gravity) Unity → Isaac.</summary>
        public static Vector3 PolarToIsaac(Vector3 u) => new Vector3(u.z, -u.x, u.y);

        /// <summary>Polar vector Isaac → Unity.</summary>
        public static Vector3 PolarFromIsaac(Vector3 i) => new Vector3(-i.y, i.z, i.x);

        /// <summary>Axial vector (angular velocity) Unity → Isaac.</summary>
        public static Vector3 AxialToIsaac(Vector3 u) => new Vector3(-u.z, u.x, -u.y);

        /// <summary>Axial vector Isaac → Unity.</summary>
        public static Vector3 AxialFromIsaac(Vector3 i) => new Vector3(i.y, -i.z, -i.x);

        /// <summary>Isaac quaternion (w, x, y, z) → Unity rotation.</summary>
        public static Quaternion QuatFromIsaac(float w, float x, float y, float z)
        {
            var v = AxialFromIsaac(new Vector3(x, y, z));
            return new Quaternion(v.x, v.y, v.z, w);
        }

        /// <summary>Unity rotation → Isaac quaternion as (w, x, y, z).</summary>
        public static Vector4 QuatToIsaac(Quaternion q)
        {
            var v = AxialToIsaac(new Vector3(q.x, q.y, q.z));
            return new Vector4(q.w, v.x, v.y, v.z);
        }

        /// <summary>World polar vector (Unity) expressed in the base frame, in Isaac axes: R_base^T v.</summary>
        public static Vector3 PolarInBase(Quaternion baseRotation, Vector3 worldUnity) =>
            PolarToIsaac(Quaternion.Inverse(baseRotation) * worldUnity);

        /// <summary>World axial vector (Unity) expressed in the base frame, in Isaac axes: R_base^T ω.</summary>
        public static Vector3 AxialInBase(Quaternion baseRotation, Vector3 worldUnity) =>
            AxialToIsaac(Quaternion.Inverse(baseRotation) * worldUnity);

        /// <summary>Unity world "down" (unit gravity direction); in Isaac it is (0, 0, -1).</summary>
        public static readonly Vector3 GravityDirection = Vector3.down;
    }
}
